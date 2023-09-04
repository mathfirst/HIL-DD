import torch, random, os, yaml
import torch.nn.functional as F
from torch_geometric.nn import knn_graph
from easydict import EasyDict
import utils.transforms as trans
from datetime import datetime  # we use time as a filename when creating that logfile.
import numpy as np
from typing import Optional
from torch import Tensor
from torch_scatter import scatter
from tqdm import tqdm


# https://github.com/ehoogeboom/e3_diffusion_for_molecules/blob/main/utils.py
class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ema_model, current_model):
        for ema_params, current_params in zip(ema_model.parameters(), current_model.parameters()):
            ema_weight, curr_weight = ema_params.data, current_params.data
            ema_params.data = self.update_average(ema_weight, curr_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


# https://github.com/ehoogeboom/e3_diffusion_for_molecules/blob/main/utils.py
def remove_mean(x, dim=0, return_mean=False):
    mean = torch.mean(x, dim=dim, keepdim=True)
    x = x - mean
    if return_mean:
        return x, mean
    return x


# https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/norm/mean_subtraction_norm.html#MeanSubtractionNorm
# I modified this code to let it return mean values that I need later.
class MeanSubtractionNorm(torch.nn.Module):
    r"""Applies layer normalization by subtracting the mean from the inputs
    as described in the  `"Revisiting 'Over-smoothing' in Deep GCNs"
    <https://arxiv.org/pdf/2003.13663.pdf>`_ paper

    .. math::
        \mathbf{x}_i = \mathbf{x}_i - \frac{1}{|\mathcal{V}|}
        \sum_{j \in \mathcal{V}} \mathbf{x}_j
    """

    def reset_parameters(self):
        pass

    def forward(self, x: Tensor, batch: Optional[Tensor] = None,
                dim_size: Optional[int] = None, verbose: Optional[bool] = False) -> Tensor:
        """"""
        if batch is None:
            mean = x.mean(dim=0, keepdim=True)
            if verbose:
                return x - mean, mean
            return x - mean

        mean = scatter(x, batch, dim=0, dim_size=dim_size, reduce='mean')
        if verbose:
            return x - mean[batch], mean
        return x - mean[batch]


def subtract_mean(x, batch=None, dim_size=None, verbose=False):
    if batch is None:
        mean = x.mean(dim=0, keepdim=True)
        if verbose:
            return x - mean, mean
        return x - mean

    mean = scatter(x, batch, dim=0, dim_size=dim_size, reduce='mean')
    if verbose:
        return x - mean[batch], mean
    return x - mean[batch]


def scale(x, batch=None, scale_factor=None, adaptive=True, verbose=False):
    '''
    For the adaptive mode, we aim to scale the values of each sample within the current minibatch to [-1, 1].
    Note that this is a sample-wise operation.
    If adaptive is True, we scale x sample-wise by the maximum absolute value of each sample in a minibatch.
    If adaptive is False and scale_factor is not None, we scale x using scale_factor.
    If verbose is True, we return the corresponding maximum absolute values.
    '''

    if adaptive:
        from torch_scatter import scatter
        if batch is None:
            max_values, _ = torch.max(x.abs(), dim=0, keepdim=True)
        else:
            max_values = scatter(x.abs(), batch, dim=0, reduce='max')  # get a row vector for each sample
        max_values, _ = max_values.max(dim=-1, keepdim=True)  # get row-wise maximum abs value for each sample
        x = x / max_values[batch] if batch is not None else x / max_values  # scaling
    else:
        assert scale_factor is not None, "scale_factor is not given for nonadaptive mode"
        x = x * scale_factor
        return x

    if verbose:
        return x, max_values  # only when adaptive=True
    else:
        return x


# https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/norm/instance_norm.html#InstanceNorm
def standardization(x, batch=None, eps=1.0e-15):
    '''
    This function performs sample-wise standardization.
    After standardization, each sample has a mean of 0 and an std of 1.
    '''

    if batch is None:
        means = torch.mean(x, dim=0, keepdim=True)
        stds = torch.std(x, dim=0, keepdim=True)
        x = (x - means) / (stds + eps)
        return x

    batch_size = int(batch.max()) + 1  # the number of samples in x
    from torch_geometric.utils import degree
    norm = degree(batch, batch_size, dtype=torch.int64).clamp_(
        min=1)  # the num of nodes in each sample, at least one node.
    norm = norm.view(-1, 1)
    from torch_scatter import scatter
    means = scatter(x, batch, dim=0, dim_size=batch_size, reduce='add') / norm  # get means for each sample along dim=0
    x = x - means[
        batch]  # .index_select(0, batch)    # demeaning is important for the next step, i.e. calculating the variance
    var = scatter(x * x, batch, dim=0, dim_size=batch_size, reduce='add')
    var = var / norm  # a little math, i.e. var = 1/n * (x - mean)^2 where mean is already 0.
    out = x / (var.sqrt() + eps)[batch]  # .index_select(0, batch)
    return out


def get_grad_norm(model):
    # https://discuss.pytorch.org/t/check-the-norm-of-gradients/27961/4
    total_norm = 0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def load_config(path):
    with open(path, 'r') as f:
        return EasyDict(yaml.safe_load(f))


def get_logger(name, log_filename=None):
    import logging
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # formatter = logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s')
    # formatter = logging.Formatter('[%(asctime)s-%(name)s] %(message)s')
    formatter = logging.Formatter('[%(asctime)s] %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    #     logger.addHandler(stream_handler)
    #     https://alexandra-zaharia.github.io/posts/fix-python-logger-printing-same-entry-multiple-times/
    if not logger.hasHandlers():  # check whether any handlers are already attached before adding them to the logger
        logger.addHandler(stream_handler)
        # logger.addHandler(file_handler)
    if log_filename is not None:
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_time():
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


class OnlineAveraging(object):
    def __init__(self, averaging_range=2000):
        self.span = averaging_range
        self.counter = 0
        self.mem_dict = {}

    def __call__(self, *args, **kwargs):
        if self.counter == 0:
            self.counter += 1
            self.mem_dict = kwargs
        else:
            new_values = np.array(list(kwargs.values()))
            old_values = np.array(list(self.mem_dict.values()))
            old_values = old_values * self.counter / (self.counter + 1) + new_values / (self.counter + 1)
            self.mem_dict = dict(zip(kwargs.keys(), old_values))
            self.counter += 1
            if self.counter >= self.span:
                self.counter = 0

        return self.mem_dict


class TimeStepSampler(object):
    def __init__(self, num_timesteps, sampling_method="uniform", init_weight=10.0, low=3, high=1):
        self.sampling_method = sampling_method
        self.num_timesteps = num_timesteps
        self.last_t = None
        if sampling_method == 'importance':
            print('You will sample time steps based on loss values.')
            self.weights = torch.ones((num_timesteps,), dtype=torch.float) * init_weight
            self.freq = torch.ones((num_timesteps,), dtype=torch.float)
        elif sampling_method == 'linear':
            print(f'You will sample time steps linearly. The range of weights is [{low}, {high}].')
            self.weights = torch.from_numpy(np.linspace(low, high, num_timesteps))
        else:
            self.weights = torch.ones((num_timesteps,), dtype=torch.float)
            print(f'You will sample time steps uniformly.')

    def update_weights(self, t, loss_value):
        if self.sampling_method == 'importance':
            self.weights[t] = self.weights[t] * self.freq[t] / (self.freq[t] + 1) + loss_value / (self.freq[t] + 1)
            self.freq[t] += 1.0

    def __call__(self, n_steps_per_iter=1):
        self.last_t = torch.multinomial(self.weights, n_steps_per_iter, replacement=False)
        return self.last_t


def get_all_edges_faster(Xt_pos, protein_positions, num_proposals, num_ligand_atoms, num_timesteps=1000, k=48):
    edge_list = []
    num_atoms_per_step = num_ligand_atoms * num_proposals
    offset2protein = (num_timesteps - 1) * num_atoms_per_step + (num_proposals - 1) * num_ligand_atoms
    for t_step in range(num_timesteps):
        for mol_idx in range(num_proposals):
            #             pos = torch.randn_like(Xt_pos) + 1000000
            atom_id_base = mol_idx * num_ligand_atoms + num_atoms_per_step * t_step
            start_id = (num_proposals * num_ligand_atoms) * t_step + mol_idx * num_ligand_atoms
            end_id = (num_proposals * num_ligand_atoms) * t_step + (mol_idx + 1) * num_ligand_atoms
            #             print("start", start_id, "end", end_id)
            #             pos[start_id:end_id, :] = Xt_pos[start_id:end_id, :]
            pos = Xt_pos[start_id:end_id, :]
            edge = knn_graph(torch.cat([pos, protein_positions], dim=0), k=k,
                             flow="target_to_source", num_workers=64, loop=False)
            edge_ligand = edge[:, :k * num_ligand_atoms]  # only get the edges from ligand atoms to other atoms
            edge_ligand[0, :] += atom_id_base
            edge_ligand[1, :] = torch.where(edge_ligand[1, :] < num_ligand_atoms,
                                            edge_ligand[1, :] + atom_id_base, edge_ligand[1, :] + offset2protein)
            edge_list.append(edge_ligand)
            edges = torch.cat(edge_list, 1)

    return edges


def get_all_bond_edges(bond_edges, num_proposals, num_ligand_atoms):
    edges_list = [bond_edges, ]
    for p in range(1, num_proposals):
        edges_list.append(bond_edges + p * num_ligand_atoms)

    edges = torch.cat(edges_list, dim=1)
    return edges


def cal_indices(edge_index, bond_index):
    '''
    This function is used to get the index correspondence between edge_attr and bond_attr via
    traverse edge_index and bond_index. It can also used to assign bond embeddings to all edges within a ligand.
    '''
    num_edges = edge_index.shape[1]
    num_bond_edges = bond_index.shape[1]
    ee = edge_index.t().reshape(-1, 1).repeat(1, num_bond_edges)
    bb = bond_index.repeat(num_edges, 1)
    diff = (ee - bb).abs()
    odd_indices = torch.Tensor([1, 0]).bool().repeat(num_edges, )
    even_indices = torch.Tensor([0, 1]).bool().repeat(num_edges, )
    odd_rows, even_rows = diff[odd_indices], diff[even_indices]
    res = odd_rows + even_rows

    return (res == 0).nonzero()  # [:, 1]  # the column 1 should be the desired indices


def measure_straightness(X0, X1, v_ls, method='l1'):
    '''
    inputs: pos_data is a tuple of X0_pos, X1_pos, and list of v_pos
    outputs: straightness is the straightness of position flow
    '''
    v_gt = X1 - X0.to(X1.device)
    n_steps = len(v_ls)
    if method == 'l1':
        v_gt = torch.stack(n_steps * [v_gt])
        # print("v_gt", v_gt.shape)
        v = torch.stack(v_ls)
        # print("v", v.shape)
        straightness = torch.mean(torch.abs(v_gt - v))
    elif method == 'l2':
        straightness = torch.mean((torch.stack(n_steps * [v_gt]) - torch.stack(v_ls).to(v_gt.device))**2)
    else:
        raise ValueError("We only support 'l1' or 'l2' for method options.")

    return straightness


def seed_torch(seed=2023):
    # random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # The randomness of hash is not allowed for reproducibility.
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def prepare_pairs_fast(pairs_path='./tmp/pref_pairs_np', num_positive_samples=3, num_negative_samples=None,
                       temperature=5.0, logger=print, query=6, ckps_ls=None, noise=None, time_sampler=None,
                       soft_labels_all=False, variance_trials=5, t=None):
    if num_negative_samples is None:
        num_negative_samples = num_positive_samples
    if ckps_ls is None:
        pairs_file_ls = [x for x in os.listdir(pairs_path) if '.pt' in x]
        random.shuffle(pairs_file_ls)
        ckp = torch.load(os.path.join(pairs_path, pairs_file_ls[0]), map_location='cpu')
    else:  # The probablity of sampling a specific list is proportional to the number of positive samples.
        ckp_path = ckps_ls[np.random.choice(len(ckps_ls), 1, replace=False)[0]]
        ckp = torch.load(ckp_path, map_location='cpu')
    X0_pos_array = ckp['X0_pos_array']
    X1_pos_array = ckp['X1_pos_array']
    X0_ele_emb_array = ckp['X0_ele_emb_array']
    X1_ele_emb_array = ckp['X1_ele_emb_array']
    X0_bond_emb_array = ckp['X0_bond_emb_array']
    X1_bond_emb_array = ckp['X1_bond_emb_array']
    num_samples, num_atoms, pos_dim = X1_pos_array.shape
    _, _, ele_emb_dim = X1_ele_emb_array.shape
    _, _, bond_emb_dim = X1_bond_emb_array.shape
    positive_indices = ckp[str(query) + '_positive_indices']
    negative_indices = ckp[str(query) + '_negative_indices']
    if variance_trials and 'benz' not in query and 'large_ring' not in query:
        indices_ls, res_ls = [], []
        for i in range(variance_trials):
            positive_choices = np.random.choice(positive_indices, num_positive_samples, replace=False)
            negative_choices = np.random.choice(negative_indices, num_negative_samples, replace=False)
            indices_ls.append(np.concatenate([positive_choices, negative_choices]))
            res_ls.append(ckp[str(query) + '_array'][indices_ls[-1]])
        stds = np.std(np.array(res_ls), axis=1)
        i = np.argmax(stds)
        positive_choices, negative_choices = indices_ls[i][:num_positive_samples], indices_ls[i][num_positive_samples:]
    else:
        positive_choices = np.random.choice(positive_indices, num_positive_samples, replace=False)
        negative_choices = np.random.choice(negative_indices, num_negative_samples, replace=False)

    sample_X1_pos = torch.cat((X1_pos_array[positive_choices], X1_pos_array[negative_choices]), dim=0).reshape(-1,
                                                                                                               pos_dim)
    sample_X0_pos = torch.cat((X0_pos_array[positive_choices], X0_pos_array[negative_choices]), dim=0).reshape(-1,
                                                                                                               pos_dim)
    sample_X1_ele_emb = torch.cat((X1_ele_emb_array[positive_choices], X1_ele_emb_array[negative_choices]),
                                  dim=0).reshape(-1, ele_emb_dim)
    sample_X0_ele_emb = torch.cat((X0_ele_emb_array[positive_choices], X0_ele_emb_array[negative_choices]),
                                  dim=0).reshape(-1, ele_emb_dim)
    sample_X1_bond_emb = torch.cat((X1_bond_emb_array[positive_choices], X1_bond_emb_array[negative_choices]),
                                   dim=0).reshape(-1, bond_emb_dim)
    sample_X0_bond_emb = torch.cat((X0_bond_emb_array[positive_choices], X0_bond_emb_array[negative_choices]),
                                   dim=0).reshape(-1, bond_emb_dim)
    if noise is not None:
        sample_X0_pos = (sample_X0_pos + torch.randn_like(sample_X0_pos) * noise) / np.sqrt(1.0 + noise ** 2)
        sample_X0_ele_emb = (sample_X0_ele_emb + torch.randn_like(sample_X0_ele_emb) * noise) / np.sqrt(
            1.0 + noise ** 2)
        sample_X0_bond_emb = (sample_X0_bond_emb + torch.randn_like(sample_X0_bond_emb) * noise) / np.sqrt(
            1.0 + noise ** 2)
    if t is None:
        t = time_sampler()
    t_float = (1.0 * t) / time_sampler.num_timesteps
    Xt_pos_pair = t_float * sample_X1_pos + (1.0 - t_float) * sample_X0_pos
    Xt_ele_emb_pair = t_float * sample_X1_ele_emb + (1.0 - t_float) * sample_X0_ele_emb
    Xt_bond_emb_pair = t_float * sample_X1_bond_emb + (1.0 - t_float) * sample_X0_bond_emb
    target_pos_pair = sample_X1_pos - sample_X0_pos
    target_ele_emb_pair = sample_X1_ele_emb - sample_X0_ele_emb
    target_bond_emb_pair = sample_X1_bond_emb - sample_X0_bond_emb
    if 'benzene' in query or 'large_ring' in query:
        labels = torch.zeros(num_positive_samples + num_negative_samples)
        labels[:num_positive_samples] = 1.0 / num_positive_samples
    else:
        positive_result = torch.from_numpy(ckp[str(query) + '_array'][positive_choices])
        if query == 'vina_score' or 'angle_deviation' in query:
            positive_result *= -1.0
        if soft_labels_all:
            negative_result = torch.from_numpy(ckp[str(query) + '_array'][negative_choices])
            if query == 'vina_score' or 'angle_deviation' in query:
                negative_result *= -1.0
            all_results = torch.cat([positive_result, negative_result], dim=-1)
            labels = F.softmax(all_results * temperature, dim=-1)
        else:
            labels = torch.zeros(num_positive_samples + num_negative_samples)
            labels[:num_positive_samples] = F.softmax(positive_result * temperature, dim=-1)

    return target_pos_pair, target_ele_emb_pair, target_bond_emb_pair, \
        Xt_pos_pair, Xt_ele_emb_pair, Xt_bond_emb_pair, t, labels


def prepare_proposals(num_positive_samples, num_negative_samples, pairs_path=None,
                      temperature=5.0, logger=print, query=6, ckps_ls=None, noise=None,
                      soft_labels_all=False, variance_trials=5, perturb_signal=False, ckp_id_records=[]):
    if ckps_ls is None:
        pairs_file_ls = [x for x in os.listdir(pairs_path) if '.pt' in x]
        random.shuffle(pairs_file_ls)
        ckp = torch.load(os.path.join(pairs_path, pairs_file_ls[0]), map_location='cpu')
    else:  # The probablity of sampling a specific list is proportional to the number of positive samples.
        if len(ckp_id_records) == 0:
            logger(f"We have traversed all {len(ckps_ls)} lists.")
            ckp_id_records = list(range(len(ckps_ls)))
        ckp_id = np.random.choice(ckp_id_records, 1, replace=False)[0]
        ckp = torch.load(ckps_ls[ckp_id], map_location='cpu')
        ckp_id_records.remove(ckp_id)
    positive_indices = ckp[str(query) + '_positive_indices']
    negative_indices = ckp[str(query) + '_negative_indices']
    if variance_trials and 'benzene' not in query and 'large_ring' not in query and (
            num_positive_samples * num_negative_samples != 0):
        indices_ls, res_ls = [], []
        for i in range(variance_trials):
            positive_choices = np.random.choice(positive_indices, num_positive_samples, replace=False)
            negative_choices = np.random.choice(negative_indices, num_negative_samples, replace=False)
            indices_ls.append(np.concatenate([positive_choices, negative_choices]))
            res_ls.append(ckp[str(query) + '_array'][indices_ls[-1]])
        stds = np.std(np.array(res_ls), axis=1)
        i = np.argmax(stds)
        positive_choices, negative_choices = indices_ls[i][:num_positive_samples], indices_ls[i][num_positive_samples:]
    else:
        positive_choices = np.random.choice(positive_indices, num_positive_samples, replace=False)
        negative_choices = np.random.choice(negative_indices, num_negative_samples, replace=False)
    X0_pos_array = ckp['X0_pos_array']
    X1_pos_array = ckp['X1_pos_array']
    X0_ele_emb_array = ckp['X0_ele_emb_array']
    X1_ele_emb_array = ckp['X1_ele_emb_array']
    X0_bond_emb_array = ckp['X0_bond_emb_array']
    X1_bond_emb_array = ckp['X1_bond_emb_array']
    num_samples, num_atoms, pos_dim = X1_pos_array.shape
    _, _, ele_emb_dim = X1_ele_emb_array.shape
    _, _, bond_emb_dim = X1_bond_emb_array.shape

    sample_X1_pos = torch.cat((X1_pos_array[positive_choices], X1_pos_array[negative_choices]), dim=0).reshape(-1,
                                                                                                               pos_dim)
    sample_X0_pos = torch.cat((X0_pos_array[positive_choices], X0_pos_array[negative_choices]), dim=0).reshape(-1,
                                                                                                               pos_dim)
    sample_X1_ele_emb = torch.cat((X1_ele_emb_array[positive_choices], X1_ele_emb_array[negative_choices]),
                                  dim=0).reshape(-1, ele_emb_dim)
    sample_X0_ele_emb = torch.cat((X0_ele_emb_array[positive_choices], X0_ele_emb_array[negative_choices]),
                                  dim=0).reshape(-1, ele_emb_dim)
    sample_X1_bond_emb = torch.cat((X1_bond_emb_array[positive_choices], X1_bond_emb_array[negative_choices]),
                                   dim=0).reshape(-1, bond_emb_dim)
    sample_X0_bond_emb = torch.cat((X0_bond_emb_array[positive_choices], X0_bond_emb_array[negative_choices]),
                                   dim=0).reshape(-1, bond_emb_dim)
    if noise is not None:
        sample_X0_pos = np.sqrt(1.0 - noise) * sample_X0_pos + np.sqrt(noise) * torch.randn_like(sample_X0_pos)
        sample_X0_ele_emb = np.sqrt(1.0 - noise) * sample_X0_ele_emb + np.sqrt(noise) * torch.randn_like(
            sample_X0_ele_emb)
        sample_X0_bond_emb = np.sqrt(1.0 - noise) * sample_X0_bond_emb + np.sqrt(noise) * torch.randn_like(
            sample_X0_bond_emb)
    if perturb_signal:
        sample_X1_pos = np.sqrt(1.0 - noise) * sample_X1_pos + np.sqrt(noise) * torch.randn_like(sample_X1_pos)
        sample_X1_ele_emb = np.sqrt(1.0 - noise) * sample_X1_ele_emb + np.sqrt(noise) * torch.randn_like(
            sample_X1_ele_emb)
        sample_X1_bond_emb = np.sqrt(1.0 - noise) * sample_X1_bond_emb + np.sqrt(noise) * torch.randn_like(
            sample_X1_bond_emb)

    target_pos_pair = sample_X1_pos - sample_X0_pos
    target_ele_emb_pair = sample_X1_ele_emb - sample_X0_ele_emb
    target_bond_emb_pair = sample_X1_bond_emb - sample_X0_bond_emb
    if 'benzene' in query or 'large_ring' in query:
        labels = torch.zeros(num_positive_samples + num_negative_samples)
        if num_positive_samples:
            labels[:num_positive_samples] = 1.0 / num_positive_samples
        results = labels
    else:
        positive_result = torch.from_numpy(ckp[str(query) + '_array'][positive_choices])
        negative_result = torch.from_numpy(ckp[str(query) + '_array'][negative_choices])
        if query == 'vina_score' or 'angle_deviation' in query:
            positive_result *= -1.0
            negative_result *= -1.0
        if soft_labels_all:
            all_results = torch.cat([positive_result, negative_result], dim=-1)
            labels = F.softmax(all_results * temperature, dim=-1)
        else:
            labels = torch.zeros(num_positive_samples + num_negative_samples)
            labels[:num_positive_samples] = F.softmax(positive_result * temperature, dim=-1)
        results = torch.cat([positive_result, negative_result], dim=-1)
    return target_pos_pair, target_ele_emb_pair, target_bond_emb_pair, \
        sample_X1_pos, sample_X0_pos, sample_X1_ele_emb, sample_X0_ele_emb, \
        sample_X1_bond_emb, sample_X0_bond_emb, labels, results


def get_Xt(sample_X1_pos, sample_X0_pos, sample_X1_ele_emb, sample_X0_ele_emb, sample_X1_bond_emb, sample_X0_bond_emb,
           num_timesteps, t=None, time_sampler=None, num_steps=1, noise=0):
    if t is None:
        t = time_sampler(num_steps)
    t = t.reshape(-1, 1)
    t_int = t.repeat(1, int(sample_X1_pos.shape[0] / num_steps)).flatten()
    t_bond_int = t.repeat(1, int(sample_X1_bond_emb.shape[0] / num_steps)).flatten()
    t_float = ((1.0 * t_int) / num_timesteps).reshape(-1, 1)
    if noise:
        noise = np.random.random() * noise
        sample_X0_pos = sample_X0_pos * np.sqrt(1 - noise) + torch.randn_like(sample_X0_pos) * np.sqrt(noise)
        sample_X0_ele_emb = sample_X0_ele_emb * np.sqrt(1 - noise) + torch.randn_like(sample_X0_ele_emb) * np.sqrt(
            noise)
        sample_X0_bond_emb = sample_X0_bond_emb * np.sqrt(1 - noise) + torch.randn_like(sample_X0_bond_emb) * np.sqrt(
            noise)
    Xt_pos_pair = t_float * sample_X1_pos + (1.0 - t_float) * sample_X0_pos
    Xt_ele_emb_pair = t_float * sample_X1_ele_emb + (1.0 - t_float) * sample_X0_ele_emb
    t_float = ((1.0 * t_bond_int) / num_timesteps).reshape(-1, 1)
    Xt_bond_emb_pair = t_float * sample_X1_bond_emb + (1.0 - t_float) * sample_X0_bond_emb
    return Xt_pos_pair, Xt_ele_emb_pair, Xt_bond_emb_pair, t_int, t_bond_int


def classifier_val(**kwargs):
    device = kwargs['device']
    num_ligand_atom_per_proposal = kwargs['num_ligand_atom_per_proposal']
    cls_loss_type = kwargs['cls_loss_type']
    model, num_positive_proposals = kwargs['model'], kwargs['num_positive_samples']
    num_negative_proposals = kwargs['num_negative_samples']
    online_avg = OnlineAveraging()
    model.eval()
    with torch.no_grad():
        for _ in range(kwargs['num_iterations']):
            _, _, _, Xt_pos_pair, Xt_ele_emb_pair, Xt_bond_emb_pair, t, labels = \
                prepare_pairs_fast(pairs_path=kwargs['pairs_path'], num_positive_samples=num_positive_proposals,
                                   num_negative_samples=num_negative_proposals,
                                   logger=print, query=kwargs['query'], ckps_ls=kwargs['ckps_ls'],
                                   noise=None, temperature=kwargs['temperature'],
                                   time_sampler=kwargs['time_sampler'], soft_labels_all=False,
                                   variance_trials=kwargs['variance_trials'], t=None)
            batch_proposals = torch.tensor(range(kwargs['num_total_proposals'] * 1)).repeat(
                num_ligand_atom_per_proposal, 1).t().flatten()
            t = t.reshape(-1, 1)
            t_int = t.repeat(1, int(Xt_pos_pair.shape[0] / 1)).flatten()
            t_bond_int = t.repeat(1, int(Xt_bond_emb_pair.shape[0] / 1)).flatten()
            logits_cls = model(Xt_pos_pair.to(device), Xt_ele_emb_pair.to(device), t_int.to(device),
                               num_atoms_per_ligand=num_ligand_atom_per_proposal,
                               batch_ligand=batch_proposals.to(device),
                               bond_features=Xt_bond_emb_pair.to(device), t_bond=t_bond_int.to(device),
                               protein_positions=kwargs['protein_positions'], protein_ele=kwargs['protein_ele'],
                               protein_amino_acid=kwargs['protein_amino_acid'],
                               protein_is_backbone=kwargs['protein_is_backbone'],
                               num_proposals=kwargs['num_total_proposals'], n_steps_per_iter=1
                               )
            loss_cls = CLASSIFIER_LOSS_FUNC_DICT[cls_loss_type](logits=logits_cls, labels=labels.to(device),
                                                                n_steps=1,
                                                                num_positive_proposals=num_positive_proposals)
            online_avg_loss = online_avg(loss_cls=loss_cls.item())

    return online_avg_loss['loss_cls']


def ode(model, protein_pos, protein_ele, protein_amino_acid, protein_is_backbone, z_pos, z_feat, num_timesteps,
        batch_protein=None, batch_ligand=None, logger=print, bond_features=None, bond_edges=None, device='cpu',
        reverse=False, return_v=False):
    '''
    If reverse=True, it is ode simulation for sampling. Otherwise, it is ode simulation for getting a latent code.
    '''
    model.eval()
    with torch.no_grad():
        protein_pos, protein_ele = protein_pos.to(device), protein_ele.to(device)
        protein_amino_acid, protein_is_backbone = protein_amino_acid.to(device), protein_is_backbone.to(device)
        z_pos, z_feat = z_pos.to(device), z_feat.to(device)
        if bond_features is not None:
            bond_features, bond_edges = bond_features.to(device), bond_edges.to(device)
        dt = 1. / num_timesteps
        num_atoms_per_ligand = z_pos.shape[0]
        num_bond_edges = bond_features.shape[0]
        if reverse:
            timesteps, desc = reversed(range(num_timesteps)), 'getting a latent code'
        else:
            timesteps, desc = range(num_timesteps), 'sampling'
        if return_v:
            v_pos, v_ele, v_bond = [], [], []
        for i in tqdm(timesteps, desc=desc, total=num_timesteps):
            t_int = torch.as_tensor(i).reshape(-1, 1).to(device)
            t = t_int.repeat(num_atoms_per_ligand, 1).squeeze()
            t_bond = t_int.repeat(num_bond_edges, 1).squeeze()
            pred_pos_v, pred_ele_v, pred_bond_v = model(protein_pos, protein_ele, protein_amino_acid,
                                                        protein_is_backbone, z_pos, z_feat, t, batch_protein,
                                                        batch_ligand,
                                                        bond_features=bond_features, bond_indices=bond_edges,
                                                        num_atoms_per_ligand=num_atoms_per_ligand, t_bond=t_bond)
            if reverse:
                z_pos -= pred_pos_v * dt
                z_feat -= pred_ele_v * dt
                if bond_features is not None:
                    bond_features -= pred_bond_v * dt
            else:
                z_pos += pred_pos_v * dt
                z_feat += pred_ele_v * dt
                if bond_features is not None:
                    bond_features += pred_bond_v * dt
            if return_v:
                v_pos.append(pred_pos_v)
                v_ele.append(pred_ele_v)
                v_bond.append(pred_bond_v)

    return z_pos, z_feat, bond_features, (v_pos, v_ele, v_bond)


def convert_ele_emb2ele_types(model, z_feat, mode='add_aromatic_wo_h'):
    # calculate the distances between predicted features and the constructed feature embeddings
    dist_betw_X1_feat_and_embedding = torch.cdist(z_feat, model.ligand_element_embedding.weight.data,
                                                  p=2)  # * model.emb_scale_factor
    pred_elements = torch.argmin(dist_betw_X1_feat_and_embedding, dim=-1)
    pred_atom_type = trans.get_atomic_number_from_index(pred_elements, mode=mode)
    return pred_atom_type



class EarlyStopping():
    def __init__(self, patience=10, mode='min'):
        self.best_score = None
        self.mode = mode
        self.patience = patience
        self.counter = 0

    def comparison(self, metric):
        if self.mode == 'min':
            return True if metric < self.best_score else False
        elif self.mode == 'max':
            return True if metric > self.best_score else False
        else:
            raise NotImplementedError(f"mode {self.mode} is not implemented.")

    def __call__(self, path, logger=print, **kwargs):
        if self.best_score is None:
            torch.save(kwargs, path)
            self.best_score = kwargs['metric']
            return False
        elif self.comparison(kwargs['metric']):
            self.best_score = kwargs['metric']
            torch.save(kwargs, path)
            self.counter = 0
            return False
        else:
            self.counter += 1
            if not self.comparison(kwargs['metric']) and self.counter >= self.patience:
                logger(f"Early stopping: the best score is {self.best_score:.3f}")
                return True
            else:
                logger(f"No improvements for {self.counter} times and the patience is {self.patience}. "
                       f"The best score is {self.best_score:.3f}.")
                return False


def split_positive_negative_logits(**kwargs):
    n_steps, num_positive_proposals = kwargs['n_steps'], kwargs['num_positive_proposals']
    logits = kwargs['logits'].reshape(n_steps, -1)
    logits_positive = logits[:, :num_positive_proposals].flatten()
    logits_negative = logits[:, num_positive_proposals:].flatten()
    labels_positive = kwargs['labels'][:num_positive_proposals].reshape(-1, 1).repeat(1, n_steps).flatten()
    if len(labels_positive):
        labels_positive = labels_positive / torch.max(labels_positive)
        labels_positive = torch.where(labels_positive < 0.6, 0.6, labels_positive)
    labels_negative = kwargs['labels'][num_positive_proposals:].reshape(-1, 1).repeat(1, n_steps).flatten()
    return logits_positive, logits_negative, labels_positive, labels_negative


def classifier_BCE_loss(**kwargs):
    return F.binary_cross_entropy_with_logits(kwargs['logits'], kwargs['labels'])


def classifier_MSE_loss(**kwargs):
    logits_positive, logits_negative, labels_positive, labels_negative = split_positive_negative_logits(**kwargs)
    labels_negative = torch.ones_like(labels_positive)
    loss_positive = torch.mean(labels_positive * (logits_positive - 1) ** 2)
    loss_negative = torch.mean(labels_negative * (logits_negative + 1) ** 2)
    return loss_positive + loss_negative


def classifier_MSE_hinge_loss(**kwargs):
    logits_positive, logits_negative, labels_positive, labels_negative = split_positive_negative_logits(**kwargs)
    labels_negative = torch.ones_like(labels_positive)
    selected_idx_positive = (logits_positive.detach() < 1.0).long()
    selected_idx_negative = (logits_negative.detach() > -1.0).long()
    loss_positive = torch.mean(selected_idx_positive * labels_positive * (logits_positive - 1) ** 2) / (
            selected_idx_positive.count_nonzero() + 1.0e-6)
    loss_negative = torch.mean(selected_idx_negative * labels_negative * (logits_negative + 1) ** 2) / (
            selected_idx_negative.count_nonzero() + 1.0e-6)
    return loss_positive + loss_negative


def classifier_MAE_loss(**kwargs):
    logits_positive, logits_negative, labels_positive, labels_negative = split_positive_negative_logits(**kwargs)
    labels_negative = torch.ones_like(labels_positive)
    loss_positive = torch.mean(labels_positive * (logits_positive - 1).abs())
    loss_negative = torch.mean(labels_negative * (logits_negative + 1).abs())
    return loss_positive + loss_negative


def classifier_MAE_hinge_loss(**kwargs):
    logits_positive, logits_negative, labels_positive, labels_negative = split_positive_negative_logits(**kwargs)
    labels_negative = torch.ones_like(labels_positive)
    selected_idx_positive = (logits_positive.detach() < 1.0).long()
    selected_idx_negative = (logits_negative.detach() > -1.0).long()
    loss_positive = torch.sum(selected_idx_positive * labels_positive * (logits_positive - 1).abs()) / (
            selected_idx_positive.count_nonzero() + 1.0e-6)
    loss_negative = torch.sum(selected_idx_negative * labels_negative * (logits_negative + 1).abs()) / (
            selected_idx_negative.count_nonzero() + 1.0e-6)
    return loss_positive + loss_negative


CLASSIFIER_LOSS_FUNC_DICT = {'BCE': classifier_BCE_loss,
                             'MSE_hinge': classifier_MSE_hinge_loss, 'MSE': classifier_MSE_loss,
                             'MAE': classifier_MAE_loss, 'MAE_hinge': classifier_MAE_hinge_loss}


def preference_CE_loss(**kwargs):
    logits, n_steps, labels = kwargs['logits'], kwargs['n_steps'], kwargs['labels']
    return F.cross_entropy(-logits.reshape(n_steps, -1), labels.repeat(n_steps).reshape(n_steps, -1))


def preference_BCE_loss(**kwargs):
    logits_positive, logits_negative, labels_positive, labels_negative = split_positive_negative_logits(**kwargs)
    loss_positive = F.binary_cross_entropy(1.0 / (1.0 + logits_positive), labels_positive) if len(
        labels_positive) else 0
    loss_negative = F.binary_cross_entropy(1.0 / (1.0 + logits_negative), labels_negative) if len(
        labels_negative) else 0
    return loss_positive + loss_negative


def preference_MAE_loss(**kwargs):
    logits_positive, logits_negative, labels_positive, labels_negative = split_positive_negative_logits(**kwargs)
    if len(labels_positive):
        loss_positive = torch.mean(labels_positive * logits_positive)
    else:
        loss_positive = 0
    if len(labels_negative):
        labels_negative = torch.ones_like(labels_negative)
        loss_negative_tmp = torch.maximum(3.5 - logits_negative, torch.tensor(0))
        loss_negative = torch.sum(labels_negative * loss_negative_tmp) / (
                    loss_negative_tmp.detach().count_nonzero() + 1.0e-6)
    else:
        loss_negative = 0
    return loss_positive + loss_negative


PREFERENCE_LOSS_FUNC_DICT = {'CE': preference_CE_loss, 'BCE': preference_BCE_loss, 'MAE': preference_MAE_loss}
