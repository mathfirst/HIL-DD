import json, os, torch, time
from datasets.pl_pair_dataset import PocketLigandPairDataset
from torch.utils.data import Subset
from torch_geometric.data import Data
import numpy as np
from .util_flow import prepare_proposals, get_Xt, get_all_bond_edges, subtract_mean
from datetime import datetime

def torchify_dict(data):
    # from TargetDiff code
    output = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            output[k] = torch.from_numpy(v)
        else:
            output[k] = v
    return output

class PocketLigandPairData(Data):
    def __init__(self, *args, **kwargs):
        super(PocketLigandPairData, self).__init__(*args, **kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'ligand_bond_index':
            return self['ligand_element'].size(0)
        else:
            return super().__inc__(key, value)


ProteinElement2IndexDict = {1: 0, 6: 1, 7: 2, 8: 3, 16: 4}

MAP_ATOM_TYPE_AROMATIC_TO_INDEX_wo_h = {
    # (1, False): 0,
    (6, False): 0,
    (6, True): 1,
    (7, False): 2,
    (7, True): 3,
    (8, False): 4,
    (8, True): 5,
    (9, False): 6,
    (15, False): 7,
    (15, True): 8,
    (16, False): 9,
    (16, True): 10,
    (17, False): 11
}

def get_dataset(config, *args, **kwargs):
    name = config.name
    root = config.path
    if name == 'pl':
        dataset = PocketLigandPairDataset(root, split=config.split, *args, **kwargs)
    else:
        raise NotImplementedError('Unknown dataset: %s' % name)

    if 'split' in config:
        split_by_name = torch.load(config.split)
        split = {
            k: [dataset.name2id[n] for n in names if n in dataset.name2id]
            for k, names in split_by_name.items()
        }
        subsets = {k: Subset(dataset, indices=v) for k, v in split.items()}
        return dataset, subsets
    else:
        return dataset

def extract_data(batch, dict_protein_ele):
    protein_pos = batch['protein_pos']  # * pos_scale  # get the positions of protein atoms
    protein_elements = batch['protein_element']  # get the elements of protein atoms
    protein_ele = torch.as_tensor([dict_protein_ele[key.item()] for key in
                                   protein_elements])  # get the indices of the elements in the dictionary
    protein_amino_acid = batch['protein_atom_to_aa_type']  # get the amino acid info of protein atoms
    protein_is_backbone = batch['protein_is_backbone']  # get the backbone info of protein atoms
    if 'ligand_pos' in batch:
        ligand_pos = batch['ligand_pos']
        ligand_ele = batch['ligand_atom_feature_full']  # get the elements of ligand atoms
    else:
        ligand_pos = ligand_ele = None
    return protein_pos, protein_ele, protein_amino_acid, protein_is_backbone, ligand_pos, ligand_ele

def prepare_inputs(batch, model, ProteinElement2IndexDict, num_timesteps, pos_scale, add_bond, timestep_sampler,
                   n_steps_per_iter=1, device='cpu', t=None):
    if 'ligand_element_batch' not in batch:
        protein_batch, ligand_batch = None, None
    else:
        protein_batch, ligand_batch = batch['protein_element_batch'], batch['ligand_element_batch']
    protein_pos, protein_ele, protein_amino_acid, protein_is_backbone, ligand_pos, ligand_ele \
        = extract_data(batch, ProteinElement2IndexDict)
    ligand_bond_types = batch['ligand_bond_type']  # .to(device)
    ligand_bond_indices = batch['ligand_bond_index']  # .to(device)
    if t is None:
        t = timestep_sampler(n_steps_per_iter)
    t = t.reshape(-1, 1)
    if add_bond:
        ligand_bond_types_all, bond_edges = model.get_ligand_bond_types_all(num_ligand_atoms=ligand_pos.shape[0],
                                                                            ligand_bond_indices=ligand_bond_indices,
                                                                            ligand_bond_types=ligand_bond_types,
                                                                            device=device)
        num_edges = bond_edges.shape[1]
        ligand_bond_features = model.get_bond_type_features(ligand_bond_types_all)
        X0_pos, X0_element_embedding, X0_bond_embedding = model.get_X0(num_ligand_atoms=ligand_pos.shape[0],
                                                                       num_edges=num_edges)
        if n_steps_per_iter > 1:
            bond_edges = get_all_bond_edges(bond_edges, n_steps_per_iter, ligand_pos.shape[0])
        bond_edges = bond_edges.to(device)
        t_bond = t.repeat(1, ligand_bond_features.shape[0]).reshape(-1, 1).to(device)
        t_float_bond = (t_bond * 1.0) / num_timesteps  # t_bond.float()
        Xt_bond_embedding = (t_float_bond * ligand_bond_features.repeat(n_steps_per_iter, 1)
                             + (1. - t_float_bond) * X0_bond_embedding.repeat(n_steps_per_iter, 1))
        target_bond_features = (ligand_bond_features - X0_bond_embedding).repeat(n_steps_per_iter, 1)
        t_bond = t_bond.squeeze()
    else:
        num_edges = None
        X0_pos, X0_element_embedding, X0_bond_embedding = model.get_X0(num_ligand_atoms=ligand_pos.shape[0],
                                                                       num_edges=num_edges)
        bond_edges, Xt_bond_embedding, target_bond_features, t_bond = None, None, None, None
    # center and scale positions
    protein_pos, mean_vec = subtract_mean(protein_pos, protein_batch, verbose=True)
    X1_pos = ligand_pos - mean_vec[ligand_batch]
    X1_pos = X1_pos.to(device)
    protein_pos *= pos_scale
    X1_pos *= pos_scale
    X1_element_embedding = model.get_ligand_features(ligand_ele)
    target_pos = X1_pos - X0_pos
    target_element_embedding = X1_element_embedding - X0_element_embedding
    t = t.repeat(1, ligand_pos.shape[0]).reshape(-1, 1).to(device)  # n_steps_per_iter=2, e.g. [0,0,0,1,1,1]
    t_float = (t * 1.0) / num_timesteps
    # print("X1_pos.shape", X1_pos.squeeze().shape, "X1_pos.shape", X1_pos.squeeze().shape, "n_steps_per_iter", n_steps_per_iter)
    Xt_pos = t_float * X1_pos.squeeze().repeat(n_steps_per_iter, 1) + (1. - t_float) * X0_pos.squeeze().repeat(n_steps_per_iter, 1)
    Xt_element_embedding = t_float * X1_element_embedding.repeat(n_steps_per_iter, 1) \
                           + (1. - t_float) * X0_element_embedding.repeat(n_steps_per_iter, 1)
    return protein_pos.to(device), protein_ele.to(device), protein_amino_acid.to(device), protein_is_backbone.to(
        device), \
        Xt_pos.to(device), Xt_element_embedding.to(device), t.squeeze().to(
        device), protein_batch, ligand_batch, bond_edges, \
        Xt_bond_embedding, target_pos.squeeze().repeat(n_steps_per_iter, 1).to(device), \
        target_element_embedding.repeat(n_steps_per_iter, 1).to(device), \
        target_bond_features, X1_pos, t_bond


class PreparePrefData4UI:
    def __init__(self, **kwargs):
        self.sample_X1_pos = self.sample_X0_pos = torch.empty(0)
        self.sample_X1_ele = self.sample_X0_ele = torch.empty(0)
        self.sample_X1_bond = self.sample_X0_bond = torch.empty(0)
        self.target_pos = self.target_ele = self.target_bond = torch.empty(0)
        self.positive_indices, self.negative_indices = [], []
        self.ele_emb_dim = kwargs['ele_emb_dim']
        self.bond_emb_dim = kwargs['bond_emb_dim']
        self.num_timesteps = kwargs['num_timesteps']

    def storeData(self, ckp, like_ls, dislike_ls):
        data = {}
        data['X0_pos'] = torch.stack(ckp['X0_pos_list'], dim=0)
        data['X1_pos'] = torch.stack(ckp['X1_pos_list'], dim=0)
        data['X0_ele'] = torch.stack(ckp['X0_ele_emb_list'], dim=0)
        data['X1_ele'] = torch.stack(ckp['X1_ele_emb_list'], dim=0)
        data['X0_bond'] = torch.stack(ckp['X0_bond_emb_list'], dim=0)
        data['X1_bond'] = torch.stack(ckp['X1_bond_emb_list'], dim=0)
        sample_X1_pos = torch.cat((data['X1_pos'][like_ls], data['X1_pos'][dislike_ls]), dim=0)
        sample_X0_pos = torch.cat((data['X0_pos'][like_ls], data['X0_pos'][dislike_ls]), dim=0)
        sample_X1_ele = torch.cat((data['X1_ele'][like_ls], data['X1_ele'][dislike_ls]), dim=0)
        sample_X0_ele = torch.cat((data['X0_ele'][like_ls], data['X0_ele'][dislike_ls]), dim=0)
        sample_X1_bond = torch.cat((data['X1_bond'][like_ls], data['X1_bond'][dislike_ls]), dim=0)
        sample_X0_bond = torch.cat((data['X0_bond'][like_ls], data['X0_bond'][dislike_ls]), dim=0)
        target_pos = sample_X1_pos - sample_X0_pos
        target_ele = sample_X1_ele - sample_X0_ele
        target_bond = sample_X1_bond - sample_X0_bond
        self.target_pos = torch.cat([self.target_pos, target_pos], dim=0)
        self.target_ele = torch.cat([self.target_ele, target_ele], dim=0)
        self.target_bond = torch.cat([self.target_bond, target_bond], dim=0)
        self.sample_X1_pos = torch.cat([self.sample_X1_pos, sample_X1_pos], dim=0)
        self.sample_X0_pos = torch.cat([self.sample_X0_pos, sample_X0_pos], dim=0)
        self.sample_X1_ele = torch.cat([self.sample_X1_ele, sample_X1_ele], dim=0)
        self.sample_X0_ele = torch.cat([self.sample_X0_ele, sample_X0_ele], dim=0)
        self.sample_X1_bond = torch.cat([self.sample_X1_bond, sample_X1_bond], dim=0)
        self.sample_X0_bond = torch.cat([self.sample_X0_bond, sample_X0_bond], dim=0)
        num_proposals_prev = len(self.positive_indices) + len(self.negative_indices)
        self.positive_indices += list(range(num_proposals_prev, num_proposals_prev + len(like_ls)))
        self.negative_indices += list(range(num_proposals_prev + len(like_ls), num_proposals_prev + len(like_ls) + len(dislike_ls)))
        self.prob_positive = np.linspace(0.5, 1, len(self.positive_indices))
        if len(self.positive_indices):
            self.prob_positive /= np.sum(self.prob_positive)
        self.prob_negative = np.linspace(0.5, 1, len(self.negative_indices))
        if len(self.negative_indices):
            self.prob_negative /= np.sum(self.prob_negative)

    def sample_pair(self, num_positive, num_negative, device):
        if len(self.positive_indices) < num_positive:
            num_positive = len(self.positive_indices)
        if len(self.negative_indices) < num_negative:
            num_negative = len(self.negative_indices)
        positive_choices = np.random.choice(self.positive_indices, num_positive, replace=False,
                                            p=self.prob_positive) if len(self.prob_positive) != 0 else np.array([])
        negative_choices = np.random.choice(self.negative_indices, num_negative, replace=False,
                                            p=self.prob_negative) if len(self.prob_negative) != 0 else np.array([])
        target_pos = torch.cat((self.target_pos[positive_choices], self.target_pos[negative_choices]), dim=0).reshape(-1, 3)
        target_ele = torch.cat((self.target_ele[positive_choices], self.target_ele[negative_choices]), dim=0).reshape(-1, self.ele_emb_dim)
        target_bond = torch.cat((self.target_bond[positive_choices], self.target_bond[negative_choices]), dim=0).reshape(-1, self.bond_emb_dim)
        self.X1_pos = torch.cat((self.sample_X1_pos[positive_choices], self.sample_X1_pos[negative_choices]), dim=0).reshape(-1, 3)
        self.X0_pos = torch.cat((self.sample_X0_pos[positive_choices], self.sample_X0_pos[negative_choices]), dim=0).reshape(-1, 3)
        self.X1_ele = torch.cat((self.sample_X1_ele[positive_choices], self.sample_X1_ele[negative_choices]), dim=0).reshape(-1, self.ele_emb_dim)
        self.X0_ele = torch.cat((self.sample_X0_ele[positive_choices], self.sample_X0_ele[negative_choices]), dim=0).reshape(-1, self.ele_emb_dim)
        self.X1_bond = torch.cat((self.sample_X1_bond[positive_choices], self.sample_X1_bond[negative_choices]), dim=0).reshape(-1, self.bond_emb_dim)
        self.X0_bond = torch.cat((self.sample_X0_bond[positive_choices], self.sample_X0_bond[negative_choices]), dim=0).reshape(-1, self.bond_emb_dim)
        labels = torch.zeros(num_positive + num_negative).to(device)
        labels[:num_positive] = 1.0
        return target_pos, target_ele, target_bond, labels, num_positive, num_negative

    def get_xt(self, t, timestep_sampler):
        Xt_pos_pair, Xt_ele_emb_pair, Xt_bond_emb_pair, t_int, t_bond = get_Xt(self.X1_pos,
                                                                               self.X0_pos,
                                                                               self.X1_ele,
                                                                               self.X0_ele,
                                                                               self.X1_bond,
                                                                               self.X0_bond,
                                                                               self.num_timesteps, t=t,
                                                                               time_sampler=timestep_sampler)
        return Xt_pos_pair, Xt_ele_emb_pair, Xt_bond_emb_pair, t_int, t_bond


def get_proposals(proposals_dir, proposal_base_dict, logger=print):
    pt_list = [f for f in os.listdir(proposals_dir) if '.pt' in f]
    for pt in pt_list:
        logger(f"loading {pt}")
        time.sleep(0.1)
        pt_path = os.path.join(proposals_dir, pt)
        try:
            proposals = torch.load(pt_path, map_location='cpu')
            for k, v in proposals.items():
                if isinstance(v, list):
                    if k in proposal_base_dict:
                        proposal_base_dict[k] += v
                    else:
                        proposal_base_dict[k] = v
            os.remove(pt_path)
        except Exception as err:
            logger(f"when loading {pt}, {err}")


def proposal_stacked_json(proposals_dir, evaluation_path, proposal_base_dict, stacked_proposal_dict, num_inj, num_proposals_ui, logger=print):
    if len(proposal_base_dict['mol_list']) >= num_proposals_ui:
        logger(f"{num_inj} making proposals")
        if os.path.isfile(evaluation_path):
            check_prompt = True  # evaluation condition
        else:
            check_prompt = False
        dict_proposals = {'next_molecules': [], 'check_prompt': check_prompt}
        for i in range(num_proposals_ui):
            molecule = {'id': i,
                        'smiles': proposal_base_dict['smiles_list'][i],
                        'url': proposal_base_dict['sdf_path_list'][i],
                        'png_url': proposal_base_dict['png_path_list'][i],
                        'metrics': {'Vina': proposal_base_dict['vina_score_list'][i],
                                    'QED': proposal_base_dict['qed_list'][i],
                                    'SA': proposal_base_dict['sa_list'][i],
                                    'Lipinski': proposal_base_dict['lipinski_list'][i],
                                    'NumAtoms': proposal_base_dict['num_atoms_list'][i],
                                    'NumBonds': proposal_base_dict['num_bonds_list'][i],
                                    'NumRings': proposal_base_dict['num_rings_list'][i],
                                    'NumBenzeneRings': proposal_base_dict['num_benzene_rings_list'][i],
                        },
            }
            dict_proposals['next_molecules'].append(molecule)
        proposals_path = os.path.join(proposals_dir, f'proposals_{datetime.now().strftime("%H%M%S%f")[:-3]}.json')
        with open(proposals_path, "w") as outfile:
            json.dump(dict_proposals, outfile, cls=NpEncoder, indent=2)
        for k, v in proposal_base_dict.items():
            if isinstance(v, list):
                if k in stacked_proposal_dict:
                    stacked_proposal_dict[k] += v[:num_proposals_ui]
                else:
                    stacked_proposal_dict[k] = v[:num_proposals_ui]
                proposal_base_dict[k] = proposal_base_dict[k][num_proposals_ui:]  # delete mols that have been stacked
        return True  # made proposals
    else:
        return False  # did not make proposals


def load_proposal(load_flag, pt_dir, proposals_dir, evaluation_path, proposal_base_dict, stacked_proposal_dict,
                  num_inj, num_proposals_ui, logger=print):
    # if load_flag or len(proposals_dir) != 0:
    get_proposals(pt_dir, proposal_base_dict, logger=logger)
    load_flag = not proposal_stacked_json(proposals_dir, evaluation_path, proposal_base_dict,
                                          stacked_proposal_dict, num_inj, num_proposals_ui, logger=print)
    if load_flag is False:
        logger(f"num_inj {num_inj}, proposals loaded {load_flag}")
    return load_flag


import shutil


def load_annotation(annotation_dir, num_inj, num_total_positive_annotations, num_total_negative_annotations,
                    prepare_pref_data, stacked_proposal_dict, num_proposals_ui, load_flag, logger=print):
    annotation_path = os.path.join(annotation_dir, 'annotations.json')
    if os.path.isfile(annotation_path):
        with open(annotation_path, 'r') as f:
            time.sleep(0.5)
            annotations = json.load(f)
            like_ls = annotations['liked_ids']
            dislike_ls = annotations['disliked_ids']
        dst = os.path.join(annotation_dir, f'annotations_{num_inj}.json')
        shutil.move(annotation_path, dst)
        logger(f"Interaction {len(os.listdir(annotation_dir))}, positive annotations: {len(like_ls)}, "
               f"negative annotations: {len(dislike_ls)}")
        num_total_positive_annotations += len(like_ls)
        num_total_negative_annotations += len(dislike_ls)
        # time.sleep(0.05)
        if len(like_ls) + len(dislike_ls) == 0:
            logger("There are no annotations this time.")
        else:
            prepare_pref_data.storeData(stacked_proposal_dict, like_ls, dislike_ls)
        for k, v in stacked_proposal_dict.items():
            stacked_proposal_dict[k] = stacked_proposal_dict[k][num_proposals_ui:] #delete mols that have been used
        load_flag = True
    return load_flag, num_total_positive_annotations, num_total_negative_annotations


def proposal2json(proposals_path, evaluation_path, proposal_base_dict, num_total_positive_annotations,
                  num_total_negative_annotations, num_inj, num_proposals_ui, logger=print):
    if not os.path.isfile(proposals_path) and len(proposal_base_dict['mol_list']) >= num_proposals_ui:
        logger(f"{num_inj} making proposals")
        if num_total_positive_annotations >= 2 and num_total_negative_annotations >= 2 and os.path.isfile(evaluation_path):
            check_prompt = True  # evaluation condition
        else:
            check_prompt = False
        dict_proposals = {'next_molecules': [], 'check_prompt': check_prompt}
        for i in range(num_proposals_ui):
            molecule = {'id': i,
                        'smiles': proposal_base_dict['smiles_list'][i],
                        'url': proposal_base_dict['sdf_path_list'][i],
                        'png_url': proposal_base_dict['png_path_list'][i],
                        'metrics': {'Vina': proposal_base_dict['vina_score_list'][i],
                                    'QED': proposal_base_dict['qed_list'][i],
                                    'SA': proposal_base_dict['sa_list'][i],
                                    'Lipinski': proposal_base_dict['lipinski_list'][i],
                                    'NumAtoms': proposal_base_dict['num_atoms_list'][i],
                                    'NumBonds': proposal_base_dict['num_bonds_list'][i],
                                    'NumRings': proposal_base_dict['num_rings_list'][i],
                                    'NumBenzeneRings': proposal_base_dict['num_benzene_rings_list'][i],
                        },
            }
            dict_proposals['next_molecules'].append(molecule)
        with open(proposals_path, "w") as outfile:
            json.dump(dict_proposals, outfile, cls=NpEncoder, indent=2)
        return True  # made proposals
    else:
        return False  # did not make proposals


def evaluation2json(proposal_base_dict, evaluation_path, num_samples):
    dict_proposals = {'next_molecules': [], }
    for i in range(num_samples):
        molecule = {'id': i + 1,
                    'smiles': proposal_base_dict['smiles_list'][i],
                    'url': proposal_base_dict['sdf_path_list'][i],
                    'png_url': proposal_base_dict['png_path_list'][i],
                    'metrics': {'Vina': proposal_base_dict['vina_score_list'][i],
                                'QED': proposal_base_dict['qed_list'][i],
                                'SA': proposal_base_dict['sa_list'][i],
                                'Lipinski': proposal_base_dict['lipinski_list'][i],
                                'NumAtoms': proposal_base_dict['num_atoms_list'][i],
                                'NumBonds': proposal_base_dict['num_bonds_list'][i],
                                'NumRings': proposal_base_dict['num_rings_list'][i],
                                'NumBenzeneRings': proposal_base_dict['num_benzene_rings_list'][i],
                    },
        }
        dict_proposals['next_molecules'].append(molecule)
    with open(evaluation_path, "w") as outfile:
        json.dump(dict_proposals, outfile, cls=NpEncoder, indent=2)


class PreparePreferenceData:
    def __init__(self, **kwargs):
        self.sample_X1_pos_array = self.sample_X0_pos_array = torch.empty(0)
        self.sample_X1_ele_array = self.sample_X0_ele_array = torch.empty(0)
        self.sample_X1_bond_array = self.sample_X0_bond_array = torch.empty(0)
        self.target_pos_array = self.target_ele_array = self.target_bond_array = torch.empty(0)
        self.positive_indices, self.negative_indices = [], []
        self.raw_probs_positive, self.raw_probs_negative = [], []
        self.labels_values = torch.empty(0)
        self.n_steps = kwargs['n_steps']
        self.file_ls = kwargs['ckps_ls']
        self.file_ls_space = list(range(len(self.file_ls) - 1))
        self.n_atoms_per_mol = kwargs['n_atoms_per_mol']
        self.n_factor = kwargs['n_proposal_factor']
        self.n_positive_samples = kwargs['n_positive_samples']
        self.n_negative_samples = kwargs['n_negative_samples']
        self.n_samples_minibatch = self.n_positive_samples + self.n_negative_samples
        self.n_positive_samples_batch = kwargs['n_positive_samples'] * kwargs['n_proposal_factor']
        self.n_negative_samples_batch = kwargs['n_negative_samples'] * kwargs['n_proposal_factor']
        self.n_mols_batch = self.n_positive_samples_batch + self.n_negative_samples_batch
        self.variance_trials = kwargs['variance_trials']
        self.query, self.temperature = kwargs['query'], kwargs['temperature']
        self.ele_emb_dim = kwargs['ele_emb_dim']
        self.n_bond_edges_per_mol = kwargs['num_bond_edges_per_ligand']
        self.bond_emb_dim = kwargs['bond_emb_dim']
        # new proposals will have larger probabilities to be chosen
        self.n_inj_per_round = kwargs['num_injections_per_round']
        self.raw_probs = np.linspace(0.5, 1, self.n_inj_per_round)
        self.num_timesteps = kwargs['num_timesteps']
        if kwargs['all_pref_data']:
            self.sample_all_data()

    def sample_preference_data(self, **kwargs):
        target_pos_pair, target_ele_emb_pair, target_bond_emb_pair, \
        sample_X1_pos, sample_X0_pos, sample_X1_ele_emb, sample_X0_ele_emb, \
        sample_X1_bond_emb, sample_X0_bond_emb, labels, real_labels = prepare_proposals(
            self.n_positive_samples_batch, self.n_negative_samples_batch,
            query=self.query, temperature=self.temperature, ckps_ls=self.file_ls[:-1],
            variance_trials=self.variance_trials, ckp_id_records=self.file_ls_space)
        self.target_pos_array = torch.cat([self.target_pos_array, target_pos_pair.reshape(self.n_mols_batch, self.n_atoms_per_mol, 3)], dim=0)
        self.target_ele_array = torch.cat([self.target_ele_array, target_ele_emb_pair.reshape(self.n_mols_batch, self.n_atoms_per_mol, self.ele_emb_dim)], dim=0)
        self.target_bond_array = torch.cat([self.target_bond_array, target_bond_emb_pair.reshape(self.n_mols_batch, self.n_bond_edges_per_mol, self.bond_emb_dim)], dim=0)
        self.sample_X1_pos_array = torch.cat([self.sample_X1_pos_array, sample_X1_pos.reshape(self.n_mols_batch, self.n_atoms_per_mol, 3)], dim=0)
        self.sample_X0_pos_array = torch.cat([self.sample_X0_pos_array, sample_X0_pos.reshape(self.n_mols_batch, self.n_atoms_per_mol, 3)], dim=0)
        self.sample_X1_ele_array = torch.cat([self.sample_X1_ele_array, sample_X1_ele_emb.reshape(self.n_mols_batch, self.n_atoms_per_mol, self.ele_emb_dim)], dim=0)
        self.sample_X0_ele_array = torch.cat([self.sample_X0_ele_array, sample_X0_ele_emb.reshape(self.n_mols_batch, self.n_atoms_per_mol, self.ele_emb_dim)], dim=0)
        self.sample_X1_bond_array = torch.cat([self.sample_X1_bond_array, sample_X1_bond_emb.reshape(self.n_mols_batch, self.n_bond_edges_per_mol, self.bond_emb_dim)], dim=0)
        self.sample_X0_bond_array = torch.cat([self.sample_X0_bond_array, sample_X0_bond_emb.reshape(self.n_mols_batch, self.n_bond_edges_per_mol, self.bond_emb_dim)], dim=0)
        self.labels_values = torch.cat([self.labels_values, real_labels])
        self.raw_probs_positive += [self.raw_probs[kwargs['inject_counter']]] * self.n_positive_samples_batch
        self.probs_positive = np.array(self.raw_probs_positive) / np.sum(self.raw_probs_positive)
        self.raw_probs_negative += [self.raw_probs[kwargs['inject_counter']]] * self.n_negative_samples_batch
        self.probs_negative = np.array(self.raw_probs_negative) / np.sum(self.raw_probs_negative)
        num_proposals_prev = len(self.positive_indices) + len(self.negative_indices)
        self.positive_indices += list(range(num_proposals_prev, num_proposals_prev + self.n_positive_samples_batch))
        self.negative_indices += list(range(num_proposals_prev + self.n_positive_samples_batch, num_proposals_prev + self.n_mols_batch))

    def sample_pref_data_minibatch(self, device):
        positive_choices = np.random.choice(self.positive_indices, self.n_positive_samples, replace=False, p=self.probs_positive) if len(self.probs_positive) != 0 else np.array([])
        negative_choices = np.random.choice(self.negative_indices, self.n_negative_samples, replace=False, p=self.probs_negative) if len(self.probs_negative) != 0 else np.array([])
        target_pos_pair = torch.cat((self.target_pos_array[positive_choices], self.target_pos_array[negative_choices]), dim=0).reshape(-1, 3)
        target_ele_emb_pair = torch.cat((self.target_ele_array[positive_choices], self.target_ele_array[negative_choices]), dim=0).reshape(-1, self.ele_emb_dim)
        target_bond_emb_pair = torch.cat((self.target_bond_array[positive_choices], self.target_bond_array[negative_choices]), dim=0).reshape(-1, self.bond_emb_dim)
        sample_X1_pos = torch.cat((self.sample_X1_pos_array[positive_choices], self.sample_X1_pos_array[negative_choices]), dim=0).reshape(-1, 3)
        sample_X0_pos = torch.cat((self.sample_X0_pos_array[positive_choices], self.sample_X0_pos_array[negative_choices]), dim=0).reshape(-1, 3)
        sample_X1_ele_emb = torch.cat((self.sample_X1_ele_array[positive_choices], self.sample_X1_ele_array[negative_choices]), dim=0).reshape(-1, self.ele_emb_dim)
        sample_X0_ele_emb = torch.cat((self.sample_X0_ele_array[positive_choices], self.sample_X0_ele_array[negative_choices]), dim=0).reshape(-1, self.ele_emb_dim)
        sample_X1_bond_emb = torch.cat((self.sample_X1_bond_array[positive_choices], self.sample_X1_bond_array[negative_choices]), dim=0).reshape(-1, self.bond_emb_dim)
        sample_X0_bond_emb = torch.cat((self.sample_X0_bond_array[positive_choices], self.sample_X0_bond_array[negative_choices]), dim=0).reshape(-1, self.bond_emb_dim)
        target_pos_pair = target_pos_pair.repeat(self.n_steps, 1)
        target_ele_emb_pair = target_ele_emb_pair.repeat(self.n_steps, 1)
        target_bond_emb_pair = target_bond_emb_pair.repeat(self.n_steps, 1)
        self.sample_X1_pos = sample_X1_pos.repeat(self.n_steps, 1)
        self.sample_X0_pos = sample_X0_pos.repeat(self.n_steps, 1)
        self.sample_X1_ele_emb = sample_X1_ele_emb.repeat(self.n_steps, 1)
        self.sample_X0_ele_emb = sample_X0_ele_emb.repeat(self.n_steps, 1)
        self.sample_X1_bond_emb = sample_X1_bond_emb.repeat(self.n_steps, 1)
        self.sample_X0_bond_emb = sample_X0_bond_emb.repeat(self.n_steps, 1)
        labels = torch.zeros(self.n_samples_minibatch).to(device)
        labels[:self.n_positive_samples] = torch.softmax(self.labels_values[positive_choices] * self.temperature, dim=-1)
        return target_pos_pair, target_ele_emb_pair, target_bond_emb_pair, labels

    def get_xt(self, t, timestep_sampler, noise):
        Xt_pos_pair, Xt_ele_emb_pair, Xt_bond_emb_pair, t_int, t_bond = get_Xt(self.sample_X1_pos,
                                                                               self.sample_X0_pos,
                                                                               self.sample_X1_ele_emb,
                                                                               self.sample_X0_ele_emb,
                                                                               self.sample_X1_bond_emb,
                                                                               self.sample_X0_bond_emb,
                                                                               self.num_timesteps, t=t,
                                                                               time_sampler=timestep_sampler,
                                                                               num_steps=self.n_steps,
                                                                               noise=noise)
        return Xt_pos_pair, Xt_ele_emb_pair, Xt_bond_emb_pair, t_int, t_bond

    def sample_all_data(self):
        self.raw_probs[:] = 1
        for i in range(self.n_inj_per_round):
            self.sample_preference_data(inject_counter=i)


def get_batch_vec(elements_per_sample, batch_size):
    batch_info = torch.tensor(range(batch_size)).repeat(elements_per_sample, 1).t().flatten()
    return batch_info

def perterb_X0(**kwargs):
    batch_size = kwargs['batch_size']
    if kwargs['noise_type'] == 'VP':
        noise_level = np.sqrt(kwargs['noise_level'])
        signal_level = np.sqrt(1.0 - kwargs['noise_level'])
    elif kwargs['noise_type'] == 'VE':
        noise_level = kwargs['noise_level']
        signal_level = 1.0
    else:
        raise NotImplementedError(f"Invalid {kwargs['noise_type']}")
    X0_pos = kwargs['X0_pos']
    X0_element_embedding = kwargs['X0_element_embedding']
    X0_bond_embedding = kwargs['X0_bond_embedding']
    batch = get_batch_vec(X0_pos.shape[0], batch_size)
    noise = torch.randn(X0_pos.shape[0] * batch_size, X0_pos.shape[1], device=X0_pos.device)
    X0_pos = signal_level * X0_pos.repeat(batch_size, 1) + noise_level * noise
    X0_pos = subtract_mean(X0_pos, batch=batch.to(X0_pos.device))
    noise = torch.randn(X0_element_embedding.shape[0] * batch_size, X0_element_embedding.shape[1],
                        device=X0_element_embedding.device)
    X0_element_embedding = signal_level * X0_element_embedding.repeat(batch_size, 1) + noise_level * noise
    noise = torch.randn(X0_bond_embedding.shape[0] * batch_size, X0_bond_embedding.shape[1],
                        device=X0_bond_embedding.device)
    X0_bond_embedding = signal_level * X0_bond_embedding.repeat(batch_size, 1) + noise_level * noise
    return X0_pos, X0_element_embedding, X0_bond_embedding


class NpEncoder(json.JSONEncoder):
    '''
    https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable
    '''
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
