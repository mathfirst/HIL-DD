import numpy as np
import time, torch
from utils.evaluate import CalculateBenchmarks, print_flattened_result
from .util_data import extract_data, perterb_X0
from .util_flow import subtract_mean, get_all_bond_edges, measure_straightness
from utils.analyze import check_stability
import utils.transforms as trans
from scipy import integrate
import torch.linalg as la
from tqdm import tqdm

def sample_ode(model, protein_pos, protein_ele, protein_amino_acid, protein_is_backbone, z_pos, z_feat, num_steps,
               batch_protein=None, batch_ligand=None, logger=print, bond_features=None, bond_edges=None, device='cpu',
               num_spacing_steps=False, get_bond_types=False, trajectory=False, return_v=False,
               t_sampling_strategy='uniform', cls_fn=None, s=0.1, batchsize=1, guidance_type='MSE', return_g=False):
    ### NOTE: Use Euler method to sample from the learned flow
    model.eval()
    if cls_fn and s > 0:
        cls_fn.eval()
    with torch.no_grad():
        dt = 1. / num_steps
        if num_spacing_steps:
            if t_sampling_strategy == 'quadratic':
                c = num_steps / num_spacing_steps ** 1.5
                step_indices = np.arange(0, num_spacing_steps + 1)
                spaced_steps = (c * step_indices ** 1.5).astype(int)
                if spaced_steps[-1] != num_steps:
                    logger(f"The last step is {spaced_steps[-1]}, so we need to set it to be {num_steps}.")
                    spaced_steps[-1] = num_steps
            else:  # uniform spacing by default
                spaced_steps = np.linspace(0, num_steps, num_spacing_steps + 1, dtype=int)
            tqdm_steps = num_spacing_steps
        else:
            spaced_steps = None
            tqdm_steps = num_steps
        num_atoms_per_ligand = int(z_pos.shape[0] / batchsize)
        if trajectory:
            traj, v_pos_ls, v_ele_ls, v_bond_ls = [], [], [], []  # to store the trajectory
        v_pos_ls, v_ele_ls, v_bond_ls = [], [], []
        if return_g:
            grad_pos_norm_ls, grad_ele_norm_ls, grad_bond_norm_ls = [], [], []
        if bond_features is not None:
            bond_features, bond_edges = bond_features.to(device), bond_edges.to(device)
        z_pos, z_feat = z_pos.to(device), z_feat.to(device)
        protein_pos, protein_ele = protein_pos.to(device), protein_ele.to(device)
        protein_amino_acid, protein_is_backbone = protein_amino_acid.to(device), protein_is_backbone.to(device)
        for i in tqdm(range(tqdm_steps), desc='sampling', total=tqdm_steps):
            # print('sampling time step', i, sep=' ')
            if spaced_steps is not None:
                dt = (spaced_steps[i + 1] - spaced_steps[i]) * 1. / num_steps
                i = int(spaced_steps[i])
            t_int = torch.as_tensor(i).reshape(-1, 1).to(device)
            t = t_int.repeat(z_pos.shape[0], 1).squeeze()
            if bond_features is not None:
                t_bond = t_int.repeat(bond_features.shape[0], 1).squeeze()
            else:
                t_bond = None
            pred_pos_v, pred_ele_v, pred_bond_v = \
                model(protein_pos, protein_ele, protein_amino_acid, protein_is_backbone,
                      z_pos, z_feat, t, batch_protein, batch_ligand, num_proposals=batchsize,
                      bond_features=bond_features, bond_indices=bond_edges,
                      num_atoms_per_ligand=num_atoms_per_ligand, t_bond=t_bond)
            if return_v:
                v_pos_ls.append(pred_pos_v)
                v_ele_ls.append(pred_ele_v)
                v_bond_ls.append(pred_bond_v)
            if cls_fn is not None and s > 0:
                with torch.enable_grad():
                    pos = z_pos.detach().clone().requires_grad_(True)
                    ele = z_feat.detach().clone().requires_grad_(True)
                    if bond_features is not None:
                        bond = bond_features.detach().clone().requires_grad_(True)
                    else:
                        bond = None
                    logits = cls_fn(pos, ele, t, bond_features=bond, num_atoms_per_ligand=num_atoms_per_ligand,
                                    t_bond=t_bond, batch_ligand=batch_ligand, protein_positions=protein_pos,
                                    protein_ele=protein_ele, protein_amino_acid=protein_amino_acid,
                                    protein_is_backbone=protein_is_backbone, num_proposals=batchsize,
                                    n_steps_per_iter=1)
                    pred = GUIDANCE_FUNC[guidance_type](logits)

                    if bond_features is not None:
                        gradients = torch.autograd.grad(pred, [pos, ele, bond])
                    else:
                        gradients = torch.autograd.grad(pred, [pos, ele])
            else:
                gradients = (0.0, 0.0, 0.0)

            if return_g and cls_fn is not None:
                grad_pos_norm_ls.append(la.norm(gradients[0].detach()))
                grad_ele_norm_ls.append(la.norm(gradients[1].detach()))
                grad_bond_norm_ls.append(la.norm(gradients[2].detach()))
            z_pos = z_pos.to(device) + pred_pos_v * dt + gradients[0] * s
            z_feat = z_feat.to(device) + pred_ele_v * dt + gradients[1] * s
            if bond_features is not None:
                bond_features = bond_features.to(device) + pred_bond_v * dt + gradients[2] * s

            if trajectory:
                dist_betw_X1_feat_and_embedding = torch.cdist(z_feat, model.ligand_element_embedding.weight.data, p=2)
                pred_elements = torch.argmin(dist_betw_X1_feat_and_embedding, dim=-1)
                traj.append((z_pos.detach().cpu(), pred_elements.cpu()))

        # calculate the distances between predicted features and the constructed feature embeddings
        dist_betw_X1_feat_and_embedding = torch.cdist(z_feat, model.ligand_element_embedding.weight.data,
                                                      p=2)  # * model.emb_scale_factor
        pred_elements = torch.argmin(dist_betw_X1_feat_and_embedding, dim=-1)
        if bond_features is not None and get_bond_types:
            dist_betw_bond_feat_and_embedding = torch.cdist(bond_features, model.bond_type_embedding.weight.data,
                                                            p=2)  # * model.emb_scale_factor
            pred_bond_types = torch.argmin(dist_betw_bond_feat_and_embedding, dim=-1)
        else:
            pred_bond_types = None

        # pred_elements = torch.as_tensor([map_ind_ele[key.item()] for key in pred_elements], device=device)
        # logger.info("------- The predicted positions before scaling back and predicted embedding distances are:")
        # for ind in range(dist_betw_X1_feat_and_embedding.shape[0]):
        #     # logger.info(f"atom {ind}, atomic position before scaling back: {z_pos.cpu()[ind]}")
        #     logger.info(f"argmin: {pred_elements.cpu()[ind]}, distances: {dist_betw_X1_feat_and_embedding.cpu()[ind]}")
    if trajectory:
        return traj

    return z_pos, pred_elements, pred_bond_types, z_feat, bond_features, [v_pos_ls, v_ele_ls,
                                                                          v_bond_ls]  # , [grad_pos_norm_ls, grad_ele_norm_ls, grad_bond_norm_ls]


def sampling_val(model, batch, pocket_id, pos_scale, result_dir, logger, device, ProteinElement2IndexDict,
                 num_timesteps=1000, mode='add_aromatic_wo_h', num_samples=10,
                 cal_vina_score=True, num_spacing_steps=None, bond_emb=False,
                 cal_straightness=False, t_sampling_strategy='uniform',
                 drop_unconnected_mol=True, cls_fn=None, s=0.1, batch_size=10,
                 guidance_type='BCE', protein_pdbqt_file_path='',  **kwargs):
    time_consumption = []
    protein_pos, protein_ele, protein_amino_acid, protein_is_backbone, ligand_pos, ligand_ele \
        = extract_data(batch, ProteinElement2IndexDict)
    protein_batch = torch.zeros(protein_ele.shape[0], dtype=int).to(device)  # batch['protein_element_batch'].to(device)
    protein_pos, mean_vec = subtract_mean(protein_pos.to(device), batch=protein_batch, verbose=True)
    protein_pos *= pos_scale
    calculator = CalculateBenchmarks(pocket_id=pocket_id, save_dir=result_dir,
                                     aromatic=True, logger=logger, cal_vina_score=cal_vina_score)
    if kwargs['num_atoms'] == 'reference':
        num_atoms_per_ligand = ligand_pos.shape[0]
    else:
        num_atoms_per_ligand = int(kwargs['num_atoms'])
    if 'stability_rate' in kwargs:
        stability_rate = kwargs['stability_rate']
    else:
        stability_rate = 0.7
    num_stable_atoms_threshold = int(num_atoms_per_ligand * stability_rate)  # the results in our paper do not involve this constraint.
    while True:
        batchsize = min(batch_size, 2 * (num_samples - len(calculator.result_dict['mol_list'])))
        logger.info(f"current batchsize: {batchsize}")
        ligand_batch = torch.tensor(range(batchsize)).repeat(num_atoms_per_ligand, 1).t().flatten()
        if bond_emb:
            bond_edges = model.get_ligand_bond_types_all(num_ligand_atoms=num_atoms_per_ligand,
                                                         ligand_bond_indices=None,
                                                         ligand_bond_types=None, device=device)
            all_bond_edges = get_all_bond_edges(bond_edges, batchsize, num_atoms_per_ligand)
        else:
            all_bond_edges = None
        num_edges = all_bond_edges.shape[1] if bond_emb else None
        if 'X0_pos' in kwargs:
            logger.info("loading predefined X0")
            X0_pos, X0_element_embedding, X0_bond_embedding = perterb_X0(batch_size=batchsize, **kwargs)
        else:
            logger.info("generating random X0")
            X0_pos, X0_element_embedding, X0_bond_embedding \
                = model.get_X0(num_ligand_atoms=num_atoms_per_ligand, num_edges=num_edges,
                               batch_ligands=ligand_batch.to(device), batchsize=batchsize)
        sampling_start = time.time()
        X1_pos, X1_ele, pred_bond_types, X1_ele_emb, X1_bond_emb, v_ls = sample_ode(model, protein_pos, protein_ele,
                                                                                    protein_amino_acid,
                                                                                    protein_is_backbone, X0_pos,
                                                                                    X0_element_embedding,
                                                                                    num_steps=num_timesteps,
                                                                                    batch_protein=protein_batch,
                                                                                    batch_ligand=ligand_batch,
                                                                                    device=device,
                                                                                    bond_features=X0_bond_embedding,
                                                                                    batchsize=batchsize,
                                                                                    bond_edges=all_bond_edges,
                                                                                    logger=logger,
                                                                                    num_spacing_steps=num_spacing_steps,
                                                                                    return_v=cal_straightness,
                                                                                    t_sampling_strategy=t_sampling_strategy,
                                                                                    cls_fn=cls_fn, s=s,
                                                                                    guidance_type=guidance_type,
                                                                                    return_g=False)
        time_consumed = time.time() - sampling_start
        assert pos_scale > 0, f"pos_scale is supposed to be a positive scalar instead of {pos_scale}."
        if cal_straightness:
            pos_straightness = measure_straightness(X0_pos, X1_pos, v_ls[0])
            ele_straightness = measure_straightness(X0_element_embedding, X1_ele_emb, v_ls[1])
            bond_straightness = measure_straightness(X0_bond_embedding, X1_bond_emb, v_ls[2]) if bond_emb else 10
            logger.info(f"pos_straightness: {pos_straightness:.3f}, ele_straightness: {ele_straightness:.3f}, "
                        f"bond_straightness: {bond_straightness:.3f}")
        X1_pos_copy = X1_pos.detach().clone().cpu()
        X1_pos = X1_pos.cpu() / pos_scale + mean_vec.cpu()  # scale back and then move back to the original gravity center
        pred_atom_type = trans.get_atomic_number_from_index(X1_ele, mode=mode)
        xyz = X1_pos.tolist()
        X0_pos = X0_pos.reshape(batchsize, -1, 3)
        X0_element_embedding = X0_element_embedding.reshape(batchsize, -1, X0_element_embedding.shape[1])
        X1_pos_copy = X1_pos_copy.reshape(batchsize, -1, 3)
        X1_ele_emb = X1_ele_emb.reshape(batchsize, -1, X1_ele_emb.shape[1])
        if bond_emb:
            X0_bond_embedding = X0_bond_embedding.reshape(batchsize, -1, X0_bond_embedding.shape[1])
            X1_bond_emb = X1_bond_emb.reshape(batchsize, -1, X1_bond_emb.shape[1])
        X1_pos = X1_pos.reshape(batchsize, -1, 3)
        num_generated_mols = 0
        for idx in range(batchsize):
            start = idx * num_atoms_per_ligand
            end = (idx + 1) * num_atoms_per_ligand
            atom_types = pred_atom_type[start:end]
            try:
                r_stable = check_stability(X1_pos[idx], atom_types)
                logger.info(f"stability: {r_stable}")
                if r_stable[1] < num_stable_atoms_threshold:
                    logger.info("This molecule is not stable enough to save.")
                    continue
            except Exception as e:
                r_stable = None
                logger.info(f"Failed when calculating stability. {e}")
            return_value_recon_without_aromatic = calculator.reconstruct_mol(xyz=xyz[start:end],
                                                                             atom_nums=atom_types,
                                                                             aromatic=None, bond_type=None,
                                                                             bond_index=None, stable=r_stable,
                                                                             drop_unconnected_mol=drop_unconnected_mol,
                                                                             protein_pdbqt_dir=kwargs['protein_pdbqt_dir'],
                                                                             protein_pdbqt_file_path=protein_pdbqt_file_path)
            if return_value_recon_without_aromatic is None:
                logger.info(f"The mol reconstructed without aromatic info is invalid.")
                continue
            if return_value_recon_without_aromatic == 'unconnected':
                continue
            if not return_value_recon_without_aromatic['contain_benzene_rings'][-1]:  # if the current mol does not contain benzene rings
                logger.info("This mol does not contain benzene rings.")
            else:
                logger.info(
                    f"benzene rings found. fused benzene rings {return_value_recon_without_aromatic['contain_fused_benzene_rings'][-1]}")
            time_consumption.append(time_consumed)
            logger.info(f"sampling time: {time_consumption[-1]:.3f}, {np.mean(time_consumption):.3f} \u00B1 "
                        f"{np.std(time_consumption):.2f} with {len(time_consumption)} samples.")
            # calculator.result_dict['time_list'].append(time_consumption[-1])  # if batchsize is not varying
            if bond_emb:
                calculator.save_X0(X0_pos[idx], X0_element_embedding[idx], X1_pos_copy[idx], X1_ele_emb[idx],
                                   X0_bond_embedding[idx], X1_bond_emb[idx])
            else:
                calculator.save_X0(X0_pos[idx], X0_element_embedding[idx], X1_pos_copy[idx], X1_ele_emb[idx])
            num_generated_mols = len(calculator.result_dict['mol_list'])
            if num_generated_mols >= num_samples:
                calculator.result_dict['time_list'] = [time_consumption[0]]  # because batchsize is varying
                break

        if num_generated_mols >= num_samples:
            logger.info(f"We are done with generating {num_generated_mols} samples for Pocket {pocket_id}.")
            logger.info(f"The metrics of Pocket {pocket_id} with {num_generated_mols} ligands evaluated.")
            calculator.print_mean_std(desc="Not using aromatic and bond info when reconstructing.\n")
            torch.save(calculator.result_dict, calculator.pt_filename)
            return calculator


GUIDANCE_FUNC = {'BCE': lambda x: torch.log(torch.sigmoid(x)).sum(),
                 'MSE': lambda x: -((x - 1) ** 2).sum(),
                 'MSE_hinge': lambda x: -torch.sum((x - 1) ** 2 * (x.detach() < 1.0).long()),
                 'MAE': lambda x: -torch.sum(torch.abs(x - 1)),
                 'MAE_hinge': lambda x: torch.sum(torch.minimum(x - 1, torch.tensor(0)))}


def sample4pocket(model, val_loader, pos_scale, result_dir, logger, device, ProteinElement2IndexDict,
                  num_timesteps=1000, num_pockets=5, mode='add_aromatic_wo_h', num_samples=10,
                  cal_vina_score=True, all_val_pockets=False, num_spacing_steps=None, bond_emb=False,
                  cal_straightness=False, t_sampling_strategy='uniform', starting_pocket_id=0,
                  drop_unconnected_mol=True, sampling_method='Euler', cls_fn=None, s=0.1, batchsize=10,
                  guidance_type='MSE', protein_pdbqt_file_path='', **kwargs):
    result_all_pockets, result_aromatic_all_pockets, result_bond_all_pockets, \
        result_aromatic_and_bond_all_pockets = [], [], [], []
    diversity_list, diversity_aromatic_list, diversity_bond_list, \
        diversity_aromatic_and_bond_list = [], [], [], []
    time_consumption = []
    batch_size = batchsize  # make a copy such that we can resue this value
    for pocket_id, batch in enumerate(val_loader):
        if pocket_id != num_pockets and all_val_pockets is False:  # here we only generate samples using Pocket [num_pockets]
            logger.info(f"We skip pocket {pocket_id}.")
            continue
        if pocket_id < starting_pocket_id:
            logger.info(f"We skip pocket {pocket_id}.")
            continue
        protein_pos, protein_ele, protein_amino_acid, protein_is_backbone, ligand_pos, ligand_ele \
            = extract_data(batch, ProteinElement2IndexDict)
        protein_batch, ligand_batch = batch.protein_element_batch.to(device), batch.ligand_element_batch.to(device)
        protein_pos, mean_vec = subtract_mean(protein_pos.to(device), batch=protein_batch, verbose=True)
        protein_pos *= pos_scale
        calculator = CalculateBenchmarks(pocket_id=pocket_id, save_dir=result_dir,
                                         aromatic=True, logger=logger, cal_vina_score=cal_vina_score)
        num_atoms_per_ligand = ligand_pos.shape[0]
        while True:
            batchsize = min(batch_size, 2 * (num_samples - len(calculator.result_dict['mol_list'])))
            logger.info(f"current batch_size: {batchsize}")
            ligand_batch = torch.tensor(range(batchsize)).repeat(num_atoms_per_ligand, 1).t().flatten()
            if bond_emb:
                bond_edges = model.get_ligand_bond_types_all(num_ligand_atoms=ligand_pos.shape[0],
                                                             ligand_bond_indices=None,
                                                             ligand_bond_types=None, device=device)
                all_bond_edges = get_all_bond_edges(bond_edges, batchsize, num_atoms_per_ligand)
            else:
                bond_edges = all_bond_edges = None
            num_edges = all_bond_edges.shape[1] if bond_emb else None
            if 'X0_pos' in kwargs:
                logger.info("loading predefined X0")
                X0_pos, X0_element_embedding, X0_bond_embedding = perterb_X0(batch_size=batchsize, **kwargs)
            else:
                logger.info("generating random X0")
                X0_pos, X0_element_embedding, X0_bond_embedding \
                    = model.get_X0(num_ligand_atoms=ligand_pos.shape[0], num_edges=num_edges,
                                   batch_ligands=ligand_batch.to(device), batchsize=batchsize)
            sampling_start = time.time()
            if sampling_method == 'rk45':
                X1_pos, X1_ele, X1_ele_emb, X1_bond_emb, success = rk45_sampler(model, protein_pos,
                                                                                protein_ele,
                                                                                protein_amino_acid,
                                                                                protein_is_backbone,
                                                                                num_timesteps=num_timesteps,
                                                                                z_pos=X0_pos,
                                                                                z_ele_embedding=X0_element_embedding,
                                                                                batch_protein=protein_batch,
                                                                                batch_ligand=ligand_batch,
                                                                                device=device,
                                                                                z_bond_embedding=X0_bond_embedding,
                                                                                bond_edges=bond_edges,
                                                                                logger=logger.info,
                                                                                num_ligand_atoms=ligand_pos.shape[0])
                if success is not True:
                    logger.info("rk45 solver failed.")
                    continue
            else:
                X1_pos, X1_ele, pred_bond_types, X1_ele_emb, X1_bond_emb, v_ls = sample_ode(model, protein_pos,
                                                                                            protein_ele,
                                                                                            protein_amino_acid,
                                                                                            protein_is_backbone, X0_pos,
                                                                                            X0_element_embedding,
                                                                                            num_steps=num_timesteps,
                                                                                            batch_protein=protein_batch,
                                                                                            batch_ligand=ligand_batch,
                                                                                            device=device,
                                                                                            bond_features=X0_bond_embedding,
                                                                                            batchsize=batchsize,
                                                                                            bond_edges=all_bond_edges,
                                                                                            logger=logger,
                                                                                            num_spacing_steps=num_spacing_steps,
                                                                                            return_v=cal_straightness,
                                                                                            t_sampling_strategy=t_sampling_strategy,
                                                                                            cls_fn=cls_fn, s=s,
                                                                                            guidance_type=guidance_type,
                                                                                            return_g=False)
            time_consumed = time.time() - sampling_start
            assert pos_scale > 0, f"pos_scale is supposed to be a positive scalar instead of {pos_scale}."
            if cal_straightness:
                pos_straightness = measure_straightness(X0_pos, X1_pos, v_ls[0])
                ele_straightness = measure_straightness(X0_element_embedding, X1_ele_emb, v_ls[1])
                bond_straightness = measure_straightness(X0_bond_embedding, X1_bond_emb, v_ls[2])
                logger.info(f"pos_straightness: {pos_straightness:.3f}, ele_straightness: {ele_straightness:.3f}, "
                            f"bond_straightness: {bond_straightness:.3f}")
            X1_pos_copy = X1_pos.detach().clone().cpu()
            X1_pos = X1_pos.cpu() / pos_scale + mean_vec.cpu()  # scale back and then move back to the original gravity center
            pred_atom_type = trans.get_atomic_number_from_index(X1_ele, mode=mode)
            xyz = X1_pos.tolist()
            X0_pos = X0_pos.reshape(batchsize, -1, 3)
            X0_element_embedding = X0_element_embedding.reshape(batchsize, -1, X0_element_embedding.shape[1])
            X0_bond_embedding = X0_bond_embedding.reshape(batchsize, -1, X0_bond_embedding.shape[1])
            X1_pos_copy = X1_pos_copy.reshape(batchsize, -1, 3)
            X1_ele_emb = X1_ele_emb.reshape(batchsize, -1, X1_ele_emb.shape[1])
            X1_bond_emb = X1_bond_emb.reshape(batchsize, -1, X1_bond_emb.shape[1])
            X1_pos = X1_pos.reshape(batchsize, -1, 3)
            for idx in range(batchsize):
                start = idx * num_atoms_per_ligand
                end = (idx + 1) * num_atoms_per_ligand
                atom_types = pred_atom_type[start:end]
                try:
                    r_stable = check_stability(X1_pos[idx], atom_types)
                    logger.info(f"stability: {r_stable}")
                except Exception as e:
                    r_stable = None
                    logger.info(f"Failed when calculating stability. {e}")
                return_value_recon_without_aromatic = calculator.reconstruct_mol(xyz=xyz[start:end],
                                                                                 atom_nums=atom_types,
                                                                                 aromatic=None, bond_type=None,
                                                                                 bond_index=None, stable=r_stable,
                                                                                 drop_unconnected_mol=drop_unconnected_mol,
                                                                                 protein_pdbqt_file_path=protein_pdbqt_file_path)
                if return_value_recon_without_aromatic is None:
                    logger.info(f"The mol reconstructed without aromatic info is invalid.")
                    continue
                if return_value_recon_without_aromatic == 'unconnected':
                    continue
                if not return_value_recon_without_aromatic['contain_benzene_rings'][
                    -1]:  # if the current mol does not contain benzene rings
                    logger.info("This mol does not contain benzene rings.")
                else:
                    logger.info(
                        f"benzene rings found. fused benzene rings {return_value_recon_without_aromatic['contain_fused_benzene_rings'][-1]}")

                time_consumption.append(time_consumed)
                logger.info(f"sampling time: {time_consumption[-1]:.3f}, {np.mean(time_consumption):.3f} \u00B1 "
                            f"{np.std(time_consumption):.2f} with {len(time_consumption)} samples.")
                calculator.result_dict['time_list'].append(time_consumption[-1])
                calculator.save_X0(X0_pos[idx], X0_element_embedding[idx], X1_pos_copy[idx], X1_ele_emb[idx],
                                   X0_bond_embedding[idx], X1_bond_emb[idx])
            num_generated_mols = len(calculator.result_dict['mol_list'])
            if num_generated_mols >= num_samples:
                logger.info(f"We are done with generating {num_generated_mols} samples for Pocket {pocket_id}.")
                logger.info(f"The metrics of Pocket {pocket_id} with {num_generated_mols} ligands evaluated.")
                calculator.print_mean_std(desc="Not using aromatic and bond info when reconstructing.\n")
                diversity_list.append(calculator.stat_dict['diversity'])
                torch.save(calculator.result_dict, calculator.pt_filename)
                result_all_pockets.append(calculator.result_dict)
                if all_val_pockets is False:
                    return calculator
                else:
                    break

        # calculate benchmarks for all pockets we have used for generating ligands
        logger.info(f'The {num_samples * (pocket_id + 1)} generated ligands conditioned on {pocket_id + 1} pockets.')
        print_flattened_result(result_all_pockets, desc='wo_aromatic', logger=logger)


def to_flattened_numpy(x):
    return x.detach().reshape(-1,).cpu().numpy()

def from_flattened_numpy(x, shape):
    return torch.from_numpy(x).reshape(shape).type(torch.float32)

def rk45_sampler(model, protein_pos, protein_ele, protein_amino_acid, protein_is_backbone, num_timesteps,
                 z_pos=None, z_ele_embedding=None, z_bond_embedding=None, bond_edges=None, batch_protein=None,
                 batch_ligand=None, logger=print, device='cpu', num_ligand_atoms=None, num_bond_edges=None):
    '''
    Explicit Runge-Kutta method of order 5(4).
    Parameters:
        model: The velocity flow network
        z: The initial states
    Reference:
        1. https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.RK45.html#scipy.integrate.RK45
        2. https://github.com/gnobitab/RectifiedFlow/blob/main/ImageGeneration/sampling.py
    '''

    with torch.no_grad():
        if num_bond_edges is None and num_ligand_atoms is not None:
            num_bond_edges = num_ligand_atoms * (num_ligand_atoms - 1)
        if z_pos is None:
            assert num_ligand_atoms is not None, 'num_ligand_atoms should be specified.'
            z_pos, z_ele_embedding, z_bond_embedding \
                = model.get_X0(num_ligand_atoms=num_ligand_atoms, num_edges=num_bond_edges)

        pos_shape, ele_emb_shape, bond_emb_shape = z_pos.shape, z_ele_embedding.shape, z_bond_embedding.shape
        len_pos, len_ele_emb, len_bond_emb = np.prod(pos_shape), np.prod(ele_emb_shape), np.prod(bond_emb_shape)
        flattened_pos = to_flattened_numpy(z_pos)
        flattened_ele_emb = to_flattened_numpy(z_ele_embedding)
        flattened_bond_emb = to_flattened_numpy(z_bond_embedding)
        flattened_z = np.concatenate([flattened_pos, flattened_ele_emb, flattened_bond_emb])

        def ode_func(t, x):
            z_pos = from_flattened_numpy(x[:len_pos], pos_shape)
            z_ele_emb = from_flattened_numpy(x[len_pos:-len_bond_emb], ele_emb_shape)
            z_bond_emb = from_flattened_numpy(x[-len_bond_emb:], bond_emb_shape)
            t_int = torch.as_tensor(int(t * (num_timesteps - 1))).reshape(-1, 1)
            t = t_int.repeat(num_ligand_atoms, 1).squeeze()
            t_bond = t_int.repeat(num_bond_edges, 1).squeeze()
            v_pos, v_ele_emb, v_bond_emb = model(protein_pos.to(device), protein_ele.to(device),
                      protein_amino_acid.to(device), protein_is_backbone.to(device),
                      z_pos.to(device), z_ele_emb.to(device), t.to(device), batch_protein, batch_ligand,
                      bond_features=z_bond_emb.to(device), bond_indices=bond_edges.to(device),
                      num_atoms_per_ligand=num_ligand_atoms, t_bond=t_bond.to(device))

            flattened_pos = to_flattened_numpy(v_pos)
            flattened_ele_emb = to_flattened_numpy(v_ele_emb)
            flattened_bond_emb = to_flattened_numpy(v_bond_emb)
            flattened_z = np.concatenate([flattened_pos, flattened_ele_emb, flattened_bond_emb])
            return flattened_z

        # solution = integrate.RK45(ode_func, t0=0.0, y0=flattened_z, t_bound=1.0)
        solution = integrate.solve_ivp(ode_func, t_span=(0.0, 1.0), y0=flattened_z, method='RK45')
        success = solution.success
        logger(f"No. of evaluations: {solution.nfev}, status: {success}")
        solution = solution.y[:, -1]
        z_pos = from_flattened_numpy(solution[:len_pos], pos_shape)
        z_ele_emb = from_flattened_numpy(solution[len_pos:-len_bond_emb], ele_emb_shape)
        z_bond_emb = from_flattened_numpy(solution[-len_bond_emb:], bond_emb_shape)
        dist_betw_X1_feat_and_embedding = torch.cdist(z_ele_emb, model.ligand_element_embedding.weight.data.cpu(),
                                                      p=2)  # * model.emb_scale_factor
        pred_elements = torch.argmin(dist_betw_X1_feat_and_embedding, dim=-1)

        return z_pos, pred_elements, z_ele_emb, z_bond_emb, success


def sample4val_eval(model, val_loader, pos_scale, result_dir, logger, device, ProteinElement2IndexDict,
                    num_timesteps=1000, num_pockets=5, mode='add_aromatic_wo_h'):
    time_consumption = []
    for pocket_id, batch in enumerate(val_loader):
        if pocket_id >= num_pockets:
            break
        protein_pos, protein_ele, protein_amino_acid, protein_is_backbone, ligand_pos, ligand_ele \
            = extract_data(batch, ProteinElement2IndexDict)
        ligand_pos = ligand_pos.to(device)
        protein_batch, ligand_batch = batch.protein_element_batch.to(device), batch.ligand_element_batch.to(device)
        protein_pos, mean_vec = subtract_mean(protein_pos.to(device), batch=protein_batch, verbose=True)
        protein_pos *= pos_scale
        calculator = CalculateBenchmarks(pocket_id=pocket_id,
                                         save_dir=os.path.join(result_dir, 'wo_aromatic_bond'),
                                         aromatic=True, logger=logger)
        if model.add_bond_model:
            bond_edges = model.get_ligand_bond_types_all(num_ligand_atoms=ligand_pos.shape[0],
                                                         ligand_bond_indices=None,
                                                         ligand_bond_types=None, device=device)
            X0_pos, X0_elements_embedding, X0_bond_embedding \
                = model.get_X0(num_ligand_atoms=ligand_pos.shape[0], num_edges=bond_edges.shape[1])
        else:
            bond_edges = None
            X0_pos, X0_elements_embedding, X0_bond_embedding \
                = model.get_X0(num_ligand_atoms=ligand_pos.shape[0])
        sampling_start = time.time()
        X1_pos, X1_ele, pred_bond_types, X1_ele_emb, X1_bond_emb, _ = sample_ode(model, protein_pos, protein_ele,
                                                                                 protein_amino_acid,
                                                                                 protein_is_backbone,
                                                                                 X0_pos, X0_elements_embedding,
                                                                                 num_steps=num_timesteps,
                                                                                 batch_protein=protein_batch,
                                                                                 batch_ligand=ligand_batch,
                                                                                 device=device,
                                                                                 bond_features=X0_bond_embedding,
                                                                                 bond_edges=bond_edges, logger=logger)
        time_consumption.append(time.time() - sampling_start)
        assert pos_scale > 0, f"pos_scale is supposed to be a positive scalar instead of {pos_scale}."
        # --- scale back and then move back to the original gravity center ---
        X1_pos = X1_pos / pos_scale + mean_vec.to(device)

        pred_atom_type = trans.get_atomic_number_from_index(X1_ele, mode=mode)
        xyz = X1_pos.cpu().tolist()
        try:
            r_stable = check_stability(X1_pos.cpu(), pred_atom_type)
            logger.info(f"stability: {r_stable}")
        except Exception as e:
            r_stable = None
            logger.info(f"Failed when calculating stability. {e}")
        return_value = calculator.reconstruct_mol(xyz=xyz, atom_nums=pred_atom_type, aromatic=None, bond_type=None,
                                                  bond_index=None, stable=r_stable)
        if return_value is None:
            continue
        elif return_value == 'unconnected':
            continue

        # logger.info(f"--------- After scaling back and restoring its center,")
        # for ind in range(X1_pos.shape[0]):
        #     logger.info(
        #         f"atom {ind}, predicted index: {X1_ele.cpu()[ind]}, ground truth index: {ligand_ele.cpu()[ind]}")
        #     logger.info(f"predicted position---: {X1_pos.cpu()[ind]}")
        #     logger.info(f"ground truth position: {ligand_pos.cpu()[ind]}")

        num_ligand = len(set(ligand_batch.tolist()))
        for data_idx in range(num_ligand):
            current_data = X1_pos.detach()[ligand_batch == data_idx]
            curr_dist = torch.max(torch.cdist(current_data, current_data, p=2))
            current_gt_data = ligand_pos[ligand_batch == data_idx]
            curr_gt_dist = torch.max(torch.cdist(current_gt_data, current_gt_data, p=2))
            logger.info(f"sample_dist: {curr_dist.item():.2f}, gt_dist: {curr_gt_dist.item():.2f}")

        logger.info(f"sampling time: {time_consumption[-1]:.3f}, {np.mean(time_consumption):.3f} \u00B1 "
                    f"{np.std(time_consumption):.2f} with {len(time_consumption)} samples.")
