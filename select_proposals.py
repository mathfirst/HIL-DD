import sys

from sklearn.mixture import GaussianMixture as GM
import torch, os
import numpy as np
import pandas as pd

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def prepare_indices(pairs_path, dst_path, query_ls, upper_th_ls, lower_th_ls,
                    keyword=None, logger=print):
    from copy import deepcopy
    pairs_file_ls = os.listdir(pairs_path)
    if keyword is not None:
        pairs_file_ls = [i for i in pairs_file_ls if keyword in i]
    os.makedirs(dst_path, exist_ok=True)
    if lower_th_ls is None:
        lower_th_ls = deepcopy(upper_th_ls)
    qed_ls, sa_ls, vina_ls, diversity_ls = [], [], [], []
    num_samples = num_connected_benzene_rings = 0
    for file in pairs_file_ls:
        file_path = os.path.join(pairs_path, file)
        logger(f"processing {file_path}")
        ckp = torch.load(file_path, map_location='cpu')
        ckp_copy = {}
        print(f"num of mols containing benzenes: {np.sum(ckp['contain_benzene_rings'])}")
        print(f"num of mols containing fused benzenes: {np.sum(ckp['contain_fused_benzene_rings'])}")
        num_samples += len(ckp['mol_list'])
        indices = np.nonzero(ckp['qed_list'])[0]
        print(f"num of connected mols: {len(indices)}")
        qed_ls += list(np.array(ckp['qed_list'])[indices])
        sa_ls += list(np.array(ckp['sa_list'])[indices])
        mol_ls = ckp['mol_list']
        if 'NA' not in ckp['vina_score_list']:
            indices = np.nonzero(ckp['vina_score_list'])[0]
            vina_ls += list(np.array(ckp['vina_score_list'])[indices])
        else:
            vina_ls += [x for x in ckp['vina_score_list'] if x != 0 and x != 'NA']

        for query, upper_th, lower_th in zip(query_ls, upper_th_ls, lower_th_ls):
            if query == 'fused_benzenes':
                res_positive = np.logical_and(np.array(ckp['contain_fused_benzene_rings']), np.array(ckp['sa_list']))
                res_negative = np.logical_not(np.array(ckp['contain_benzene_rings']))
            elif query == 'benzene_ring':
                res_positive = np.logical_and(np.array(ckp['contain_benzene_rings']), np.array(ckp['sa_list']))
                res_positive = np.logical_and(res_positive, np.array(ckp['bond_length_log_likelihood_ls']) > 0.6)
                num_connected_benzene_rings += len(np.nonzero(res_positive)[0])
                res_negative = np.logical_not(ckp['contain_benzene_rings'])
            elif query == 'bond_length':
                query_array = np.array(ckp['bond_length_log_likelihood_ls'])
                # Since we work on the 10 most common bond types, we also prefer the molecules that contain more common bond types.
                res_positive = np.logical_and(query_array > upper_th, np.array(ckp['bond_type_score_ls']) > 0.2)
                res_negative = np.logical_and(query_array < lower_th, np.array(ckp['bond_type_score_ls']) < 0.18)
                ckp_copy[query + '_array'] = query_array
            elif query == 'bond_angle':
                query_array = np.array(ckp['bond_angle_log_likelihood_ls'])
                # Since we work on the 10 most common bond angle types, we also prefer the molecules that contain more common bond angle types.
                res_positive = np.logical_and(query_array > upper_th, np.array(ckp['bond_angle_type_proba_ls']) > 0.1)
                res_negative = np.logical_and(query_array < lower_th, np.array(ckp['bond_angle_type_proba_ls']) < 0.08)
                ckp_copy[query + '_array'] = query_array
            elif query == 'dihedral_angle':
                query_array = np.array(ckp['dihedral_log_likelihood_ls'])
                res_positive = np.logical_and(query_array > upper_th, np.array(ckp['dihedral_type_proba_ls']) > 0.05)
                res_negative = np.logical_and(query_array < lower_th, np.array(ckp['dihedral_type_proba_ls']) <= 0.05)
                ckp_copy[query + '_array'] = query_array
            elif query == 'dihedral_angle_deviation':
                query_array = np.array(ckp['dihedral_angle_deviation_ls'])
                res_positive = np.logical_and(query_array < upper_th, np.array(ckp['num_dihedral_angle_ls']))
                res_negative = query_array > lower_th  # if n = 0, deviation will be 0 too.
                ckp_copy[query + '_array'] = query_array
            elif query == 'bond_angle_deviation':
                query_array = np.array(ckp['bond_angle_deviation_ls'])
                res_positive = np.logical_and(query_array < upper_th, np.array(ckp['num_bond_angle_ls']))
                res_negative = query_array > lower_th  # if n = 0, deviation will be 0 too.
                ckp_copy[query + '_array'] = query_array
            elif query == 'ring_score':
                query_array = np.array(ckp['ring_score_ls'])
                res_positive = query_array > upper_th
                res_negative = query_array < lower_th
                ckp_copy[query + '_array'] = query_array
            elif query == 'large_ring':
                query_array = np.array(ckp['large_ring_ls'])
                res_positive = ckp['only_small_ring_ls'] #np.logical_not(query_array)
                res_negative = np.logical_and(query_array, np.logical_not(ckp['small_ring_ls']))
                ckp_copy[query + '_array'] = query_array
            elif query == 'mixed':
                qed_array = np.array(ckp['qed_list'])
                sa_array = np.array(ckp['sa_list'])
                vina_array = np.array([v if v != 'NA' else 0 for v in ckp['vina_score_list']])
                bond_len_array = np.array(ckp['bond_length_log_likelihood_ls'])
                query_array = (-0.1) * vina_array + qed_array + sa_array + sigmoid(0.5*bond_len_array)
                res_positive = (qed_array >= 0.65) & (sa_array >= 0.65) & (vina_array < -9)
                res_negative = (qed_array < 0.5) & (sa_array < 0.6) & (vina_array > -8)
                ckp_copy[query + '_array'] = query_array
            else:
                if query != 'vina_score':
                    query_array = np.array(ckp[query + '_list'])
                elif query == 'vina_score':
                    query_array = np.array([v if v != 'NA' else 0 for v in ckp[query + '_list']])
                else:
                    raise NotImplementedError(f'unrecoginzed query {query}')
                ckp_copy[query + '_array'] = query_array
                res_positive = query_array > upper_th if query != 'vina_score' else query_array < upper_th
                res_negative = np.logical_and(query_array < lower_th, query_array > 0) if query != 'vina_score' \
                    else np.logical_and(query_array > lower_th, query_array < 0)
            positive_indices = np.nonzero(res_positive)[0]
            print(f"{query}, num of positive samples {len(positive_indices)}")
            negative_indices = np.nonzero(res_negative)[0]
            print(f"{query}, num of negative samples {len(negative_indices)}")
            ckp_copy[str(query) + '_positive_indices'] = torch.from_numpy(positive_indices)
            ckp_copy[str(query) + '_negative_indices'] = torch.from_numpy(negative_indices)
        ckp_copy['X0_pos_array'] = torch.stack(ckp['X0_pos_list'], dim=0) #.astype(np.float32)
        ckp_copy['X1_pos_array'] = torch.stack(ckp['X1_pos_list'], dim=0) #.astype(np.float32)
        ckp_copy['X0_ele_emb_array'] = torch.stack(ckp['X0_ele_emb_list'], dim=0) #.astype(np.float32)
        ckp_copy['X1_ele_emb_array'] = torch.stack(ckp['X1_ele_emb_list'], dim=0) #.astype(np.float32)
        ckp_copy['X0_bond_emb_array'] = torch.stack(ckp['X0_bond_emb_list'], dim=0) #.astype(np.float32)
        ckp_copy['X1_bond_emb_array'] = torch.stack(ckp['X1_bond_emb_list'], dim=0) #.astype(np.float32)
        ckp_copy['mol_list'] = ckp['mol_list']

        torch.save(ckp_copy, os.path.join(dst_path, file))
    num_connected_samples = len(sa_ls)
    print(f"num_samples: {num_samples}, num_connected_samples: {num_connected_samples}, "
          f"connectedness rate: {num_connected_samples/num_samples:.4f}")
    print(f"num_samples: {num_samples}, num_connected_samples: {num_connected_samples}, "
          f"benzene ring (with favorable bond lengths) rate: {num_connected_benzene_rings/num_connected_samples:.4f}")
    print(f"qed: {np.mean(qed_ls):.3f} \u00B1 {np.std(qed_ls):.2f}")
    print(f"sa: {np.mean(sa_ls):.3f} \u00B1 {np.std(sa_ls):.2f}")
    print(f"vina: {np.mean(vina_ls):.3f} \u00B1 {np.std(vina_ls):.2f}")


if __name__ == '__main__':
    src, dst = sys.argv[1], sys.argv[2]
    prepare_indices(pairs_path=src,  # './tmp/samples_pocket2'
                    dst_path=dst,  # './tmp/samples_pocket2_proposals',
                    query_ls=('qed', 'sa', 'vina_score', 'benzene_ring', 'bond_length', 'bond_angle',
                              'large_ring', 'dihedral_angle_deviation', 'mixed'),
                    upper_th_ls=(0.7, 0.7, -9, None, 0.5, -4.5, 9, 5, None),  # 0.94
                    lower_th_ls=(0.5, 0.61, -7, None, 0, -5, 6, 10, None),  # 0.9
                    keyword='.pt')