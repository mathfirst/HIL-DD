from scipy.spatial.distance import jensenshannon
from sklearn.mixture import GaussianMixture as GM
import torch, os, sys
import numpy as np
import pandas as pd
from utils.evaluate import get_bond_length_distribution_log_likelihood, get_bond_type_distribution_score, \
    get_bond_angle_distribution_log_likelihood, get_bond_angle_type_distribution, \
    get_triple_bond_type_score, get_dihedral_angle_distribution_log_likelihood, get_ring_distribution_score, \
    find_large_rings, find_small_rings, get_average_deviation, calculate_diversity, js_type, js_histogram_distribution, \
    get_ring_statistics, calculate_vina_score, true_bond_length_distribution, true_bond_proba, \
    true_bond_angle_type_proba, true_bond_angle_distribution, true_dihedral_type_proba, \
    true_dihedral_log_likehood_distribution, true_ring_distribution, typical_dihedral_angles, typical_bond_angles


def add_more_info_to_result_dict(pt_dir, true_bond_length_distribution, true_bond_proba,
                                 true_bond_angle_type_proba, true_bond_angle_distribution,
                                 true_dihedral_type_proba, true_dihedral_log_likehood_distribution):
    f_ls = os.listdir(pt_dir)
    f_ls = [f for f in f_ls if '.pt' in f]
    basename = os.path.basename(pt_dir)
    print('basename:', basename)
    pocket_id = int(basename.split('_')[1][6:])
    print('pocket id:', pocket_id)
    for f in f_ls:
        f_path = os.path.join(pt_dir, f)
        ckp = torch.load(f_path)
        mol_ls = ckp['mol_list']
        print(f"processing {f}, {len(mol_ls)} mols")
        flag = False
        for mol_idx, v in enumerate(ckp['vina_score_list']):
            if v == 0 or v == 'NA':
                print('mol_idx', mol_idx, 'does not have vina')
                sdf_filename_path = ckp['sdf_path_list']['mol_idx']
                ckp['vina_score_list'][mol_idx] = calculate_vina_score(mol_ls[mol_idx], pocket_id, sdf_filename_path,
                                                                       ligand_id=mol_idx, temp_dir='./tmp/temp_files', logger=print)
                flag = True
        if 'bond_length_log_likelihood_ls' in ckp and (not flag):
            continue
        bond_length_log_likelihood_ls, bond_lengths = get_bond_length_distribution_log_likelihood(mol_ls,
                                                                                       true_bond_length_distribution)
        ckp['bond_length_log_likelihood_ls'] = bond_length_log_likelihood_ls
        ckp['js_bond_length_distr'] = js_histogram_distribution(true_bond_length_distribution, bond_lengths,
                                                                        first_n=10)
        bond_type_score_ls, bond_freq = get_bond_type_distribution_score(mol_ls, true_bond_proba)
        ckp['bond_type_score_ls'] = bond_type_score_ls
        ckp['js_bond_type_distr'] = js_type(true_bond_proba, bond_freq)
        bond_angle_type_proba, bond_angle_type = get_bond_angle_type_distribution(mol_ls, true_bond_angle_type_proba)
        ckp['bond_angle_type_proba_ls'] = bond_angle_type_proba
        ckp['js_bond_angle_type_distr'] = js_type(true_bond_angle_type_proba, bond_angle_type)
        bond_angle_log_likelihood_ls, bond_angles = get_bond_angle_distribution_log_likelihood(mol_ls, true_bond_angle_distribution, first_n=10)
        ckp['bond_angle_log_likelihood_ls'] = bond_angle_log_likelihood_ls
        ckp['js_bond_angle_distr'] = js_histogram_distribution(true_bond_angle_distribution, bond_angles,
                                                                       first_n=10)
        ckp['dihedral_type_proba_ls'], dihedral_type_statistics = get_triple_bond_type_score(mol_ls, true_dihedral_type_proba)
        ckp['dihedral_log_likelihood_ls'], dihedral_angles = get_dihedral_angle_distribution_log_likelihood(mol_ls,
                                                               true_dihedral_log_likehood_distribution,
                                                                                              first_n=10)
        ckp['js_dihedral_type_distr'] = js_type(true_dihedral_type_proba, dihedral_type_statistics)
        ckp['js_dihedral_distr'] = js_histogram_distribution(true_dihedral_log_likehood_distribution,
                                                                     dihedral_angles,
                                                                     first_n=10)
        ckp['ring_score_ls'] = get_ring_distribution_score(mol_ls, true_ring_distribution)
        ckp['large_ring_ls'] = find_large_rings(mol_ls, size=9)
        ring_statistics, _ = get_ring_statistics(ckp['mol_list'], first_n=9)
        ckp['js_ring_distribution'] = jensenshannon(list(true_ring_distribution.values()),
                                                    list(ring_statistics.values()), base=2)
        ckp['only_small_ring_ls'] = list(np.logical_not(find_large_rings(mol_ls, size=7)))
        ckp['small_ring_ls'] = find_small_rings(mol_ls, size=6, inculde_no_ring=True)
        ckp['dihedral_angle_deviation_mean'], ckp['dihedral_angle_deviation_ls'], ckp['num_dihedral_angle_ls'] = get_average_deviation(mol_ls,
                                                                                                             typical_dihedral_angles,
                                                                                                             key='dihedral_angle_deviation')
        print(f"dihedral_angle_mean: {ckp['dihedral_angle_deviation_mean']:.3f}")
        ckp['bond_angle_deviation_mean'], ckp['bond_angle_deviation_ls'], ckp['num_bond_angle_ls'] = get_average_deviation(mol_ls,
                                                                                                     typical_bond_angles,
                                                                                                     key='bond_angle_deviation')
        print(f"bond_angle_mean: {ckp['bond_angle_deviation_mean']:.3f}")
        torch.save(ckp, f_path)


if __name__ == '__main__':
    pt_dir = sys.argv[1] #"/datapool/data2/home/pengxingang/zhaoyouming/flow_mol_target/tmp/samples_pocket2"
    # pt_dir = r"E:\pytorch-related\TargetDiff2RectifiedFlow\tmp\samples_wd3_wo_unconnected_mols"
    add_more_info_to_result_dict(pt_dir, true_bond_length_distribution, true_bond_proba,
                                 true_bond_angle_type_proba, true_bond_angle_distribution,
                                 true_dihedral_type_proba, true_dihedral_log_likehood_distribution)

    f_ls = os.listdir(pt_dir)
    f_ls = [f for f in f_ls if '.pt' in f]
    metric_of_interest = {'vina_score_list': [], 'qed_list': [], 'sa_list': [], 'logp_list': [], 'lipinski_list': [],
                          'diversity': [], 'connectedness_rate': [], 'contain_benzene_rings': [],
                          'contain_fused_benzene_rings': [],
                          'bond_length_log_likelihood_ls': [], 'bond_type_score_ls': [], 'bond_angle_type_proba_ls': [],
                          'bond_angle_log_likelihood_ls': [], 'dihedral_type_proba_ls': [],
                          'dihedral_log_likelihood_ls': [],
                          'ring_score_ls': [], 'large_ring_ls': [], 'only_small_ring_ls': [], 'small_ring_ls': [],
                          'dihedral_angle_deviation_mean': [], 'dihedral_angle_deviation_ls': [],
                          'bond_angle_deviation_mean': [], 'bond_angle_deviation_ls': [], 'js_bond_length_distr': [],
                          'js_bond_type_distr': [], 'js_bond_angle_type_distr': [], 'js_bond_angle_distr': [],
                          'js_dihedral_type_distr': [], 'js_dihedral_distr': [], 'js_ring_distribution': [],
                          }
    for f in f_ls:
        f_path = os.path.join(pt_dir, f)
        ckp = torch.load(f_path)
        num_samples = len(ckp['mol_list'])
        for k, v in ckp.items():
            if 'vina' in k:
                vina_list_of_interest = [v for v in ckp['vina_score_list'] if v != 0 and v != 'NA']
                metric_of_interest[k] += vina_list_of_interest
                assert len(vina_list_of_interest) == num_samples, 'there may exist ligands that have no Vina score.'
            elif k in metric_of_interest:
                if isinstance(v, list):
                    metric_of_interest[k] += v
                elif isinstance(v, np.ndarray):
                    print(k)
                    metric_of_interest[k] += list(v)
                else:
                    metric_of_interest[k].append(v)

    for k, v in metric_of_interest.items():
        print(f"{k}, {np.mean(v):.4f} \u00B1 {np.std(v):.4f}")
