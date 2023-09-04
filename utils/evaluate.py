# This code is used to calculate metrics for mols which are either from datasets or generative models.
# Author: Youming Zhao, Date: 2022-12-28-10:53

from datetime import datetime
import os, re, torch, copy, csv
from copy import deepcopy
import numpy as np
import uuid  # https://newbedev.com/best-way-to-generate-random-file-names-in-python
from rdkit import Chem, DataStructs
from rdkit.Chem import QED, Crippen, Descriptors, Lipinski, rdMolDescriptors, rdMolTransforms, Draw, AllChem
from utils.sascorer import compute_sa_score
from scipy.spatial.distance import jensenshannon
from utils import reconstruct
import pandas as pd

MAP_QUERY2RES_DICT_KEY = {'qed': 'qed_list', 'sa': 'sa_list', 'vina_score': 'vina_score_list',
                          'benzene_ring': 'contain_benzene_rings', 'diversity': 'diversity',
                          'fused_benzenes': 'contain_fused_benzene_rings',
                          'connectedness_rate': 'connectedness_rate',
                          'bond_type_score': 'bond_type_score_ls',
                          'bond_length': 'bond_length_log_likelihood_ls',
                          'bond_angle': 'bond_angle_log_likelihood_ls',
                          'bond_angle_type_score': 'bond_angle_type_score_ls',
                          'bond_angle_deviation': 'avg_bond_angle',
                          'dihedral_type_score': 'dihedral_type_proba_ls',
                          'dihedral_angle': 'dihedral_log_likelihood_ls',
                          'dihedral_angle_deviation': 'avg_dihedral_angle',
                          'ring_score': 'ring_score_ls', 'large_ring': 'large_ring_ls',
                          'js_bond_type': 'js_bond_type_distr',
                          'js_bond_length': 'js_bond_length_distr',
                          'js_bond_angle': 'js_bond_angle_distr',
                          'js_bond_angle_type': 'js_bond_angle_type_distr',
                          'js_dihedral_type': 'js_dihedral_type_distr',
                          'js_dihedral_angle': 'js_dihedral_distr',
                          'js_ring_dist': 'js_ring_distribution',
                          }

true_bond_proba = torch.load('configs/statistics/bond_appearance_prob.pt')
true_bond_angle_type_proba = torch.load('configs/statistics/bond_angle_type_proba.pt')
true_bond_length_distribution = torch.load('configs/statistics/bond_length_distribution_dict_bin_width_0.02.pt')
true_bond_angle_distribution = torch.load('configs/statistics/bond_angle_model_dict_bin_width_2.pt')
true_dihedral_type_proba = torch.load('configs/statistics/true_triple_bond_type_distribution.pt')
true_dihedral_log_likehood_distribution = torch.load('configs/statistics/dihedral_angle_model_dict.pt')
true_ring_distribution = torch.load('configs/statistics/ring_distribution.pt')['Dataset'].to_dict()
typical_dihedral_angles = torch.load('configs/statistics/typical_dihedral_angles.pt')
typical_bond_angles = torch.load('configs/statistics/typical_bond_angles.pt')
PDB_dict = {}  # keys: 4-character ID, values: numbers
with open('configs/PDB_ID_CrossDocked_testset.csv') as f:
    lines = csv.reader(f)
    for line in lines:
        if line[0].isdigit():
            PDB_dict[line[1]] = line[0]

class CalculateBenchmarks(object):
    def __init__(self, pocket_id, save_dir='./tmp', aromatic=False, logger=None,
                 protein_pdbqt_dir=r"E:\pytorch-related\TargetDiff2RectifiedFlow\test_protein",
                 cal_vina_score=True, cal_diversity=False):
        self.pocket_id = pocket_id
        self.cal_vina_score = cal_vina_score
        self.cal_diversity = cal_diversity
        self.stat_dict = {'diversity': 0.0, 'connectedness_rate': 0.0, 'num_trials': 0, }
        self.result_dict = {'pocket_id': pocket_id, 'mol_list': [], 'smiles_list': [], 'atom_nums_list': [],
                            'xyz_list': [], 'qed_list': [], 'sa_list': [], 'logp_list': [], 'lipinski_list': [],
                            'stable_list': [], 'diversity': 0.0, 'vina_score_list': [], 'time_list': [],
                            'X0_pos_list': [], 'X0_ele_emb_list': [], 'X0_bond_emb_list': [],
                            'X1_pos_list': [], 'X1_ele_emb_list': [], 'X1_bond_emb_list': [],
                            'bond_edges_list': [], 'connectedness_rate': 0.0, 'num_trials': 0,
                            'is_high_affinity_list': [], 'png_path_list': [], 'num_atoms_list': [],
                            'sdf_path_list': [], 'contain_benzene_rings': [], 'contain_fused_benzene_rings': [],
                            'num_bonds_list': [], 'num_rings_list': [], 'num_benzene_rings_list': []}
        self.unconnected_mol_dict = {'X0_pos_list': [], 'X0_ele_emb_list': [], 'X0_bond_emb_list': [],
                                     'X1_pos_list': [], 'X1_ele_emb_list': [], 'X1_bond_emb_list': [],
                                     'bond_edges_list': []}
        if aromatic:
            self.result_dict['is_aromatic_list'] = []
        # self.mol_metrics = MoleculeProperties()
        self.num_ligands = 0
        self.num_valid_samples = 0
        self.save_dir = save_dir
        self.ligand_name = '_ligand_%d_of_pocket_%d' % (self.num_ligands, self.pocket_id)
        self.sdf_dir = os.path.join(save_dir, 'SDF')
        self.pt_dir = os.path.join(save_dir, 'pt_files')
        self.ligand_pdbqt_out_dir = os.path.join(save_dir, 'ligand_pdbqt_out_dir')
        self.ligand_pdbqt_dir = os.path.join(save_dir, 'ligand_pdbqt_dir')
        self.protein_pdbqt_dir = protein_pdbqt_dir
        self.test_set_benchmarks = torch.load('configs/statistics/test_set_benchmarks.pt')
        os.makedirs(self.sdf_dir, exist_ok=True)
        os.makedirs(self.pt_dir, exist_ok=True)
        os.makedirs(self.ligand_pdbqt_out_dir, exist_ok=True)
        os.makedirs(self.ligand_pdbqt_dir, exist_ok=True)
        self.pt_filename = os.path.join(self.pt_dir,
                                        datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '_pocket_%d.pt' % pocket_id)
        if logger is None:
            self.logger = print
        else:
            self.logger = logger.info

    def reconstruct_mol(self, xyz, atom_nums, aromatic=None, bond_type=None, bond_index=None, stable=None,
                        drop_unconnected_mol=True, protein_pdbqt_dir='configs/test_protein', **kwargs):
        try:
            if aromatic is not None:
                basic_mode = False
            else:
                basic_mode = True
            self.result_dict['num_trials'] += 1
            try:
                mol = reconstruct.reconstruct_from_generated(xyz, atom_nums, aromatic=aromatic,
                                                             basic_mode=basic_mode,
                                                             bond_type=None,
                                                             bond_index=None)
            except Exception as err:
                self.logger(f"error occurred when reconstructing: {err}")
                return None
            self.num_valid_samples += 1
            try:
                smiles = Chem.MolToSmiles(mol)
            except Exception as err:
                self.logger(f"error occurred when converting mol to Smiles: {err}")
                return None
            if '.' in smiles:  # We dump the unconnected molecules.
                self.logger(f"pocket {self.result_dict['pocket_id']}, sample {self.num_ligands}, "
                            f"num_trials {self.result_dict['num_trials']}. mol is not connected.")
                # to check what the unconnected mols look like
                # sdf_filename_path = os.path.join(self.sdf_path, datetime.now().strftime("%Y-%m-%d-%H-%M-%S") +
                #                                  '_uncon_ligand_%d_of_pocket_%d.sdf' % (
                #                                  self.num_ligands, self.pocket_id))
                # Chem.MolToMolFile(mol, sdf_filename_path)  # save the mol into a sdf file
                if drop_unconnected_mol:
                    return 'unconnected'
                else:
                    qed = sa = logp = lipinski = vina_score = 0
                    sdf_filename_path = png_path = ''
            else:
                self.logger(f"Connected mol. Working on calculating metrics of "
                            f"sample {self.num_ligands} of pocket {self.result_dict['pocket_id']}, {self.save_dir}...")
                result_dict = evaluate_metrics(mol, logger=self.logger)
                qed, sa, logp, lipinski = result_dict['qed'], result_dict['sa'], result_dict['logp'], result_dict[
                    'lipinski']
                self.ligand_name = '_ligand_%d_of_pocket_%d' % (self.num_ligands, self.pocket_id)
                current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                sdf_filename_path = os.path.join(self.sdf_dir, current_time + self.ligand_name + '.sdf')
                Chem.MolToMolFile(mol, sdf_filename_path)  # save the mol as an SDF file
                png_path = os.path.join(self.sdf_dir, current_time + self.ligand_name + '.png')
                mol_copy = deepcopy(mol)
                AllChem.Compute2DCoords(mol_copy)
                Chem.Draw.MolToFile(mol_copy, png_path, size=(300, 300), kekulize=True)
                if self.cal_vina_score:
                    try:
                        self.logger(f"calculating vina score for {sdf_filename_path}")
                        vina_score = calculate_vina_score(mol, pocket_id=self.pocket_id, ligand_id=self.num_ligands,
                                                          sdf_filename_path=sdf_filename_path, logger=self.logger,
                                                          protein_pdbqt_dir=protein_pdbqt_dir,
                                                          protein_pdbqt_file_path=kwargs['protein_pdbqt_file_path'])
                        # self.logger(f"The vina score is {vina_score:.4f}")
                    except Exception as err:
                        self.logger(f"{err} during calculating vina score. ")
                        vina_score = 'NA'
                        return None
                else:
                    vina_score = 0
            self.result_dict['vina_score_list'].append(vina_score)
            self.result_dict['is_high_affinity_list'].append(
                self.test_set_benchmarks['vina_score_list'][
                    self.pocket_id] >= vina_score if vina_score != 'NA' else 'NA')
            self.result_dict['mol_list'].append(mol)
            self.result_dict['smiles_list'].append(smiles)
            self.result_dict['atom_nums_list'].append(atom_nums)
            self.result_dict['qed_list'].append(qed)
            self.result_dict['sa_list'].append(sa)
            self.result_dict['logp_list'].append(logp)
            self.result_dict['lipinski_list'].append(lipinski)
            self.result_dict['stable_list'].append(stable)
            if aromatic is not None:
                self.result_dict['is_aromatic_list'].append(aromatic)
            self.num_ligands = len(self.result_dict['mol_list'])
            self.result_dict['sdf_path_list'].append(sdf_filename_path)
            self.result_dict['png_path_list'].append(png_path)
            self.result_dict['num_atoms_list'].append(mol.GetNumHeavyAtoms())
            self.result_dict['num_bonds_list'].append(len(mol.GetBonds()))
            self.result_dict['num_rings_list'].append(len(Chem.GetSymmSSSR(mol)))
            self.result_dict['num_benzene_rings_list'].append(Chem.Fragments.fr_benzene(mol))
            if '.' not in smiles:
                sdf_filename_res_path = sdf_filename_path[:-4] + "_results.txt"
                self.write_results_to_txt(sdf_filename_res_path)
            self.result_dict['contain_benzene_rings'].append(find_benzene_rings(smiles))
            self.result_dict['contain_fused_benzene_rings'].append(find_fused_benzene_rings(smiles))
            if self.result_dict['contain_fused_benzene_rings'][-1]:
                self.logger(f"The mol containing fused benzene rings was saved at {sdf_filename_path}.")
            # torch.save(self.result_dict, self.pt_filename)
            return self.result_dict
        except Exception as e:
            self.logger(f"Failed when constructing mol and calculating metrics. {e}")
            return None

    def write_results_to_txt(self, filename):
        with open(filename, 'a') as fn:
            lines = [f"Pocket: {self.pocket_id}, {self.ligand_name}\n",
                     f"QED: {self.result_dict['qed_list'][-1]:.3f}, "
                     f"ref: {self.test_set_benchmarks['qed_list'][self.pocket_id]:.3f}\n",
                     f"SA: {self.result_dict['sa_list'][-1]:.3f}, "
                     f"ref: {self.test_set_benchmarks['sa_list'][self.pocket_id]:.3f}\n",
                     f"Vina score: {self.result_dict['vina_score_list'][-1]:.3f}, "
                     f"ref: {self.test_set_benchmarks['vina_score_list'][self.pocket_id]:.3f}\n",
                     f"High affinity: {self.result_dict['is_high_affinity_list'][-1]}\n",
                     f"Lipinski: {self.result_dict['lipinski_list'][-1]:.3f}, "
                     f"ref: {self.test_set_benchmarks['lipinski_list'][self.pocket_id]:.3f}\n",
                     f"LogP: {self.result_dict['logp_list'][-1]:.3f}, "
                     f"ref: {self.test_set_benchmarks['logp_list'][self.pocket_id]:.3f}\n"]
            fn.writelines(lines)

    def save_unconnected_mol(self, X0_pos, X0_ele_emb, X1_pos, X1_ele_emb, X0_bond_emb=None, X1_bond_emb=None,
                             bond_edges=None, save_pair=False):
        self.unconnected_mol_dict['X0_pos_list'].append(X0_pos.cpu())
        self.unconnected_mol_dict['X0_ele_emb_list'].append(X0_ele_emb.cpu())
        self.unconnected_mol_dict['X1_pos_list'].append(X1_pos.cpu())
        self.unconnected_mol_dict['X1_ele_emb_list'].append(X1_ele_emb.cpu())
        if X0_bond_emb is not None and X1_bond_emb is not None:
            self.unconnected_mol_dict['X0_bond_emb_list'].append(X0_bond_emb.cpu())
            self.unconnected_mol_dict['X1_bond_emb_list'].append(X1_bond_emb.cpu())
            self.unconnected_mol_dict['bond_edges_list'].append(bond_edges)
        if save_pair:
            torch.save(self.unconnected_mol_dict, os.path.join(self.save_dir, datetime.now().strftime(
                "%Y-%m-%d-%H-%M-%S") + '_unconnected_mols_pocket_%d' % self.pocket_id))

    def save_X0(self, X0_pos, X0_ele_emb, X1_pos, X1_ele_emb, X0_bond_emb=None, X1_bond_emb=None, filename=None,
                skip_save=False):
        self.result_dict['X0_pos_list'].append(X0_pos.cpu())
        self.result_dict['X0_ele_emb_list'].append(X0_ele_emb.cpu())
        self.result_dict['X1_pos_list'].append(X1_pos.cpu())
        self.result_dict['X1_ele_emb_list'].append(X1_ele_emb.cpu())
        if X0_bond_emb is not None:
            self.result_dict['X0_bond_emb_list'].append(X0_bond_emb.cpu())
            self.result_dict['X1_bond_emb_list'].append(X1_bond_emb.cpu())
            # self.result_dict['bond_edges_list'].append(bond_edges)
        if skip_save:  # make sure you will save the result somewhere else
            return
        if filename is None:
            filename = self.pt_filename
        else:
            filename = os.path.join(self.save_dir, datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + filename)
        torch.save(self.result_dict, filename)

    def calculate_stat(self):
        vina_ls = [x for x in self.result_dict['vina_score_list'] if x != 'NA' and x != 0]
        qed_ls = self.result_dict['qed_list']
        sa_ls = self.result_dict['sa_list']
        lipinski_ls = self.result_dict['lipinski_list']
        logp_ls = self.result_dict['logp_list']
        nonzero_indices = np.nonzero(np.array(qed_ls))[0]
        qed_ls = np.array(qed_ls)[nonzero_indices]
        sa_ls = np.array(sa_ls)[nonzero_indices]
        lipinski_ls = np.array(lipinski_ls)[nonzero_indices]
        logp_ls = np.array(logp_ls)[nonzero_indices]
        connected_mols = [mol for x, mol in zip(self.result_dict['smiles_list'], self.result_dict['mol_list']) if
                          '.' not in x]
        is_high_affinity_ls = [x for x in self.result_dict['is_high_affinity_list'] if x != 'NA']
        self.stat_dict['vina_score_mean'] = np.mean(vina_ls) if len(vina_ls) else 0
        self.stat_dict['vina_score_std'] = np.std(vina_ls) if len(vina_ls) else 0
        self.stat_dict['qed_mean'] = np.mean(qed_ls)
        self.stat_dict['qed_std'] = np.std(qed_ls)
        self.stat_dict['sa_mean'] = np.mean(sa_ls)
        self.stat_dict['sa_std'] = np.std(sa_ls)
        self.stat_dict['logp_mean'] = np.mean(logp_ls)
        self.stat_dict['logp_std'] = np.std(logp_ls)
        self.stat_dict['lipinski_mean'] = np.mean(lipinski_ls)
        self.stat_dict['lipinski_std'] = np.std(lipinski_ls)
        self.stat_dict['time_mean'] = np.mean(self.result_dict['time_list'])
        self.stat_dict['time_std'] = np.std(self.result_dict['time_list'])
        self.stat_dict['diversity'] = self.result_dict['diversity'] = calculate_diversity(connected_mols)
        self.stat_dict['validity_rate'] = self.result_dict['validity_rate'] = self.num_valid_samples / self.result_dict[
            'num_trials']
        self.stat_dict['connectedness_rate'] = self.result_dict['connectedness_rate'] = len(
            connected_mols) / self.num_valid_samples
        self.stat_dict['high_affinity'] = self.result_dict['high_affinity'] \
            = np.sum(is_high_affinity_ls) / len(is_high_affinity_ls)
        return self.stat_dict

    def print_mean_std(self, desc=''):
        self.calculate_stat()
        self.logger(f"{desc}. The current stat of pocket {self.pocket_id} with {self.num_ligands} ligands are:\n"
                    f"Vina score: {self.stat_dict['vina_score_mean']:.3f} \u00B1 {self.stat_dict['vina_score_std']:.2f}\n"
                    f"High affinity: {self.stat_dict['high_affinity']:.4f}\n"
                    f"QED: {self.stat_dict['qed_mean']:.3f} \u00B1 {self.stat_dict['qed_std']:.2f}\n"
                    f"SA: {self.stat_dict['sa_mean']:.3f} \u00B1 {self.stat_dict['sa_std']:.2f}\n"
                    f"LogP: {self.stat_dict['logp_mean']:.3f} \u00B1 {self.stat_dict['logp_std']:.2f}\n"
                    f"Lipinski: {self.stat_dict['lipinski_mean']:.3f} \u00B1 {self.stat_dict['lipinski_std']:.2f}\n"
                    f"Diversity: {self.stat_dict['diversity']:.4f}\n"
                    f"Connected rate: {self.stat_dict['connectedness_rate']:.4f}\n"
                    f"Validity rate: {self.stat_dict['validity_rate']:.4f}\n"
                    f"Time: {self.stat_dict['time_mean']:.3f} \u00B1 {self.stat_dict['time_std']:.2f}\n"
                    )


def print_flattened_result(list_of_dicts_of_lists, logger, desc="list of dicts"):
    logger.info(f"calculating the results of {desc}")
    flattened_vina_score = [x for a_dict in list_of_dicts_of_lists for x in a_dict['vina_score_list']]
    flattened_qed = [x for a_dict in list_of_dicts_of_lists for x in a_dict['qed_list']]
    flattened_sa = [x for a_dict in list_of_dicts_of_lists for x in a_dict['sa_list']]
    flattened_logp = [x for a_dict in list_of_dicts_of_lists for x in a_dict['logp_list']]
    flattened_lipinski = [x for a_dict in list_of_dicts_of_lists for x in a_dict['lipinski_list']]
    flattened_time = [x for a_dict in list_of_dicts_of_lists for x in a_dict['time_list']]
    flattened_high_affinity = [a_dict['high_affinity'] for a_dict in list_of_dicts_of_lists]
    flattened_diversity = [a_dict['diversity'] for a_dict in list_of_dicts_of_lists]
    flattened_success_rate = [a_dict['connectedness_rate'] for a_dict in list_of_dicts_of_lists]
    logger.info(f"Vina score: {np.mean(flattened_vina_score):.3f} \u00B1 {np.std(flattened_vina_score):.2f}")
    logger.info(f"High affinity: {np.mean(flattened_high_affinity):.3f} \u00B1 {np.std(flattened_high_affinity):.2f}")
    logger.info(f"QED: {np.mean(flattened_qed):.3f} \u00B1 {np.std(flattened_qed):.2f}")
    logger.info(f"SA: {np.mean(flattened_sa):.3f} \u00B1 {np.std(flattened_sa):.2f}")
    logger.info(f"LogP: {np.mean(flattened_logp):.3f} \u00B1 {np.std(flattened_logp):.2f}")
    logger.info(f"Lipinski: {np.mean(flattened_lipinski):.3f} \u00B1 {np.std(flattened_lipinski):.2f}")
    logger.info(f"Diversity: {np.mean(flattened_diversity):.3f} \u00B1 {np.std(flattened_diversity):.2f}")
    logger.info(
        f"connectedness rate: {np.mean(flattened_success_rate):.3f} \u00B1 {np.std(flattened_success_rate):.2f}")
    logger.info(f"time: {np.mean(flattened_time):.5f} \u00B1 {np.std(flattened_time):.4f}")


def summarize_res_one_round(res_dict, query_base_dict, improvement_one_round, logger=print):
    for k, v in improvement_one_round.items():
        res_k = MAP_QUERY2RES_DICT_KEY[k]
        result = res_dict[res_k]
        if k == 'vina_score':
            result = [x for x in result if x != 0 and x != 'NA']
            if len(result) == 0:
                result = [query_base_dict[k]]
        if 'js' in k or 'deviation' in k:
            v.append((query_base_dict[k] - result) / query_base_dict[k])
        else:
            metric_mean, metric_std = np.mean(result), np.std(result)
            query_base = query_base_dict[k]
            improvement = (metric_mean - query_base) / query_base
            if 'log_likelihood' in k or 'large_ring' in k or k == 'dihedral_angle' or k == 'bond_length' or k == 'bond_angle':
                improvement = -improvement
            v.append(improvement)
            logger(f"{k}: {metric_mean:.3f} \u00B1 {metric_std:.2f}, improvement: {improvement:.4f}")
        improvement_one_round_display = [f'{100 * i:.2f}%' for i in v]
        logger(f"{k}_improvement_ls: {improvement_one_round_display}")


def obey_lipinski(mol):
    mol = deepcopy(mol)
    Chem.SanitizeMol(mol)
    rule_1 = Descriptors.ExactMolWt(mol) < 500
    rule_2 = Lipinski.NumHDonors(mol) <= 5
    rule_3 = Lipinski.NumHAcceptors(mol) <= 10
    rule_4 = (logp := Crippen.MolLogP(mol) >= -2) & (logp <= 5)
    rule_5 = rdMolDescriptors.CalcNumRotatableBonds(mol) <= 10
    return np.sum([int(a) for a in [rule_1, rule_2, rule_3, rule_4, rule_5]])


def calculate_vina_score(rdmol, pocket_id, sdf_filename_path, protein_pdbqt_dir='configs/test_protein',
                         ligand_id=0, temp_dir='./tmp/temp_files', logger=print, **kwargs):
    os.makedirs(temp_dir, exist_ok=True)
    from sys import platform
    # if platform != 'win32':
    # protein_pdbqt_dir = 'configs/test_protein'
    # 1. convert an sdf file into a pdbqt file
    vina_score_id = 'ligand_%d_of_pocket_%d' % (ligand_id, pocket_id)
    random_id = str(uuid.uuid4())
    isExist = os.path.exists(sdf_filename_path)
    if not isExist:
        sdf_filename_path = os.path.join(temp_dir, random_id + '.sdf')
        logger(f"creating sdf file {sdf_filename_path}")
        Chem.MolToMolFile(rdmol, sdf_filename_path)  # save the mol as a sdf file
    ligand_pdbqt_filename_path = os.path.join(os.path.dirname(sdf_filename_path), vina_score_id + '.pdbqt')
    command = f'obabel -isdf {sdf_filename_path} -opdbqt -O {ligand_pdbqt_filename_path}'
    if platform == 'win32':  # detecting what os the executing file is running on
        os.system('cmd /c ' + command)  # execute the command and terminate
    else:
        os.system(command)
    # self.logger(f"sdf has been converted into pdbqt {ligand_pdbqt_filename_path}.")
    # 2. get the center of the rdmol
    pos = rdmol.GetConformer(0).GetPositions()
    center = (pos.max(0) + pos.min(0)) / 2
    result_file_path = os.path.join(temp_dir, vina_score_id + '_out.pdbqt')
    if 'protein_pdbqt_file_path' in kwargs and kwargs['protein_pdbqt_file_path']:
        protein_pdbqt_file_path = kwargs['protein_pdbqt_file_path']
    else:
        protein_pdbqt_file_path = os.path.join(protein_pdbqt_dir, 'protein_%d.pdbqt' % pocket_id)
    # 3. use qvina2 to calculate vina score and print the output into a random *.txt
    fn_txt = os.path.join(temp_dir, ''.join([random_id, '.txt']))  #
    command = "qvina2 --receptor {receptor_id} \
                    --ligand {ligand_id} \
                    --center_x {center_x} \
                    --center_y {center_y} \
                    --center_z {center_z} \
                    --size_x 20 --size_y 20 --size_z 20 \
                    --exhaustiveness {exhaust} --out {output} > {txt_file} ".format(  # --out {output}
        receptor_id=protein_pdbqt_file_path,
        ligand_id=ligand_pdbqt_filename_path,
        exhaust=64,
        center_x=center[0],
        center_y=center[1],
        center_z=center[2],
        output=result_file_path,
        txt_file=fn_txt
    )
    if platform == 'win32':
        os.system('cmd /c ' + command)  # execute the command and terminate
    else:
        os.system(command)
    # 4. parse the result.txt file to get a vina score
    with open(fn_txt, 'r') as file:
        lines = file.readlines()
        for line in lines:
            split_result = line.split(sep=' ')
            # To understand the code below, you have to take a brief look at result.txt.
            if len(split_result) > 3 and split_result[3] == '1':
                split_result = list(filter(lambda x: x != '', split_result))
                nums = [float(s) for s in re.findall(r'-?\d+\.?\d*', split_result[1])]
                vina_score = nums[0]
                logger(f"{vina_score_id} vina score: {vina_score}")
                break
    dst_file = os.path.join(temp_dir, fn_txt[-5:])
    try:
        os.rename(fn_txt, dst_file)
    except:
        os.replace(fn_txt, dst_file)

    return vina_score


def evaluate_metrics(mol, logger=print, verbose=True):
    '''
    input: rdmol
    output: a dict of the corresponding result
    '''
    results = {
        'qed': QED.qed(mol),
        'sa': compute_sa_score(mol),
        'logp': Crippen.MolLogP(mol),
        'lipinski': obey_lipinski(mol),
    }
    if verbose:
        logger(f"qed: {results['qed']:.3f}")
        logger(f"sa: {results['sa']:.3f}")
        logger(f"lipinski: {results['lipinski']:.3f}")
        logger(f"logp: {results['logp']:.3f}")

    return results


def similarity(mol_a, mol_b):
    fp1 = Chem.RDKFingerprint(mol_a)
    fp2 = Chem.RDKFingerprint(mol_b)
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def calculate_diversity(pocket_mols):
    if len(pocket_mols) < 2:
        return 0.0

    div = 0
    total = 0
    for i in range(len(pocket_mols)):
        for j in range(i + 1, len(pocket_mols)):
            div += 1 - similarity(pocket_mols[i], pocket_mols[j])
            total += 1
    return div / total


def find_fused_benzene_rings(smiles):
    fused_benzene_rings = Chem.MolFromSmarts('c12ccccc1cccc2')
    try:
        mol = Chem.MolFromSmiles(smiles)
        state = mol.HasSubstructMatch(fused_benzene_rings)
        return state
    except Exception as err:
        print(err)
        return False


def find_benzene_rings(smiles):
    benzene_rings = Chem.MolFromSmarts('c1ccccc1')
    try:
        mol = Chem.MolFromSmiles(smiles)
        state = mol.HasSubstructMatch(benzene_rings)
        return state
    except Exception as err:
        print(err)
        return False


def find_substructure(smiles, smarts):
    '''
    Args:
        smiles:
        smarts: 1. benzene ring: 'c1ccccc1';
                2. fused benzene ring: 'c12ccccc1cccc2
                3. benzene ring fused with a 6-member ring containing 2 nitrogens: 'c12ccccc1cncn2',
                   e.g. Gefitinib, Erlotinib
                4. 'c1ccccc1Nc2ncccn2', e.g. Osimertinib, Olmutinib (3rd generation)
    Returns: a boolean
    '''
    pattern = Chem.MolFromSmarts(smarts)
    try:
        mol = Chem.MolFromSmiles(smiles)
        state = mol.HasSubstructMatch(pattern)
        return state
    except Exception as err:
        print(err)
        return False





def find_large_ring(mol, size=7):
    '''
    The input is rdkit mol and the output is a boolean variable.
    '''
    ssr = Chem.GetSymmSSSR(mol)
    for r in ssr:
        if len(r) >= size:
            return True

    return False


def find_large_rings(mol_ls, size=7):
    res_ls = []
    for mol in mol_ls:
        res_ls.append(find_large_ring(mol, size=size))

    return res_ls


def find_small_ring(mol, size=6, inculde_no_ring=True):
    ssr = Chem.GetSymmSSSR(mol)
    if inculde_no_ring and len(ssr) == 0:
        return True
    for r in ssr:
        if len(r) <= size:
            return True

    return False


def find_small_rings(mol_ls, size=6, inculde_no_ring=True):
    res_ls = []
    for mol in mol_ls:
        res_ls.append(find_small_ring(mol, size=size, inculde_no_ring=inculde_no_ring))

    return res_ls


def add_more_ls_to_result_dict(result_dict):
    bond_type_score_ls, bond_freq = get_bond_type_distribution_score(result_dict['mol_list'], true_bond_proba)
    result_dict['bond_type_score_ls'] = bond_type_score_ls
    result_dict['js_bond_type_distr'] = js_type(true_bond_proba, bond_freq)

    bond_angle_type_score_ls, bond_angle_type = get_bond_angle_type_distribution(result_dict['mol_list'],
                                                                                 true_bond_angle_type_proba)
    result_dict['bond_angle_type_score_ls'] = bond_angle_type_score_ls
    result_dict['js_bond_angle_type_distr'] = js_type(true_bond_angle_type_proba, bond_angle_type)

    result_dict['dihedral_type_proba_ls'], dihedral_type_statistics = get_triple_bond_type_score(
        result_dict['mol_list'],
        true_dihedral_type_proba)
    result_dict['js_dihedral_type_distr'] = js_type(true_dihedral_type_proba, dihedral_type_statistics)

    bond_length_log_likelihood_ls, bond_lengths = get_bond_length_distribution_log_likelihood(result_dict['mol_list'],
                                                                                              true_bond_length_distribution,
                                                                                              first_n=10)
    result_dict['bond_length_log_likelihood_ls'] = bond_length_log_likelihood_ls
    result_dict['js_bond_length_distr'] = js_histogram_distribution(true_bond_length_distribution, bond_lengths,
                                                                    first_n=10)
    bond_angle_log_likelihood_ls, bond_angles = get_bond_angle_distribution_log_likelihood(result_dict['mol_list'],
                                                                                           true_bond_angle_distribution,
                                                                                           first_n=10)
    result_dict['bond_angle_log_likelihood_ls'] = bond_angle_log_likelihood_ls
    result_dict['js_bond_angle_distr'] = js_histogram_distribution(true_bond_angle_distribution, bond_angles,
                                                                   first_n=10)
    result_dict['dihedral_log_likelihood_ls'], dihedral_angles = get_dihedral_angle_distribution_log_likelihood(
        result_dict['mol_list'],
        true_dihedral_log_likehood_distribution,
        first_n=10)
    result_dict['js_dihedral_distr'] = js_histogram_distribution(true_dihedral_log_likehood_distribution,
                                                                 dihedral_angles,
                                                                 first_n=10)
    result_dict['ring_score_ls'] = get_ring_distribution_score(result_dict['mol_list'], true_ring_distribution)
    ring_statistics, _ = get_ring_statistics(result_dict['mol_list'], first_n=9)
    result_dict['js_ring_distribution'] = jensenshannon(list(true_ring_distribution.values()),
                                                        list(ring_statistics.values()), base=2)
    result_dict['large_ring_ls'] = find_large_rings(result_dict['mol_list'], size=9)
    result_dict['avg_dihedral_angle'], result_dict['dihedral_angle_deviation_ls'], result_dict[
        'num_dihedral_angle_ls'] = \
        get_average_deviation(result_dict['mol_list'], typical_dihedral_angles, 'dihedral_angle_deviation')
    result_dict['avg_bond_angle'], result_dict['bond_angle_deviation_ls'], result_dict['num_bond_angle_ls'] = \
        get_average_deviation(result_dict['mol_list'], typical_bond_angles, 'bond_angle_deviation')


def get_min_bond_angle_deviation(typical_bond_angles, mol):
    '''
    Inputs:
        1. typical_dihedral_angles: a dict,
                key: the triple bond types, value: a list of typical dihedral angles
        2. mol: a rdkit mol
    Outputs:
        1. current_mean: the mean of all deviations between empirical dihedral angles and typical dihedral angles
        2. n: the number of dihedral angles in this empirical result
    '''
    typ_keys = typical_bond_angles.keys()
    n = current_mean = 0
    for k in typ_keys:
        angle_ls = get_bond_angle(mol, k)
        if len(angle_ls) == 0:
            continue
        res_k = np.array(angle_ls)
        typ_v = np.array(typical_bond_angles[k])
        diff = res_k.reshape(-1, 1) - typ_v.reshape(1, -1)
        deviations = np.abs(diff).min(axis=1)
        this_n = len(deviations)
        N = n + this_n
        current_mean = current_mean * n / N + np.sum(deviations) / N
        n = N
    return current_mean, n


def get_average_deviation(mol_ls, typical_values, key):
    deviation_ls, num_bond_angle_ls = [], []
    final_mean, N_prev = 0, 0
    for i, mol in enumerate(mol_ls):
        if key == 'bond_angle_deviation':
            deviation_mol, n = get_min_bond_angle_deviation(typical_values, mol)
        elif key == 'dihedral_angle_deviation':
            deviation_mol, n = get_min_dihedral_angle_deviation(typical_values, mol)
        else:
            raise ValueError(f"Not implemented for {key}")
        deviation_ls.append(deviation_mol)
        num_bond_angle_ls.append(n)
        if n == 0:
            print(i, n)
            continue
        N = N_prev + n
        final_mean = final_mean * N_prev / N + deviation_mol * n / N
        N_prev = N

    return final_mean, deviation_ls, num_bond_angle_ls


def get_min_dihedral_angle_deviation(typical_dihedral_angles, mol):
    '''
    Inputs:
        1. typical_dihedral_angles: a dict,
                key: the triple bond types, value: a list of typical dihedral angles
        2. mol: a rdkit mol
    Outputs:
        1. current_mean: the mean of all deviations between empirical dihedral angles and typical dihedral angles
        2. n: the number of dihedral angles in this empirical result
    '''
    typ_keys = typical_dihedral_angles.keys()
    dihedral_angle_statistics = {k: [] for k in typ_keys}
    get_dihedral_angle_faster(mol, dihedral_angle_statistics)
    n = current_mean = 0
    for k in typ_keys:
        if len(dihedral_angle_statistics[k]) == 0:  # if k does not show up, we skip it.
            continue  # if all keys do not show up, both n and current_mean will be 0. Important!
        res_k = np.array(dihedral_angle_statistics[k])
        typ_v = np.array(typical_dihedral_angles[k])
        diff = res_k.reshape(-1, 1) - typ_v.reshape(1, -1)
        deviations = np.abs(diff).min(axis=1)
        this_n = len(deviations)
        N = n + this_n
        current_mean = current_mean * n / N + np.mean(deviations) * this_n / N
        n = N
    return current_mean, n  # if n=0, current_mean=0.


def get_average_dihedral_deviation(mol_ls, typical_dihedral_angles):
    deviation_ls, num_dihedral_angle_ls = [], []
    final_mean, N_prev = 0, 0
    for i, mol in enumerate(mol_ls):
        deviation_mol, n = get_min_dihedral_angle_deviation(typical_dihedral_angles, mol)
        deviation_ls.append(deviation_mol)
        num_dihedral_angle_ls.append(n)
        if n == 0:  # to handle mols that have no interesting dihedral angle types
            print(i, n)
            continue

        N = N_prev + n
        final_mean = final_mean * N_prev / N + deviation_mol * n / N
        N_prev = N

    return final_mean, deviation_ls, num_dihedral_angle_ls


def summarize_res_all_rounds(improvement_all_rounds, improvement_one_round, save_path, logger=print):
    for k, v in improvement_all_rounds.items():
        v.append(improvement_one_round[k])
        improvements = np.array(v)
        improvements_mean = [f'{100 * x:.2f}%' for x in np.mean(improvements, axis=0)]
        improvements_std = [f'{100 * x:.1f}%' for x in np.std(improvements, axis=0)]
        logger(f"After {improvements.shape[0]} rounds, {k} {improvements_mean} \u00B1 {improvements_std}")
    torch.save(improvement_all_rounds, save_path)


def make_smile_symbols_with_bondtype(start_atom, end_atom, bondtype):
    '''make Smiles that can be used to as Smarts'''
    if start_atom > end_atom:
        start_atom, end_atom = end_atom, start_atom
    if bondtype == 12:
        return ''.join([start_atom.lower(), end_atom.lower()])  # Note that Cl does not involve any aromatic bond.
    elif bondtype == 1:
        return ''.join([start_atom, end_atom])
    elif bondtype == 2:
        return ''.join([start_atom, '=', end_atom])
    elif bondtype == 3:
        return ''.join([start_atom, '#', end_atom])
    else:
        raise ValueError(f'Not implemented for bond type {bondtype}')


def js_type(true_distribution, empirical_res):
    '''
    true_bond_freq: a dict with bond symbols as keys and probabilities as values
    bond_freq: a dict with bond symbols as keys and counts as values
    '''
    true_distribution_copy = copy.deepcopy(true_distribution)
    true_keys = true_distribution_copy.keys()
    keys = list(empirical_res.keys())
    values = np.array(list(empirical_res.values()))
    values = values / np.sum(values)
    empirical_res = {k: values[i] for i, k in enumerate(keys)}
    empirical_res_all = {}
    for k in true_keys:
        if k in keys:
            empirical_res_all[k] = empirical_res[k]
            keys.remove(k)
        else:
            empirical_res_all[k] = 0
    for k in keys:
        true_distribution_copy[k] = 0
        empirical_res_all[k] = empirical_res[k]

    return jensenshannon(np.array(list(true_distribution_copy.values())), np.array(list(empirical_res_all.values())),
                         base=2)


def js_histogram_distribution(true_distribution, empirical_res, first_n=10):
    '''
    true_bond_length: a dict with bond symbols as keys and probabilities as values
    bond_length: a dict with bond symbols as keys and lengths as values
    the first 10 most common bond types corresponds to 93.87% bond type apprearances
    '''
    true_distribution_copy = copy.deepcopy(true_distribution)
    if first_n is not None:
        true_keys = list(true_distribution_copy.keys())[:first_n]
    else:
        true_keys = list(true_distribution_copy.keys())
    keys = list(empirical_res.keys())
    js_divergence_ls = []
    for k in true_keys:
        if k in keys:
            if len(empirical_res[k]) > 0:
                res_array = np.array(empirical_res[k])
            else:
                continue
            max_value = true_distribution_copy[k]['maximum']
            min_value = true_distribution_copy[k]['minimum']
            res_array = np.where(res_array > max_value, max_value, res_array)
            res_array = np.where(res_array < min_value, min_value, res_array)
            counts = np.histogram(res_array, bins=true_distribution_copy[k]['bins'])[0]
            if np.sum(counts) == 0:
                prob_array = 1 / len(counts)
            else:
                prob_array = counts / np.sum(counts)
            js_divergence_ls.append(jensenshannon(true_distribution_copy[k]['probabilities'], prob_array, base=2))

    return np.mean(js_divergence_ls)


def get_bond_type_distribution_score(mol_list, true_bond_proba):
    bond_type_distribution_score_all = []
    true_keys = true_bond_proba.keys()
    bond_freq = {k: 0 for k in true_keys}
    for mol in mol_list:
        bond_type_distribution_score = []
        for mol_bond in mol.GetBonds():
            start_atom = mol_bond.GetBeginAtom().GetSymbol()
            end_atom = mol_bond.GetEndAtom().GetSymbol()
            # if start_atom > end_atom:
            #     bond_sym = make_smile_symbols_with_bondtype(end_atom, start_atom, int(mol_bond.GetBondType()))
            # else:
            bond_sym = make_smile_symbols_with_bondtype(start_atom, end_atom, int(mol_bond.GetBondType()))
            if bond_sym not in true_keys:
                bond_type_distribution_score.append(0)  # if bond_sym has never appeared in training and test set
            else:
                bond_type_distribution_score.append(true_bond_proba[bond_sym])
            if bond_sym in bond_freq.keys():
                bond_freq[bond_sym] += 1
            else:
                bond_freq[bond_sym] = 1
        bond_type_distribution_score_all.append(np.mean(bond_type_distribution_score))
    return copy.deepcopy(bond_type_distribution_score_all), bond_freq


def get_bond_angle_type_distribution(mol_list, true_bond_angle_type_proba):
    bond_angle_sym = true_bond_angle_type_proba.keys()
    bond_angle_type_distribution_ls = []
    bond_angle_type = {}
    for mol in mol_list:
        counter = current_mean = 0
        for angle_sym in bond_angle_sym:
            # angle_ls = get_bond_angle(mol, angle_sym)
            substructure = Chem.MolFromSmiles(angle_sym, sanitize=False)
            bond_pairs = mol.GetSubstructMatches(substructure)
            n = len(bond_pairs)
            if n > 0:
                current_mean = current_mean * counter / (counter + n) + true_bond_angle_type_proba[angle_sym] * n / (
                            counter + n)
                counter += n
                if angle_sym in bond_angle_type.keys():
                    bond_angle_type[angle_sym] += n
                else:
                    bond_angle_type[angle_sym] = n
        bond_angle_type_distribution_ls.append(current_mean)
    return copy.deepcopy(bond_angle_type_distribution_ls), bond_angle_type


bond_str_dict = {1: '', 2: '=', 3: '#'}


def get_bond_symbol(bond, bond_str_dict):
    a0 = bond.GetBeginAtom().GetSymbol()
    a1 = bond.GetEndAtom().GetSymbol()
    b = int(bond.GetBondType())
    if 0 < b < 4:
        sym = ''.join([a0, bond_str_dict[b], a1])
    elif b == 12:
        sym = ''.join([a0.lower(), a1.lower()])
    else:
        raise ValueError(f'Not implemented for bond type {b}.')

    return sym


def get_triple_bonds(mol):
    valid_triple_bonds = []
    for idx_bond, bond in enumerate(mol.GetBonds()):
        idx_begin_atom = bond.GetBeginAtomIdx()
        idx_end_atom = bond.GetEndAtomIdx()
        begin_atom = mol.GetAtomWithIdx(idx_begin_atom)
        end_atom = mol.GetAtomWithIdx(idx_end_atom)
        begin_bonds = begin_atom.GetBonds()
        valid_left_bonds = []
        for begin_bond in begin_bonds:
            if begin_bond.GetIdx() == idx_bond:
                continue
            else:
                valid_left_bonds.append(begin_bond)
        if len(valid_left_bonds) == 0:
            continue

        end_bonds = end_atom.GetBonds()
        for end_bond in end_bonds:
            if end_bond.GetIdx() == idx_bond:
                continue
            else:
                for left_bond in valid_left_bonds:
                    valid_triple_bonds.append([left_bond, bond, end_bond])
    return valid_triple_bonds


def get_equivalent_str(tri):
    t = tri[::-1]
    if t[-2:] == 'lC':
        t = t[:-2] + 'Cl'
    if t[:2] == 'lC':
        t = 'Cl' + t[2:]
    return t


def get_triple_bond_statistics(mol_list):
    # print('getting valid triple bonds')
    bond_tris = []
    for mol in mol_list:
        valid_triple_bonds = get_triple_bonds(mol)
        for bonds in valid_triple_bonds:
            bonds = '-'.join([get_bond_symbol(b, bond_str_dict) for b in bonds])
            bond_tris.append(bonds)  # the items are not consistent in terms of linking

    # to make the triple more sensible in terms of connecting
    # print('organizing valid triple bonds')
    natural_bond_tris = []
    for tri in bond_tris:
        ttt = tri.split('-')
        if (ttt[0][-1]).lower() != (ttt[1][0]).lower():  # if part A and part B do not connect naturally, we flip part A
            partA = ttt[0][::-1]
            if 'lC' == partA[:2]:  # we have to take care of the element 'Cl'
                partA = 'Cl' + partA[2:]
            elif 'lC' == partA[-2:]:
                partA = partA[:-2] + 'Cl'
        else:
            partA = ttt[0]
        if (ttt[1][-1]).lower() != (ttt[2][0]).lower():  # if part C and part B do not connect naturally, we flip part C
            partC = ttt[2][::-1]
            if 'lC' == partC[:2]:
                partC = 'Cl' + partC[2:]
            elif 'lC' == partC[-2:]:
                partC = partC[:-2] + 'Cl'
        else:
            partC = ttt[2]
        natural_bond_tris.append('-'.join([partA, ttt[1], partC]))

    # we store the statistics of bond triples in a dict
    bond_tris_dict = {}
    for tri in natural_bond_tris:
        keys = bond_tris_dict.keys()
        if tri not in keys and get_equivalent_str(tri) not in keys:
            bond_tris_dict[tri] = 1
        elif tri in keys:
            bond_tris_dict[tri] += 1
        elif get_equivalent_str(tri) in keys:
            bond_tris_dict[get_equivalent_str(tri)] += 1

    # we sort the bond triple appearances in a descending order
    keys = list(bond_tris_dict.keys())
    values = list(bond_tris_dict.values())
    sorted_idx = np.argsort(values)[::-1]
    bond_tris_dict = {keys[i]: values[i] for i in sorted_idx}
    return bond_tris_dict


def get_triple_bond_type_score(mol_list, true_triple_bond_type_distribution):
    true_keys = true_triple_bond_type_distribution.keys()
    triple_bond_type_score_ls = []
    triple_bond_type_dict = {}
    for mol in mol_list:
        mol_triple_bond_type_dict = get_triple_bond_statistics([mol])
        current_mean = counter = 0
        for k, n in mol_triple_bond_type_dict.items():
            if k in true_keys:
                current_mean = current_mean * counter / (counter + n) + true_triple_bond_type_distribution[k] * n / (
                        counter + n)
                counter += n
            if k in triple_bond_type_dict.keys():
                triple_bond_type_dict[k] += n
            else:
                triple_bond_type_dict[k] = n

        triple_bond_type_score_ls.append(current_mean)
    return triple_bond_type_score_ls, triple_bond_type_dict


def get_bond_lengths(mol, bond_smi):  # cc, CC, C=C, C#C, CN, C=N, C#N, CO, C=O, NO, etc. (35 kinds in total)
    bonds = Chem.MolFromSmiles(bond_smi, sanitize=False)
    bonds = mol.GetSubstructMatches(bonds)
    lengths = []
    for bond in bonds:
        lengths.append(rdMolTransforms.GetBondLength(mol.GetConformer(), bond[0], bond[1]))
    return lengths


def get_bond_length_distribution_log_likelihood(mol_list, true_bond_length_distribution, first_n=10):
    bond_length_distribution_log_likelihood_all = []
    bond_symbols = true_bond_length_distribution.keys()
    if first_n is not None:  # only calculate log likelihood for the fist n, i.e. most common, bond symbols of all defined bond symbols
        bond_symbols = list(bond_symbols)[:first_n]
    else:
        bond_symbols = list(bond_symbols)
    bond_lengths = {}
    for mol in mol_list:
        counter = current_mean = 0
        for bond in bond_symbols:
            bond_length = get_bond_lengths(mol, bond)
            if len(bond_length) > 0:
                if bond not in bond_lengths.keys():
                    bond_lengths[bond] = bond_length
                else:
                    bond_lengths[bond].extend(bond_length)
                log_likelihood = true_bond_length_distribution[bond]['GaussianMixture'].score_samples(
                    np.array(bond_length).reshape(-1, 1))
                n = log_likelihood.shape[0]
                current_mean = current_mean * counter / (counter + n) + np.mean(log_likelihood) * n / (counter + n)
                counter = counter + n
        bond_length_distribution_log_likelihood_all.append(current_mean)

    return copy.deepcopy(bond_length_distribution_log_likelihood_all), bond_lengths


from rdkit.Chem.rdMolTransforms import GetAngleDeg


def get_bond_angle(mol, bond_smi='CCC'):
    deg_list = []
    substructure = Chem.MolFromSmiles(bond_smi, sanitize=False)
    bond_pairs = mol.GetSubstructMatches(substructure)
    for pair in bond_pairs:
        deg_list += [GetAngleDeg(mol.GetConformer(), *pair)]
        assert mol.GetBondBetweenAtoms(pair[0], pair[1]) is not None
        assert mol.GetBondBetweenAtoms(pair[2], pair[1]) is not None
    return deg_list


def get_bond_angle_distribution_log_likelihood(mol_list, true_bond_angle_distribution, first_n=30):
    bond_angle_distribution_log_likelihood_all = []
    if first_n is not None:
        bond_symbols = list(true_bond_angle_distribution.keys())[:first_n]
    else:
        bond_symbols = list(true_bond_angle_distribution.keys())
    bond_angles = {}
    for mol in mol_list:
        counter = 0
        current_mean = 0
        for bond in bond_symbols:
            bond_angle = get_bond_angle(mol, bond)
            if len(bond_angle) > 0:
                if bond not in bond_angles.keys():
                    bond_angles[bond] = bond_angle
                else:
                    bond_angles[bond].extend(bond_angle)
                log_likelihood = true_bond_angle_distribution[bond]['GaussianMixture'].score_samples(
                    np.array(bond_angle).reshape(-1, 1))
                n = log_likelihood.shape[0]
                current_mean = current_mean * counter / (counter + n) + np.mean(log_likelihood) * n / (counter + n)
                counter = counter + n
        bond_angle_distribution_log_likelihood_all.append(current_mean)

    return copy.deepcopy(bond_angle_distribution_log_likelihood_all), bond_angles


def get_one_dihedral_angle(mol, bonds):
    bond0 = bonds[0]
    atom0 = bond0.GetBeginAtomIdx()
    atom1 = bond0.GetEndAtomIdx()

    bond1 = bonds[1]
    atom1_0 = bond1.GetBeginAtomIdx()
    atom1_1 = bond1.GetEndAtomIdx()
    if atom0 == atom1_0:
        i, j, k = atom1, atom0, atom1_1
    elif atom0 == atom1_1:
        i, j, k = atom1, atom0, atom1_0
    elif atom1 == atom1_0:
        i, j, k = atom0, atom1, atom1_1
    elif atom1 == atom1_1:
        i, j, k = atom0, atom1, atom1_0

    bond2 = bonds[2]
    atom2_0 = bond2.GetBeginAtomIdx()
    atom2_1 = bond2.GetEndAtomIdx()
    if atom2_0 == k:
        l = atom2_1
    elif atom2_1 == k:
        l = atom2_0
    angle = Chem.rdMolTransforms.GetDihedralDeg(mol.GetConformer(), i, j, k, l)
    return angle


def get_dihedral_angle_faster(mol, dihedral_angle_dict):
    bonds_list = get_triple_bonds(mol)
    keys = dihedral_angle_dict.keys()
    for bonds in bonds_list:
        sym = '-'.join([get_bond_symbol(b, bond_str_dict) for b in bonds])
        sym1 = get_equivalent_str(sym)
        if sym in keys:
            angle = get_one_dihedral_angle(mol, bonds)
            dihedral_angle_dict[sym].append(angle)
        elif sym1 in keys:
            angle = get_one_dihedral_angle(mol, bonds[::-1])
            dihedral_angle_dict[sym1].append(angle)
        else:
            continue


def get_dihedral_angle_distribution_log_likelihood(mol_list, true_dihedral_angle_distribution, first_n=10):
    dihedral_angle_distribution_log_likelihood_all = []
    if first_n is not None:
        bond_symbols = list(true_dihedral_angle_distribution.keys())[:first_n]
    else:
        bond_symbols = list(true_dihedral_angle_distribution.keys())
    dihedral_angles_all = {k: [] for k in bond_symbols}
    dihedral_angles_one_mol = {k: [] for k in bond_symbols}
    for mol in mol_list:
        dihedral_angle_dict = copy.deepcopy(dihedral_angles_one_mol)
        get_dihedral_angle_faster(mol, dihedral_angle_dict)
        counter = current_mean = 0
        for bond in bond_symbols:
            dihedral_angle = dihedral_angle_dict[bond]
            if len(dihedral_angle) > 0:
                dihedral_angles_all[bond].extend(dihedral_angle)
                gm = true_dihedral_angle_distribution[bond]['GaussianMixture']
                log_likelihood = gm.score_samples(np.array(dihedral_angle).reshape(-1, 1))
                n = log_likelihood.shape[0]
                current_mean = current_mean * counter / (counter + n) + np.mean(log_likelihood) * n / (counter + n)
                counter = counter + n
        dihedral_angle_distribution_log_likelihood_all.append(current_mean)

    return copy.deepcopy(dihedral_angle_distribution_log_likelihood_all), dihedral_angles_all


def find_num_rings(mol, return_ssr=False):
    ssr = Chem.GetSymmSSSR(mol)
    num_rings = len(ssr)
    if return_ssr:
        return num_rings, ssr
    else:
        return num_rings


def get_ring_statistics(mol_ls, first_n=9):
    has_all_i_ring_dict = {}  # {i:0 for i in i_ls}
    for mol in mol_ls:
        num_rings, ssr = find_num_rings(mol, return_ssr=True)
        if num_rings == 0:
            if num_rings not in has_all_i_ring_dict.keys():
                has_all_i_ring_dict[num_rings] = 1
            else:
                has_all_i_ring_dict[num_rings] += 1
            # continue
        else:
            ring_footprints = []
            for r in ssr:
                k = len(r)
                if k not in ring_footprints:
                    if k not in has_all_i_ring_dict.keys():
                        has_all_i_ring_dict[k] = 1
                    else:
                        has_all_i_ring_dict[k] += 1
                    ring_footprints.append(k)
    keys = [0] + list(range(3, first_n))
    formatted_dict = {}
    for k in keys:
        if k in has_all_i_ring_dict.keys():
            formatted_dict[k] = has_all_i_ring_dict[k]
        else:
            formatted_dict[k] = 0
    selected_idx = np.array(list(has_all_i_ring_dict.keys())) >= first_n
    last_v = np.sum(np.array(list(has_all_i_ring_dict.values()))[selected_idx])
    formatted_dict[str(first_n) + '+'] = last_v
    return formatted_dict, has_all_i_ring_dict


def get_ring_distribution_score(mol_ls, true_ring_distribution):
    ring_distr_score = []
    for mol in mol_ls:
        ssr = Chem.GetSymmSSSR(mol)
        if len(ssr) == 0:
            curr_mean = true_ring_distribution[0]
        else:
            n = curr_mean = 0
            for r in ssr:
                ring_size = len(r)
                if ring_size < 9:
                    score = true_ring_distribution[ring_size]
                else:
                    score = true_ring_distribution['9+']
                curr_mean = curr_mean * n / (1 + n) + score / (1 + n)
                n += 1
        ring_distr_score.append(curr_mean)
    return ring_distr_score
