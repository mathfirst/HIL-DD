# Date: 2022.12.13, Author: Kaikai Zhao
# 1. We construct embeddings for all involved features.
# 2. We define feat_dim_factor in our model so that we can control the dimension of embedding featrues.
# 3. We give greater importance for those time steps which have larger pos-loss values.
# 4. We try to use mean abosolute error (MAE) instead of mean squared error (MSE).
# 5. We add bond_attr as edge_attr when applicable. 2022-12-21-23:13
# Date: 2023.2.4, Author: Kaikai Zhao
# 1. make feature dim more compact
# 2. add pocket_flag to indicate which feature is pocket

import argparse, os, shutil, torch, sys, copy
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# sys.path.append('E:\pytorch-related\TargetDiff2RectifiedFlow\molopt')
# sys.path.append('./molopt')

from datetime import datetime # we use time as a filename when creating that logfile.
from utils.util_flow import get_logger, load_config, MeanSubtractionNorm, \
    convert_ele_emb2ele_types, measure_straightness
# from torch_geometric.transforms import Compose
# from torch_geometric.loader import DataLoader
from utils.util_data import ProteinElement2IndexDict, MAP_ATOM_TYPE_AROMATIC_TO_INDEX_wo_h, torchify_dict, extract_data, perterb_X0
from models.models import Model_CoM_free
import numpy as np
import utils.transforms as trans
from utils.data import PDBProtein, parse_sdf_file
from torch_geometric.data import Data
from utils import reconstruct
from rdkit import Chem
from utils.evaluate import evaluate_metrics, find_substructure, similarity, calculate_vina_score
from utils.util_sampling import ode
from utils.analyze import check_stability
FOLLOW_BATCH = ('protein_element', 'ligand_element', 'ligand_bond_type',)

# seed_torch()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="./configs/sampling.yml")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logdir', type=str, default='./logs_refine')
    parser.add_argument('--pdb', type=str, default='')
    parser.add_argument('--sdf', type=str, default='')
    args = parser.parse_args()
    config = load_config(args.config)
    pocket_dict = PDBProtein(args.pdb).to_dict_atom()
    data = Data()
    for k, v in torchify_dict(pocket_dict).items():
        data['protein_' + k] = v
    ligand_dict = parse_sdf_file(args.sdf)
    for k, v in torchify_dict(ligand_dict).items():
        data['ligand_' + k] = v
    mol_drug = Chem.MolToMolFile(args.sdf)
    smiles_drug = Chem.MolToSmiles(mol_drug)
    # logging
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_dir = os.path.join(args.logdir, current_time)
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, "log.txt")
    logger = get_logger('gen-mol-from-drug', log_filename)
    shutil.copy(args.config, os.path.join(log_dir, 'config' + '.yml'))
    current_file = __file__  # get the name of the currently executing python file
    shutil.copy(current_file, os.path.join(log_dir, os.path.basename(current_file).split('.')[0] + '.py'))

    device = args.device if torch.cuda.is_available() else "cpu"
    logger.info(f"device: {device}")

    hidden_dim = int(config.model.hidden_dim)  # EGNN hidden feature dim
    num_timesteps = config.model.num_timesteps  # time steps for the rectified flow model, default: 1000
    depth = config.model.depth  # this refers to how many layers EGNN will have
    use_mlp = config.model.use_mlp  # whether or not apply mlp to the output of EGNN
    gcl_layers = config.model.gcl_layers  # num of gcl layers
    feat_t_dim = config.model.feat_t_dim  # feature time embedding dim
    egnn_attn = config.model.egnn_attn  # use attn or not, this is a setting from egnn module
    agg_norm_factor = config.model.agg_norm_factor  # (default: 5)
    add_bond = config.train.bond
    logger.info(f"hidden_dim: {hidden_dim}, feat_t_dim: {feat_t_dim}, depth: {depth}, gcl_layers: {gcl_layers}, "
                f"egnn_attn: {egnn_attn}, agg_norm_factor: {agg_norm_factor}\n"
                f"use_mlp: {use_mlp}, add_bond: {add_bond}, num_timesteps: {num_timesteps}")

    pos_scale = float(config.data.pos_scale)  # training loss stability
    ema_decay = config.train.ema_decay  # EMA decay rate
    print_interval = config.train.print_interval  # logger print frequency
    distance_sin_emb = config.train.distance_sin_emb  # this is a setting from EGNN module and we set it True
    optimize_embedding = config.train.optimize_embedding  # optimize embeddings (default: True), no big difference
    n_knn = config.train.n_knn  # number of nearest neighbors
    bs = config.train.batch_size  # batch_size. Experiments show that bs=1 makes loss decrease stably.
    len_history = config.train.len_history
    loss_type = config.train.loss_type
    time_sampling_method = config.train.time_sampling
    opt_time_emb = config.train.opt_time_emb
    n_steps_per_iter = config.train.n_steps_per_iter
    logger.info(f"bs: {bs}, pos_scale: {pos_scale}, knn: {n_knn}, ema_decay: {ema_decay}, len_history: {len_history}, "
                f"print_interval: {print_interval}, dist_sin_emb: {distance_sin_emb}, optim_emb: {optimize_embedding}, "
                f"loss_type: {loss_type}, time_samping_method: {time_sampling_method}, opt_time_emb: {opt_time_emb}, "
                f"n_steps_per_iter: {n_steps_per_iter}")

    num_pockets = config.train.num_pockets
    num_samples = config.train.num_samples
    only_sample = config.train.only_sample  # if True, this code will be only used to sample
    logger.info(f"When sampling, num_pockets: {num_pockets}, num_samples: {num_samples}, only_sample: {only_sample}\n"
                f"********************************************************************")

    init_lr = config.train.optimizer.lr
    lr_warmup_steps = config.train.lr_scheduler.lr_warmup_steps
    lr_stepsize = config.train.lr_scheduler.stepsize  # lr decay stepsize
    lr_gamma = config.train.lr_scheduler.gamma  # lr decay factor
    wd = config.train.optimizer.weight_decay
    logger.info(f"init_lr: {init_lr}, lr_gamma: {lr_gamma}, lr_decay_stepsize: {lr_stepsize}, "
                f"lr_warmup_steps: {lr_warmup_steps}, wd: {wd}\n"
                f"****************************************************************************")

    # Transforms
    mode = config.data.transform.ligand_atom_mode
    # protein_featurizer = trans.FeaturizeProteinAtom(without_hydrogen=True)
    ligand_featurizer = trans.FeaturizeLigandAtom(mode)
    data = ligand_featurizer(data)
    subtract_mean = MeanSubtractionNorm()   # This is used to center positions.
    # loading data

    sample_result_dir = os.path.join(log_dir, 'sample-results')
    os.makedirs(sample_result_dir, exist_ok=True)

    logger.info("building model...")
    model = Model_CoM_free(num_protein_element=len(ProteinElement2IndexDict), num_amino_acid=20,
                  num_ligand_element=len(MAP_ATOM_TYPE_AROMATIC_TO_INDEX_wo_h), depth=depth,  # number of EGNN layers
                  hidden_dim=hidden_dim, n_steps=num_timesteps, use_mlp=use_mlp, gcl_layers=gcl_layers,
                  feat_time_dim=feat_t_dim, optimize_embedding=optimize_embedding, agg_norm_factor=agg_norm_factor,
                  distance_sin_emb=distance_sin_emb, egnn_attn=egnn_attn, n_knn=n_knn, device=device,
                  add_bond_model=add_bond, opt_time_emb=opt_time_emb).to(device)

    logger.info(f"No. of parameters: {np.sum([p.numel() for p in model.parameters()])}")
    logger.info(
        f"No. of parameters using p.requires_grad: {np.sum([p.numel() for p in model.parameters() if p.requires_grad])}")
    ckp_model = torch.load(config.train.ckp_path, map_location='cuda')
    model.load_state_dict(ckp_model['model_ema_state_dict'])
    protein_pos, protein_ele, protein_amino_acid, protein_is_backbone, ligand_pos, ligand_ele \
        = extract_data(data, ProteinElement2IndexDict)
    protein_pos, protein_pos_mean = subtract_mean(protein_pos, verbose=True)
    num_ligand_atoms = ligand_pos.shape[0]
    logger.info(f"No. of ligand atoms: {num_ligand_atoms}")
    ligand_bond_types = data.ligand_bond_type  # .to(device)
    ligand_bond_indices = data.ligand_bond_index  # .to(device)
    ligand_bond_types_all, bond_edges = model.get_ligand_bond_types_all(num_ligand_atoms=num_ligand_atoms,
                                                                        ligand_bond_indices=ligand_bond_indices,
                                                                        ligand_bond_types=ligand_bond_types, device=device)
    num_edges = bond_edges.shape[1]
    ligand_bond_features = model.get_bond_type_features(ligand_bond_types_all)
    # calculator_pref = sample4pocket(model_pref, val_loader, pos_scale, ema_sample_result_dir, logger, device,
    #                                 ProteinElement2IndexDict, num_timesteps=num_timesteps, mode=mode,
    #                                 num_pockets=pocket_idx, num_samples=1, cal_vina_score=True,
    #                                 all_val_pockets=False, cal_straightness=False, num_spacing_steps=False,
    #                                 starting_pocket_id=0, t_sampling_strategy='uniform')
    # print(f"original smiles: {calculator_pref.result_dict['smiles_list'][-1]}")
    z_pos = ligand_pos * pos_scale - protein_pos_mean
    z_feat = model.get_ligand_features(ligand_ele)
    X0_fn = os.path.join(args.logdir, f"{os.path.basename(args.pdb)[:4]}.pt")
    if os.path.isfile(X0_fn):
        X0_pos, X0_ele_feat, X0_bond_feat = torch.load(X0_fn, map_location=device)
    else:
        X0_pos, X0_ele_feat, X0_bond_feat, v_tuple = ode(model, protein_pos=protein_pos, protein_ele=protein_ele,
                                                         protein_amino_acid=protein_amino_acid,
                                                         protein_is_backbone=protein_is_backbone,
                                                         z_pos=z_pos, z_feat=z_feat, bond_features=ligand_bond_features,
                                                         bond_edges=bond_edges, device=device, reverse=True,
                                                         num_timesteps=num_timesteps, return_v=True)
        torch.save({'X0_pos': X0_pos,
                    'X0_ele': X0_ele_feat,
                    'X0_bond': X0_bond_feat}, X0_fn)
        # print(torch.allclose(X0_pos.cpu(), calculator_pref.result_dict['X0_pos_list'][-1]), f"norm: {torch.norm(X0_pos.cpu()-calculator_pref.result_dict['X0_pos_list'][-1]):.3f}")
        # print(torch.allclose(X0_ele_feat.cpu(), calculator_pref.result_dict['X0_ele_emb_list'][-1]), f"norm: {torch.norm(X0_ele_feat.cpu()-calculator_pref.result_dict['X0_ele_emb_list'][-1]):.3f}")
        # print(torch.allclose(X0_bond_feat.cpu(), calculator_pref.result_dict['X0_bond_emb_list'][-1]), f"norm: {torch.norm(X0_bond_feat.cpu()-calculator_pref.result_dict['X0_bond_emb_list'][-1]):.3f}")
        logger.info(f"straightness for v_pos: {measure_straightness(X0_pos.cpu(), z_pos, v_tuple[0], 'l1').item():.2f}")
        logger.info(f"straightness for v_ele: {measure_straightness(X0_ele_feat.cpu(), z_feat, v_tuple[1], 'l1').item():.2f}")
        logger.info(f"straightness for v_bond: {measure_straightness(X0_bond_feat.cpu(), ligand_bond_features, v_tuple[2], 'l1').item():.2f}")
    sdf_path = os.path.join(log_dir, 'SDF')
    os.makedirs(sdf_path, exist_ok=True)
    pt_path = os.path.join(log_dir, 'pt')
    os.makedirs(pt_path, exist_ok=True)
    for i in range(100):
        X0_pos_copy, X0_ele_feat_copy, X0_bond_feat_copy = perterb_X0(batch_size=1, X0_pos=X0_pos, X0_element_embedding=X0_ele_feat,
                                                                      X0_bond_embedding=X0_bond_feat, noise_level=0.1)
        X1_pos, X1_ele_feat, X1_bond_feat = ode(model, protein_pos=protein_pos, protein_ele=protein_ele,
                                                protein_amino_acid=protein_amino_acid,
                                                protein_is_backbone=protein_is_backbone,
                                                z_pos=X0_pos_copy, z_feat=X0_ele_feat_copy,
                                                bond_features=X0_bond_feat_copy, bond_edges=bond_edges,
                                                device=device, reverse=False, num_timesteps=num_timesteps)
        pred_ele_types = convert_ele_emb2ele_types(model, X1_ele_feat)
        X1_pos_copy = X1_pos / pos_scale + protein_pos_mean.to(device)
        mol = reconstruct.reconstruct_from_generated(X1_pos_copy.tolist(), pred_ele_types, aromatic=None,
                                                     basic_mode=True, bond_type=None, bond_index=None)
        smiles = Chem.MolToSmiles(mol)
        logger.info(f"smiles: {smiles}, smiles_gt: {smiles_drug}")
        logger.info(f"smilarity: {similarity(mol, mol_drug)}")
        if '.' not in smiles:
            r_stable = check_stability(X1_pos, pred_ele_types)
            logger.info(f"{r_stable}")
            if find_substructure(smiles, 'c1ccccc1Nc2ncccn2'):
                logger.info(f"c1ccccc1Nc2ncccn2 found!")
            if find_substructure(smiles, 'c12ccccc1cncn2'):
                logger.info(f"c12ccccc1cncn2 found!")
        else:
            logger.info(f"This mol is not connected.")
            continue
        mol_path = os.path.join(sdf_path, f'{current_time}_{i}.sdf')
        try:
            Chem.MolToMolFile(mol, mol_path)
            evaluate_metrics(mol, logger=logger.info)
            torch.save({'mol_path': mol_path,
                        'smiles': smiles,
                        'mol': mol,
                        'X0_pos': X0_pos,
                        'X0_ele': X0_ele_feat,
                        'X0_bond': X0_bond_feat}, os.path.join(pt_path, f'{current_time}_{i}.pt'))
        except Exception as err:
            logger.info(f"{err}")
    # calculate_vina_score(mol, pocket_idx, os.path.join(log_dir, 'test.sdf'))
    sys.exit()
