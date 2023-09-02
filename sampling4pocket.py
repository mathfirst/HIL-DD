import argparse, os, shutil, torch, sys, copy
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from datetime import datetime # we use time as a filename when creating that logfile.

import numpy as np

from utils.util_flow import get_logger, load_config, MeanSubtractionNorm, EMA
from torch_geometric.transforms import Compose
from torch_geometric.data import Data
from utils.util_data import get_dataset, ProteinElement2IndexDict, MAP_ATOM_TYPE_AROMATIC_TO_INDEX_wo_h, torchify_dict
from models.models import Model_CoM_free, Classifier
import utils.transforms as trans
from utils.util_sampling import sampling_val
from utils.data import PDBProtein

FOLLOW_BATCH = ('protein_element', 'ligand_element', 'ligand_bond_type',)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="sampling.yml")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logdir', type=str, default='./logs_sampling')
    parser.add_argument('--desc', type=str, default='')
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--pdb_path', type=str, default='')
    parser.add_argument('--receptor_path', type=str, default='')
    # specify the number of atoms in a ligand or a specific number, e.g. 18
    parser.add_argument('--num_atoms', type=str, default='18')  # 'reference' means the same number of atoms as the reference ligand in the test set
    args = parser.parse_args()

    pocket_dict = PDBProtein(args.pdb_path).to_dict_atom()
    data = Data()
    for k, v in torchify_dict(pocket_dict).items():
        data['protein_' + k] = v

    # logging
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_dir = os.path.join(args.logdir, current_time + args.suffix)
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, "log.txt")
    logger = get_logger('train', log_filename)
    logger.info(f'pdb_path: {args.pdb_path}')
    shutil.copy(args.config, os.path.join(log_dir, 'config-' + '.yml'))
    current_file = __file__  # get the name of the currently executing python file
    shutil.copy(current_file, os.path.join(log_dir, os.path.basename(current_file).split('.')[0] + '.py'))
    device = args.device if torch.cuda.is_available() else "cpu"
    logger.info(f"device: {device}")
    config = load_config(args.config)

    if args.receptor_path and config.sampling.cal_vina_score:
        logger.info('Vina scores will be calculated for the generated molecules.')
        cal_vina_score = True
    else:
        logger.info('The complete (untailored) protein pocket is not provided, so Vina score will not be calculated.')
        cal_vina_score = False

    hidden_dim = int(config.model.hidden_dim)  # EGNN hidden feature dim
    num_timesteps = config.model.num_timesteps  # time steps for the rectified flow model, default: 1000
    depth = config.model.depth  # this refers to how many layers EGNN will have
    use_mlp = config.model.use_mlp  # whether or not apply mlp to the output of EGNN
    gcl_layers = config.model.gcl_layers  # num of gcl layers
    feat_t_dim = config.model.feat_t_dim  # feature time embedding dim
    egnn_attn = config.model.egnn_attn  # use attn or not, this is a setting from egnn module
    agg_norm_factor = config.model.agg_norm_factor  # (default: 5)
    add_bond = config.train.bond
    logger.info(f"hidden_dim: {hidden_dim}, feat_t_dim: {feat_t_dim}, depth: {depth}, "
                f"gcl_layers: {gcl_layers}, egnn_attn: {egnn_attn}, agg_norm_factor: {agg_norm_factor}, "
                f"use_mlp: {use_mlp}, add_bond: {add_bond}, num_timesteps: {num_timesteps}")
    pos_scale = float(config.data.pos_scale)  # training loss stability
    ema_decay = config.train.ema_decay  # EMA decay rate
    distance_sin_emb = config.train.distance_sin_emb  # this is a setting from EGNN module and we set it True
    optimize_embedding = config.train.optimize_embedding  # optimize embeddings (default: True), no big difference
    n_knn = config.train.n_knn  # number of nearest neighbors
    opt_time_emb = config.train.opt_time_emb
    logger.info(f"pos_scale: {pos_scale}, knn: {n_knn}, ema_decay: {ema_decay}, "
                f"dist_sin_emb: {distance_sin_emb}, optim_emb: {optimize_embedding}, "
                f"opt_time_emb: {opt_time_emb}")
    # Transforms
    mode = config.data.transform.ligand_atom_mode
    ligand_featurizer = trans.FeaturizeLigandAtom(mode)
    transform_list = [ligand_featurizer, ]
    if config.data.transform.random_rot:
        logger.info("apply random rotation")
        transform_list.append(trans.RandomRotation())
    transform = Compose(transform_list)
    subtract_mean = MeanSubtractionNorm()   # This is used to center positions.
    # loading data
    sample_result_dir = os.path.join(log_dir, 'sample-results')
    os.makedirs(sample_result_dir, exist_ok=True)
    logger.info("building model...")
    model_ema = Model_CoM_free(num_protein_element=len(ProteinElement2IndexDict), num_amino_acid=20,
                  num_ligand_element=len(MAP_ATOM_TYPE_AROMATIC_TO_INDEX_wo_h), depth=depth,  # number of EGNN layers
                  hidden_dim=hidden_dim, n_steps=num_timesteps, use_mlp=use_mlp, gcl_layers=gcl_layers,
                  feat_time_dim=feat_t_dim, optimize_embedding=optimize_embedding, agg_norm_factor=agg_norm_factor,
                  distance_sin_emb=distance_sin_emb, egnn_attn=egnn_attn, n_knn=n_knn, device=device,
                  add_bond_model=add_bond, opt_time_emb=opt_time_emb).to(device)
    if config.train.checkpoint:
        logger.info("loading checkpoint " + str(config.train.ckp_path))
        ckp = torch.load(config.train.ckp_path, map_location=device)
        model_ema.load_state_dict(ckp['model_ema_state_dict'])
        if config.sampling.classifier_guidance:
            classifier = Classifier(in_node_feat=32, hidden_node_feat=64, out_node_feat=32,
                                    time_emb_feat=8, depth=2, num_timesteps=num_timesteps, device=device,
                                    opt_time_emb=False, add_bond=True, use_pocket=False,
                                    num_protein_element=len(ProteinElement2IndexDict),
                                    num_amino_acid=20, dim_protein_element=16, dim_amino_acid=16, dim_is_backbone=8)
            classifier.load_state_dict(torch.load(config.sampling.cls_ckp_path, map_location=device)['cls_state_dict'])
        else:
            classifier = None
    vina, qed, sa, lipinski, logp, diversity, time, high_affinity = [], [], [], [], [], [], [], []
    calculator = sampling_val(model_ema, data, 0, pos_scale, sample_result_dir, logger, device,
                 ProteinElement2IndexDict, num_timesteps=num_timesteps, mode=mode,
                 num_samples=config.sampling.num_samples, cal_vina_score=cal_vina_score,
                 t_sampling_strategy='uniform', cal_straightness=False,
                 num_spacing_steps=config.sampling.num_spacing_steps,
                 drop_unconnected_mol=config.sampling.drop_unconnected_mol, bond_emb=config.sampling.bond_emb,
                 sampling_method=config.sampling.sampling_method, batch_size=config.sampling.batch_size,
                 cls_fn=classifier, s=config.sampling.guidance_scale, num_atoms=args.num_atoms,
                              protein_pdbqt_dir='configs/test_protein', protein_pdbqt_file_path=args.receptor_path)
    vina.append(calculator.stat_dict['vina_score_mean'])
    qed.append((calculator.stat_dict['qed_mean']))
    sa.append(calculator.stat_dict['sa_mean'])
    lipinski.append((calculator.stat_dict['lipinski_mean']))
    logp.append(calculator.stat_dict['logp_mean'])
    diversity.append((calculator.stat_dict['diversity']))
    time.append(calculator.stat_dict['time_mean'])
    high_affinity.append((calculator.stat_dict['high_affinity']))
    logger.info(f"results for the provided pocket {args.pdb_path}:")
    logger.info(f"Vina score: {np.mean(vina):.3f} \u00b1 {np.std(vina):.3f}")
    logger.info(f"High affinity: {np.mean(high_affinity):.4f} \u00b1 {np.std(high_affinity):.4f}")
    logger.info(f"QED: {np.mean(qed):.3f} \u00b1 {np.std(qed):.3f}")
    logger.info(f"SA: {np.mean(sa):.3f} \u00b1 {np.std(sa):.3f}")
    logger.info(f"Lipinski: {np.mean(lipinski):.3f} \u00b1 {np.std(lipinski):.3f}")
    logger.info(f"LogP: {np.mean(logp):.3f} \u00b1 {np.std(logp):.3f}")
    logger.info(f"Diversity: {np.mean(diversity):.3f} \u00b1 {np.std(diversity):.3f}")
    logger.info(f"Time: {np.mean(time):.3f} \u00b1 {np.std(time):.3f}")
    logger.info(f"{config.sampling.num_samples} molecules have been sampled.")

