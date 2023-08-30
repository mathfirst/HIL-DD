import argparse, os, shutil, torch, copy, json, sys, time
from datetime import datetime  # we use time as a filename when creating that logfile.
from utils.util_flow import get_logger, load_config, MeanSubtractionNorm, EMA, get_all_bond_edges, \
    TimeStepSampler, OnlineAveraging, CLASSIFIER_LOSS_FUNC_DICT, PREFERENCE_LOSS_FUNC_DICT
from torch_geometric.transforms import Compose
from torch_geometric.loader import DataLoader
from utils.util_data import get_dataset, ProteinElement2IndexDict, MAP_ATOM_TYPE_AROMATIC_TO_INDEX_wo_h, extract_data, \
    PreparePrefData4UI, prepare_inputs, NpEncoder
from models.models import Model_CoM_free, Classifier
import numpy as np
import utils.transforms as trans
from utils.util_sampling import sampling_val
from utils.data import PDBProtein
from utils.evaluate import add_more_ls_to_result_dict, PDB_dict
import configs.metric_config
from torch_geometric.data import Data
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

FOLLOW_BATCH = ('protein_element', 'ligand_element', 'ligand_bond_type',)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('params', nargs='*')
    parser.add_argument('--config', type=str, default="configs/config_ui.yml")
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--logdir', type=str, default='logs/')
    parser.add_argument('--pdb_id', type=str, default='')  # 4-character PDB ID
    parser.add_argument('--suffix', type=str, default='')
    args = parser.parse_args()
    config = load_config(args.config)
    query = config.pref.metric
    # logging
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_dir = os.path.join(args.logdir, args.params[0])
    print('log_dir', log_dir)
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, "log-proposals.txt")
    logger = get_logger('pref', log_filename)
    shutil.copy(args.config, os.path.join(log_dir, 'config.yml'))
    shutil.copy('models/models.py', os.path.join(log_dir, 'models.py'))
    shutil.copy('utils/util_flow.py', os.path.join(log_dir, 'util_flow.py'))
    shutil.copy('utils/util_data.py', os.path.join(log_dir, 'util_data.py'))
    shutil.copy('utils/evaluate.py', os.path.join(log_dir, 'evaluate.py'))
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
                f"egnn_attn: {egnn_attn}, agg_norm_factor: {agg_norm_factor}, use_mlp: {use_mlp}, "
                f"add_bond: {add_bond}, num_timesteps: {num_timesteps}")

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
    ligand_featurizer = trans.FeaturizeLigandAtom(mode)
    transform_list = [ligand_featurizer, ]
    if config.data.transform.random_rot:
        logger.info("apply random rotation")
        transform_list.append(trans.RandomRotation())
    transform = Compose(transform_list)
    subtract_mean = MeanSubtractionNorm()  # This is used to center positions.
    dataset, subsets = get_dataset(
        config=config.data,
        transform=transform,
    )

    train_set = subsets['train']
    collate_exclude_keys = ['ligand_nbh_list']
    train_loader = DataLoader(
        train_set,
        batch_size=config.train.batch_size,
        shuffle=config.train.shuffle,
        num_workers=config.train.num_workers,
        follow_batch=FOLLOW_BATCH,
        exclude_keys=collate_exclude_keys,
        pin_memory=True
    )
    del dataset, subsets

    sample_result_dir = os.path.join(log_dir, 'sample-results')
    os.makedirs(sample_result_dir, exist_ok=True)

    logger.info("building model...")
    model_pref = Model_CoM_free(num_protein_element=len(ProteinElement2IndexDict), num_amino_acid=20,
                                num_ligand_element=len(MAP_ATOM_TYPE_AROMATIC_TO_INDEX_wo_h), depth=depth, # number of EGNN layers
                                hidden_dim=hidden_dim, n_steps=num_timesteps, use_mlp=use_mlp, gcl_layers=gcl_layers,
                                feat_time_dim=feat_t_dim, optimize_embedding=optimize_embedding,
                                agg_norm_factor=agg_norm_factor,
                                distance_sin_emb=distance_sin_emb, egnn_attn=egnn_attn, n_knn=n_knn, device=device,
                                add_bond_model=add_bond,
                                opt_time_emb=opt_time_emb).to(device)

    logger.info(f"No. of parameters: {np.sum([p.numel() for p in model_pref.parameters()])}")
    logger.info(
        f"No. of parameters using p.requires_grad: {np.sum([p.numel() for p in model_pref.parameters() if p.requires_grad])}")

    ckp_model = torch.load(config.train.ckp_path, map_location='cuda')
    MSEloss_func = torch.nn.MSELoss(reduction='mean')
    MAEloss_func = torch.nn.L1Loss(reduction='mean')
    if not args.pdb_id:
        args.pdb_id = args.params[1]
    pocket_idx = int(PDB_dict[args.pdb_id.upper()])  # convert index to a 4-character PDB ID
    logger.info(f"loading the data of PDB {args.pdb_id.upper()}")
    pocket_data = torch.load('configs/test_data_list.pt', map_location=device)[pocket_idx]
    protein_pos_val, protein_ele_val, protein_amino_acid_val, protein_is_backbone_val, ligand_pos_val, _ \
        = extract_data(pocket_data, ProteinElement2IndexDict)
    protein_pos_val, protein_ele_val_mean = subtract_mean(protein_pos_val.to(device), verbose=True)
    protein_ele_val = protein_ele_val.to(device)
    protein_amino_acid_val = protein_amino_acid_val.to(device)
    protein_is_backbone_val = protein_is_backbone_val.to(device)
    num_rounds = config.pref.num_rounds
    num_injections_per_round = config.pref.num_injections_per_round
    num_positive_proposals = config.pref.num_positive_samples
    num_negative_proposals = config.pref.num_negative_samples
    num_total_proposals = num_positive_proposals + num_negative_proposals
    num_ligand_atom_per_proposal, coor_dim = config.pref.num_atoms, 3
    ele_emb_dim = bond_emb_dim = model_pref.ligand_emb_dim
    num_bond_edges_per_ligand = num_ligand_atom_per_proposal * (num_ligand_atom_per_proposal - 1)
    bond_edges = model_pref.get_ligand_bond_types_all(num_ligand_atoms=num_ligand_atom_per_proposal)

    # get batch indices for proposals to obtain all gradients of the classifier in one loop
    n_steps = 1  # config.pref.n_steps  # No. of timesteps per preference
    all_bond_edges = get_all_bond_edges(bond_edges, num_total_proposals * n_steps, num_ligand_atom_per_proposal)
    batch_proposals = torch.tensor(range(num_total_proposals * n_steps)).repeat(num_ligand_atom_per_proposal,
                                                                                1).t().flatten()
    batch_bonds = torch.tensor(range(num_total_proposals * n_steps)).repeat(num_bond_edges_per_ligand, 1).t().flatten()
    num_samples_eval = config.pref.num_samples_eval
    logger.info(f"num_samples: {num_samples_eval}, batchsize: {config.pref.sample_batchsize}")
    num_updates_per_injection = config.pref.num_updates_per_injection
    timestep_sampler = TimeStepSampler(num_timesteps=num_timesteps)
    temperature = config.pref.temperature
    best_improvement = 0
    num_spacing_steps_pref = config.pref.num_spacing_steps
    if num_spacing_steps_pref:
        t_pref = np.linspace(0, num_timesteps, num_spacing_steps_pref + 1, dtype=int)[:-1]
    pref_loss_type = config.pref.loss_type
    guidance_scale_start = config.pref.cls_guidance.scale_start
    guidance_scale = config.pref.cls_guidance.scale
    assert guidance_scale_start <= guidance_scale, 'beginning guidance scale is supposed to be no larger than ending scale'
    s = np.linspace(guidance_scale_start, guidance_scale, num_injections_per_round)
    cls_loss_type = config.pref.cls_guidance.loss_type
    raw_probs = np.linspace(0.5, 1,
                            num_injections_per_round)  # new proposals will have larger probabilities to be chosen
    if config.pref.increasing_num_updates:
        update_footprints = np.linspace(num_updates_per_injection, config.pref.max_num_updates,
                                        num_injections_per_round, dtype=np.int32)
    else:
        update_footprints = np.array([num_updates_per_injection] * num_injections_per_round, dtype=np.int32)
    proposal_factor = config.pref.proposal_factor
    num_positive_proposals1 = proposal_factor * num_positive_proposals
    num_negative_proposals1 = proposal_factor * num_negative_proposals
    num_total_proposals1 = proposal_factor * num_total_proposals

    logger.info("loading checkpoint ...")
    model_pref.load_state_dict(copy.deepcopy(ckp_model['model_ema_state_dict']))
    if config.pref.use_ema:
        logger.info(f"We use ema with ema factor {config.pref.use_ema}")
        ema = EMA(beta=config.pref.use_ema)
        model_pref_ema = copy.deepcopy(model_pref)
    else:
        logger.info("We do not use ema.")
    optimizer_pref = torch.optim.AdamW(model_pref.parameters(), lr=init_lr, weight_decay=wd)  # AdamW
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_pref, mode='min', factor=lr_gamma,
                                                           patience=config.train.lr_scheduler.patience,
                                                           threshold=0.0001, min_lr=1.0e-6, eps=1e-09, verbose=True)
    if guidance_scale > 0:
        if config.pref.guide_with_pocket:
            logger.info(f"classifier trained with pocket info")
        classifier = Classifier(in_node_feat=ele_emb_dim, hidden_node_feat=64, out_node_feat=ele_emb_dim,
                                time_emb_feat=8, depth=2, num_timesteps=num_timesteps, device=device,
                                opt_time_emb=False, add_bond=True, use_pocket=config.pref.guide_with_pocket,
                                num_protein_element=len(ProteinElement2IndexDict),
                                num_amino_acid=20, dim_protein_element=16, dim_amino_acid=16, dim_is_backbone=8)
        num_param_cls = np.sum([p.numel() for p in classifier.parameters() if p.requires_grad])
        logger.info(
            f"classifier loss type: {cls_loss_type}. No. of cls parameters using p.requires_grad: {num_param_cls}")
        optimizer_cls = torch.optim.AdamW(classifier.parameters(), lr=init_lr * 1.0, weight_decay=wd)
        scheduler_cls = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_cls, mode='min', factor=lr_gamma,
                                                                   patience=100, threshold=0.0001, min_lr=1.0e-6,
                                                                   eps=1e-09, verbose=True)
        cls_fn = classifier
    else:
        logger.info(f"classifier will not be trained.")
        cls_fn = None
    inject_counter = 0
    show_recent_loss = OnlineAveraging(averaging_range=len_history)  # show average loss of recent N updates
    prepare_pref_data = PreparePrefData4UI(ele_emb_dim=ele_emb_dim,
                                           bond_emb_dim=bond_emb_dim, num_timesteps=num_timesteps,)
    num_lines_anno = 0
    num_proposals_ui = 8
    num_total_positive_annotations = num_total_negative_annotations = 0
    annotation_dir = os.path.join(log_dir, 'annotations')
    os.makedirs(annotation_dir, exist_ok=True)
    annotation_path = os.path.join(annotation_dir, 'annotations.json')
    proposals_dir = os.path.join(f'backend/app01/static/{args.params[0]}/', 'proposals')
    os.makedirs(proposals_dir, exist_ok=True)
    pt_dir = os.path.join(log_dir, 'final_pt_files')
    os.makedirs(pt_dir, exist_ok=True)
    t0 = time.time()
    for num_inj in range(1000):
        proposal_base_dict = {}
        logger.info(f"{num_inj}, generating proposals...")
        proposals = sampling_val(model_pref, pocket_data, pocket_idx, pos_scale, proposals_dir,
                                 logger, device, ProteinElement2IndexDict, num_timesteps=num_timesteps, mode=mode,
                                 num_pockets=pocket_idx, all_val_pockets=False, bond_emb=True,
                                 num_samples=num_samples_eval, cal_vina_score=True,
                                 cal_straightness=False, num_spacing_steps=100, num_atoms=config.pref.num_atoms,
                                 starting_pocket_id=0, t_sampling_strategy='uniform',
                                 batch_size=config.pref.sample_batchsize, protein_pdbqt_file_path='',
                                 protein_pdbqt_dir='configs/test_protein')
        shutil.move(proposals.pt_filename, os.path.join(pt_dir, f'proposal_{num_inj}.pt'))

        if os.path.isfile(os.path.join(log_dir, 'exit.txt')) or time.time() - t0 > 10*60:
            logger.info('Manually exiting...')
            sys.exit()
