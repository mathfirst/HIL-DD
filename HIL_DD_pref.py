import argparse, os, shutil, torch, copy
from datetime import datetime  # we use time as a filename when creating that logfile.
from utils.util_flow import get_logger, load_config, MeanSubtractionNorm, EMA, get_all_bond_edges, \
    TimeStepSampler, OnlineAveraging, classifier_val, CLASSIFIER_LOSS_FUNC_DICT, PREFERENCE_LOSS_FUNC_DICT
from torch_geometric.transforms import Compose
from torch_geometric.loader import DataLoader
from utils.util_data import get_dataset, ProteinElement2IndexDict, MAP_ATOM_TYPE_AROMATIC_TO_INDEX_wo_h, extract_data, \
    PreparePreferenceData, prepare_inputs
from models.models import Model_CoM_free, Classifier
import numpy as np
import utils.transforms as trans
from utils.util_sampling import sample4pocket
import torch.nn.functional as F
from utils.evaluate import add_more_ls_to_result_dict, summarize_res_one_round, summarize_res_all_rounds
import configs.metric_config
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

FOLLOW_BATCH = ('protein_element', 'ligand_element', 'ligand_bond_type',)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default="configs/config_pref.yml")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logdir', type=str, default='./logs_pref_')
    parser.add_argument('--desc', type=str, default='')
    parser.add_argument('--suffix', type=str, default='')
    args = parser.parse_args()
    config = load_config(args.config)
    query = config.pref.metric
    # logging
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    if args.suffix:
        suffix = args.suffix
    else:
        if config.pref.only_cls_guide:
            suffix = '-'.join([config.pref.cls_guidance.loss_type, 'guide', str(config.pref.cls_guidance.scale_start)])
        elif config.pref.cls_guidance.scale_start:
            suffix = '-'.join(
                [config.pref.loss_type, config.pref.cls_guidance.loss_type, str(config.pref.cls_guidance.scale_start)])
        else:
            suffix = '-'.join([config.pref.loss_type, ])
    if config.pref.all_pref_data:
        suffix = '-'.join([suffix, 'N'])
    log_dir = os.path.join(args.logdir + query + '-pocket' + str(config.pref.pocket_idx), current_time + '-' + suffix)
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, "log.txt")
    logger = get_logger('pref', log_filename)
    shutil.copy(args.config, os.path.join(log_dir, 'config-' + current_time + '.yml'))
    shutil.copy('models/models.py', os.path.join(log_dir, 'models.py'))
    shutil.copy('utils/util_flow.py', os.path.join(log_dir, 'util_flow.py'))
    shutil.copy('utils/util_data.py', os.path.join(log_dir, 'util_data.py'))
    shutil.copy('utils/evaluate.py', os.path.join(log_dir, 'evaluate.py'))
    current_file = __file__  # get the name of the currently executing python file
    shutil.copy(current_file,
                os.path.join(log_dir, os.path.basename(current_file).split('.')[0] + '-' + current_time + '.py'))

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
    transform_list = [
        ligand_featurizer,
    ]
    if config.data.transform.random_rot:
        logger.info("apply random rotation")
        transform_list.append(trans.RandomRotation())
    transform = Compose(transform_list)
    subtract_mean = MeanSubtractionNorm()  # This is used to center positions.
    dataset, subsets = get_dataset(
        config=config.data,
        transform=transform,
    )

    train_set, val_set = subsets['train'], subsets['test']
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
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                            follow_batch=FOLLOW_BATCH, exclude_keys=collate_exclude_keys, pin_memory=True)
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
                                add_bond_model=add_bond, opt_time_emb=opt_time_emb).to(device)

    logger.info(f"No. of parameters: {np.sum([p.numel() for p in model_pref.parameters()])}")
    logger.info(
        f"No. of parameters using p.requires_grad: {np.sum([p.numel() for p in model_pref.parameters() if p.requires_grad])}")

    ckp_model = torch.load(config.train.ckp_path, map_location='cuda')
    MSEloss_func = torch.nn.MSELoss(reduction='mean')
    MAEloss_func = torch.nn.L1Loss(reduction='mean')
    pocket_idx = config.pref.pocket_idx
    batch = val_set[pocket_idx]
    logger.info(f"loading the data of Pocket {pocket_idx}")
    protein_pos_val, protein_ele_val, protein_amino_acid_val, protein_is_backbone_val, ligand_pos_val, _ \
        = extract_data(batch, ProteinElement2IndexDict)
    protein_pos_val, protein_ele_val_mean = subtract_mean(protein_pos_val.to(device), verbose=True)
    protein_ele_val = protein_ele_val.to(device)
    protein_amino_acid_val = protein_amino_acid_val.to(device)
    protein_is_backbone_val = protein_is_backbone_val.to(device)
    num_rounds = config.pref.num_rounds
    num_injections_per_round = config.pref.num_injections_per_round
    num_positive_proposals = config.pref.num_positive_samples
    num_negative_proposals = config.pref.num_negative_samples
    num_total_proposals = num_positive_proposals + num_negative_proposals
    num_ligand_atom_per_proposal, coor_dim = ligand_pos_val.shape
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
    num_updates_per_injection = config.pref.num_updates_per_injection
    timestep_sampler = TimeStepSampler(num_timesteps=num_timesteps)
    temperature = config.pref.temperature
    # pairs_dir = './tmp/samples_pocket' + str(pocket_idx) + '_proposals'
    pairs_dir = config.pref.proposal_dir
    query_base_dict = configs.metric_config.METRIC_BASE_DICT[pocket_idx]
    result_all_rounds = {k: [] for k in query_base_dict.keys()}
    # query_base = query_base_dict[query]
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
    ckps_ls = []
    for file in [x for x in os.listdir(pairs_dir) if '.pt' in x]:  # and '03-07' in x]:
        logger.info(f"getting filename {file}")
        ckps_ls.append(os.path.join(pairs_dir, file))
    if config.pref.all_pref_data:
        logger.info("preparing all pref data...")
        prepare_pref_data = PreparePreferenceData(ckps_ls=ckps_ls, all_pref_data=config.pref.all_pref_data,
                                                  n_atoms_per_mol=num_ligand_atom_per_proposal,
                                                  n_proposal_factor=proposal_factor, n_steps=1,
                                                  n_positive_samples=num_positive_proposals,
                                                  n_negative_samples=num_negative_proposals,
                                                  variance_trials=config.pref.variance_trials, query=query,
                                                  temperature=temperature, ele_emb_dim=ele_emb_dim,
                                                  bond_emb_dim=bond_emb_dim, num_timesteps=num_timesteps,
                                                  num_injections_per_round=num_injections_per_round,
                                                  num_bond_edges_per_ligand=num_bond_edges_per_ligand)
    for round in range(num_rounds):
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
        else:
            logger.info(f"classifier will not be trained.")
            cls_fn = None
        inject_counter = 0
        show_recent_loss = OnlineAveraging(averaging_range=len_history)  # show average loss of recent N updates
        improvement_this_round = {k: [] for k in query_base_dict.keys()}
        best_improvement_current_round = 0
        if not config.pref.all_pref_data:
            prepare_pref_data = PreparePreferenceData(ckps_ls=ckps_ls, all_pref_data=config.pref.all_pref_data,
                                                      n_atoms_per_mol=num_ligand_atom_per_proposal,
                                                      n_proposal_factor=proposal_factor, n_steps=1,
                                                      n_positive_samples=num_positive_proposals,
                                                      n_negative_samples=num_negative_proposals,
                                                      variance_trials=config.pref.variance_trials, query=query,
                                                      temperature=temperature, ele_emb_dim=ele_emb_dim,
                                                      bond_emb_dim=bond_emb_dim, num_timesteps=num_timesteps,
                                                      num_injections_per_round=num_injections_per_round,
                                                      num_bond_edges_per_ligand=num_bond_edges_per_ligand)
        while True:
            if not config.pref.all_pref_data:
                prepare_pref_data.sample_preference_data(inject_counter=inject_counter)
            model_pref.train()
            num_actual_updates = 0
            show_recent_loss_cls = OnlineAveraging(averaging_range=len_history)
            for step, batch in enumerate(train_loader):
                t = None
                target_pos_pair, target_ele_emb_pair, target_bond_emb_pair, labels = prepare_pref_data.sample_pref_data_minibatch(device)
                if guidance_scale > 0:# or (num_positive_proposals * num_negative_proposals != 0):
                    classifier.train()
                    for _ in range(1):
                        Xt_pos_pair, Xt_ele_emb_pair, Xt_bond_emb_pair, t_int, t_bond = prepare_pref_data.get_xt(t, timestep_sampler, config.pref.noise)
                        logits_cls = classifier(Xt_pos_pair.to(device), Xt_ele_emb_pair.to(device), t_int.to(device),
                                                num_atoms_per_ligand=num_ligand_atom_per_proposal,
                                                batch_ligand=batch_proposals.to(device),
                                                bond_features=Xt_bond_emb_pair.to(device), t_bond=t_bond.to(device),
                                                protein_positions=protein_pos_val,
                                                protein_ele=protein_ele_val,
                                                protein_amino_acid=protein_amino_acid_val,
                                                protein_is_backbone=protein_is_backbone_val,
                                                num_proposals=num_total_proposals, n_steps_per_iter=1)
                        loss_cls = CLASSIFIER_LOSS_FUNC_DICT[cls_loss_type](logits=logits_cls, labels=labels,
                                                                            n_steps=1,
                                                                            num_positive_proposals=num_positive_proposals)
                        optimizer_cls.zero_grad()
                        loss_cls.backward()
                        optimizer_cls.step()
                        avg_cls_loss = show_recent_loss_cls(loss_cls=loss_cls.item())['loss_cls']
                    if config.pref.only_cls_guide:
                        logger.info(f"logits: {['%.2f' % logit for logit in logits_cls.detach()]}, "
                                    f"cls_loss: {loss_cls.item():.2f}, avg_loss: {avg_cls_loss:.2f}")
                        if step < update_footprints[inject_counter] - 1:
                            continue
                        else:
                            break
                else:
                    Xt_pos_pair, Xt_ele_emb_pair, Xt_bond_emb_pair, t_int, t_bond = prepare_pref_data.get_xt(t,
                                                                                                             timestep_sampler,
                                                                                                             config.pref.noise)
                    loss_cls = torch.tensor(0)
                pred_pos_pair, pred_element_embedding_pair, pred_bond_embedding_pair = \
                    model_pref(protein_pos_val, protein_ele_val,
                               protein_amino_acid_val,
                               protein_is_backbone_val, Xt_pos_pair.to(device), Xt_ele_emb_pair.to(device),
                               t_int.to(device), batch_protein=None, batch_ligand=None,
                               bond_indices=all_bond_edges.to(device),
                               bond_features=Xt_bond_emb_pair.to(device),
                               num_proposals=num_total_proposals,
                               num_atoms_per_ligand=num_ligand_atom_per_proposal,
                               t_bond=t_bond.to(device),
                               n_steps_per_iter=n_steps)
                pref_logit_pos = (target_pos_pair.to(device) - pred_pos_pair).view(num_total_proposals * n_steps,
                                                                                   -1).abs().mean(
                    dim=-1)  # [num_prop, num_ligand_atom_per_proposal] -> [num_prop,]
                pref_logit_pos_list = [f'{pref_logit:2.2f}' for pref_logit in pref_logit_pos.detach().data]

                pref_logit_ele_emb = (target_ele_emb_pair.to(device) - pred_element_embedding_pair).view(
                    num_total_proposals * n_steps, -1).abs().mean(dim=-1)  # [num_prop, num_ele_emb_dim] -> [num_prop,]
                pref_logit_ele_emb_list = [f'{pref_logit:2.2f}' for pref_logit in pref_logit_ele_emb.detach().data]

                pref_logit_bond_emb = (target_bond_emb_pair.to(device) - pred_bond_embedding_pair).view(
                    num_total_proposals * n_steps, -1).abs().mean(dim=-1)  # [num_prop, num_bond_emb_dim] -> [num_prop,]
                pref_logit_bond_emb_list = [f'{pref_logit:2.2f}' for pref_logit in pref_logit_bond_emb.detach().data]
                if step % 100 == 0:
                    logger.info(f"pos logit list: {pref_logit_pos_list}")
                    logger.info(f"ele_emb logit list: {pref_logit_ele_emb_list}")
                    logger.info(f"bond_emb logit list: {pref_logit_bond_emb_list}")
                if config.pref.pref_loss_type == 'combine':
                    pref_logits = pref_logit_pos + 1.0 * pref_logit_ele_emb + 1.0 * pref_logit_bond_emb
                    if num_positive_proposals * num_negative_proposals != 0:
                        all_logits = pref_logits.detach().clone().reshape(n_steps, -1)
                        positive_logits = torch.max(all_logits[:, :num_positive_proposals].flatten())
                        negative_logits = torch.min(all_logits[:, num_positive_proposals:].flatten())
                        if positive_logits < 0.8 * negative_logits:
                            logger.info(f"skip iteration {step}. positive_logits: {positive_logits:.2f}, "
                                        f"negative_logits: {negative_logits:.2f}")
                            if step >= update_footprints[inject_counter] - 1:
                                break  # num_update_steps > max_num_update_steps
                            else:  # when the model distinguishes good samples and bad samples well, we skip this iteration.
                                continue
                    loss_pref = PREFERENCE_LOSS_FUNC_DICT[pref_loss_type](logits=pref_logits, n_steps=n_steps,
                                                                          num_positive_proposals=num_positive_proposals,
                                                                          labels=labels)
                    if loss_pref.item() == 0:
                        logger.info("pref loss is 0 and this skip will be skipped")
                        continue
                    num_actual_updates += 1
                else:  # handle pos, ele, bond losses separately
                    pref_loss_pos = F.cross_entropy(-pref_logit_pos.reshape(1, num_total_proposals),
                                                    labels.reshape(1, num_total_proposals).to(device))
                    pref_loss_ele_emb = F.cross_entropy(-pref_logit_ele_emb.reshape(1, num_total_proposals),
                                                        labels.reshape(1, num_total_proposals).to(device))
                    pref_loss_bond_emb = F.cross_entropy(-pref_logit_bond_emb.reshape(1, num_total_proposals),
                                                         labels.reshape(1, num_total_proposals).to(device))
                    loss_pref = pref_loss_pos + pref_loss_ele_emb + pref_loss_bond_emb
                    if step % 100 == 0:
                        logger.info(f"pref_loss: {loss_pref.item():.3f}, pref_loss_pos: {pref_loss_pos.item():.3f}, "
                                    f"pref_loss_ele_emb: {pref_loss_ele_emb.item():.3f}, "
                                    f"pref_loss_bond_emb: {pref_loss_bond_emb.item():.3f}")
                optimizer_pref.zero_grad()
                loss_pref.backward()
                g_grad = copy.deepcopy([p.grad.clone() for p in model_pref.parameters() if p.requires_grad])
                # calculations for the regular loss
                protein_pos, protein_ele, protein_amino_acid, protein_is_backbone, \
                Xt_pos, Xt_element_embedding, t_regular, protein_batch, ligand_batch, bond_edges_regular, \
                Xt_bond_embedding, target_pos, target_element_embedding, target_bond_features, \
                X1_pos, t_bond_regular = \
                    prepare_inputs(batch, model_pref, ProteinElement2IndexDict, num_timesteps,
                                   pos_scale, add_bond, timestep_sampler, n_steps_per_iter=n_steps, t=t, device=device)
                optimizer_pref.zero_grad()
                pred_pos, pred_element_embedding, pred_bond_embedding = \
                    model_pref(protein_pos.to(device), protein_ele.to(device), protein_amino_acid.to(device),
                               protein_is_backbone.to(device), Xt_pos.to(device), Xt_element_embedding.to(device),
                               t_regular.to(device), batch_protein=protein_batch, batch_ligand=ligand_batch,
                               bond_indices=bond_edges_regular.to(device),
                               bond_features=Xt_bond_embedding.to(device),
                               t_bond=t_bond_regular.to(device),
                               n_steps_per_iter=n_steps, num_proposals=1,
                               num_atoms_per_ligand=ligand_batch.shape[0])
                loss_pos = MAEloss_func(pred_pos, target_pos.to(device)) if loss_type == 'MAE' else MSEloss_func(
                    pred_pos, target_pos.to(device))
                loss_ele_embedding = MAEloss_func(pred_element_embedding,
                                                  target_element_embedding.to(device)) if loss_type == 'MAE' \
                    else MSEloss_func(pred_element_embedding, target_element_embedding.to(device))
                if add_bond:
                    loss_bond_embedding = MAEloss_func(pred_bond_embedding,
                                                       target_bond_features.to(device)) if loss_type == 'MAE' \
                        else MSEloss_func(pred_bond_embedding, target_bond_features.to(device))
                else:
                    loss_bond_embedding = torch.tensor(0)

                loss_regular = loss_pos + loss_ele_embedding + loss_bond_embedding
                loss_regular.backward()  # retain_graph=True
                f_grad = copy.deepcopy([p.grad.clone() for p in model_pref.parameters() if p.requires_grad])
                # dynamic coefficient of regularizer
                g_grad_norm = np.sum([torch.sum(g_grad_i.cpu() ** 2) for g_grad_i in g_grad])
                phi = np.minimum(loss_pref.detach().clone().cpu(), g_grad_norm.item())
                subtractor = np.sum([torch.sum(aa.cpu() * bb.cpu()) for aa, bb in zip(f_grad, g_grad)])
                coef = np.maximum((phi - subtractor) / g_grad_norm, 0)
                params = [p for p in model_pref.parameters() if p.requires_grad]
                for i, (f_grad_i, g_grad_i, p) in enumerate(zip(f_grad, g_grad, params)):
                    p.grad = f_grad_i + coef * g_grad_i

                optimizer_pref.step()
                if config.pref.use_ema:
                    ema.update_model_average(model_pref_ema, model_pref)
                avg_result = show_recent_loss(loss=loss_regular.item() + loss_pref.item(),
                                              loss_train=loss_regular.item(),
                                              loss_pref=loss_pref.item(), loss_cls=loss_cls.item())
                if step % 20 == 0:
                    scheduler.step(avg_result['loss'])
                    if guidance_scale > 0:
                        scheduler_cls.step(avg_result['loss_cls'])
                    orig_grad_norm = torch.nn.utils.clip_grad_norm_(model_pref.parameters(),
                                                                    config.train.max_grad_norm)  # avoid gradient explosion
                    logger.info(
                        f"round {round}, inj {inject_counter}, step {step}, regular_loss: {loss_regular.item():.2f}, "
                        f"pref_loss: {loss_pref.item():.2f}, coef: {coef:.2f}, lr: {optimizer_pref.param_groups[0]['lr']:.6f}, "
                        f"grad: {orig_grad_norm.item():.2f}, {[f'{k}:{v:.2f}' for k, v in avg_result.items()]}, pref {query}, "
                        f"loss_cls: {loss_cls.item():.2f}")
                if step >= update_footprints[inject_counter] - 1:
                    break

            logger.info(f"number of actual updates for round {round}, injection {inject_counter}: {num_actual_updates}")
            inject_counter += 1
            if not config.pref.use_ema:
                model_pref_ema = copy.deepcopy(model_pref)
            model_pref_ema.eval()
            if guidance_scale > 0:# or (num_positive_proposals * num_negative_proposals != 0):
                classifier_val_loss = classifier_val(num_ligand_atom_per_proposal=num_ligand_atom_per_proposal,
                                                     cls_loss_type=cls_loss_type, model=classifier,
                                                     pairs_path=pairs_dir, device=device,
                                                     num_positive_samples=num_positive_proposals, query=query,
                                                     num_negative_samples=num_negative_proposals,
                                                     variance_trials=config.pref.variance_trials,
                                                     time_sampler=timestep_sampler, temperature=temperature,
                                                     ckps_ls=[ckps_ls[-1]], num_total_proposals=num_total_proposals,
                                                     num_iterations=config.pref.cls_guidance.num_val_iterations,
                                                     protein_positions=protein_pos_val, protein_ele=protein_ele_val,
                                                     protein_amino_acid=protein_amino_acid_val,
                                                     protein_is_backbone=protein_is_backbone_val,
                                                     num_proposals=num_total_proposals,
                                                     )
                logger.info(
                    f"cls_lr: {optimizer_cls.param_groups[0]['lr']:.6f}, cls_loss is {classifier_val_loss:.2f}")
                logger.info(f"the classifier guides the generating process with scale of {s[inject_counter - 1]:.4f}")
                cls_fn = classifier
            else:
                cls_fn = None

            calculator_pref = sample4pocket(model_pref_ema, val_loader, pos_scale, sample_result_dir, logger,
                                            device, ProteinElement2IndexDict, num_timesteps=num_timesteps, mode=mode,
                                            num_pockets=pocket_idx, all_val_pockets=False, bond_emb=True,
                                            num_samples=num_samples_eval, cal_vina_score=config.pref.cal_vina_score,
                                            cal_straightness=False, num_spacing_steps=config.pref.num_spacing_steps,
                                            starting_pocket_id=0, guidance_type=cls_loss_type,
                                            t_sampling_strategy='uniform', cls_fn=cls_fn, s=s[inject_counter - 1],
                                            batchsize=config.pref.sample_batchsize, protein_pdbqt_file_path='')
            add_more_ls_to_result_dict(calculator_pref.result_dict)
            summarize_res_one_round(calculator_pref.result_dict, query_base_dict, improvement_this_round,
                                    logger=logger.info)
            if query == 'mixed':
                if improvement_this_round['vina_score'][-1] > 0.15:
                    logger.info(f"saving model for mixed preferences")
                    torch.save({'model_ema_state_dict': model_pref_ema.state_dict()}, os.path.join(log_dir, f'model_{query}_{round}_{inject_counter}.pt'))
                    if cls_fn is not None:
                        torch.save({'cls_state_dict': cls_fn.state_dict()}, os.path.join(log_dir, f'cls_model_{query}_{round}_{inject_counter}.pt'))
            elif improvement_this_round[query][-1] > best_improvement:
                logger.info('saving the best model of all rounds')
                best_improvement = improvement_this_round[query][-1]
                torch.save({'round': round + 1, 'inj': inject_counter, 'model_pref_ema': model_pref_ema.state_dict()},
                           os.path.join(log_dir, f'model_{query}.pt'))

            if inject_counter >= num_injections_per_round:
                best_improvement_current_round = 0
                break

        summarize_res_all_rounds(result_all_rounds, improvement_this_round,
                                 os.path.join(sample_result_dir, query + '_results.pt'), logger=logger.info)
