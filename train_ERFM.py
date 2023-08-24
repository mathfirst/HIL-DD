import argparse, os, shutil, torch, sys, copy
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from datetime import datetime # we use time as a filename when creating that logfile.
from utils.util_flow import get_logger, load_config, MeanSubtractionNorm, EMA, OnlineAveraging, TimeStepSampler, EarlyStopping
from torch_geometric.transforms import Compose
from torch_geometric.loader import DataLoader
from utils.util_data import get_dataset, ProteinElement2IndexDict, MAP_ATOM_TYPE_AROMATIC_TO_INDEX_wo_h, prepare_inputs
from models.models import Model_CoM_free
import numpy as np
import utils.transforms as trans
from utils.util_sampling import sample4val_eval

FOLLOW_BATCH = ('protein_element', 'ligand_element', 'ligand_bond_type',)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default="E:\pytorch-related\TargetDiff2RectifiedFlow\molopt\configs\config_train_reflow.yml")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logdir', type=str, default='./logs_rectified_flow')
    parser.add_argument('--desc', type=str, default='')
    parser.add_argument('--suffix', type=str, default='')
    args = parser.parse_args()

    # logging
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_dir = os.path.join(args.logdir, current_time + args.suffix)
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, "log.txt")
    logger = get_logger('train', log_filename)
    shutil.copy(args.config, os.path.join(log_dir, 'config-' + current_time + '.yml'))
    current_file = __file__  # get the name of the currently executing python file
    shutil.copy(current_file, os.path.join(log_dir, os.path.basename(current_file).split('.')[0] + '.py'))

    device = args.device if torch.cuda.is_available() else "cpu"
    logger.info(f"device: {device}")
    config = load_config(args.config)
    hidden_dim = int(config.model.hidden_dim)  # EGNN hidden feature dim
    num_timesteps = config.model.num_timesteps  # time steps for the rectified flow model, default: 1000
    depth = config.model.depth  # this refers to how many layers EGNN will have
    use_mlp = config.model.use_mlp  # whether or not apply mlp to the output of EGNN
    gcl_layers = config.model.gcl_layers  # num of gcl layers
    feat_t_dim = config.model.feat_t_dim  # feature time embedding dim
    egnn_attn = config.model.egnn_attn  # use attn or not, this is a setting from egnn module
    agg_norm_factor = config.model.agg_norm_factor  # (default: 5)
    add_bond = config.train.bond
    protein_ele_dim = config.model.protein_ele_dim
    protein_aa_dim = config.model.protein_aa_dim
    protein_is_backbone_dim = config.model.protein_is_backbone_dim
    logger.info(f"hidden_dim: {hidden_dim}, feat_t_dim: {feat_t_dim}, protein_is_backbone_dim: {protein_is_backbone_dim}\n"
                f"depth: {depth}, gcl_layers: {gcl_layers}, egnn_attn: {egnn_attn}, agg_norm_factor: {agg_norm_factor}\n"
                f"use_mlp: {use_mlp}, add_bond: {add_bond}, num_timesteps: {num_timesteps}, protein_aa_dim: {protein_aa_dim},"
                f"protein_ele_dim: {protein_ele_dim}")

    pos_scale = config.data.pos_scale  # training loss stability
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
    lr_gamma = config.train.lr_scheduler.gamma  # lr decay factor
    wd = config.train.optimizer.weight_decay
    logger.info(f"init_lr: {init_lr}, lr_gamma: {lr_gamma}, lr_warmup_steps: {lr_warmup_steps}, wd: {wd}\n"
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
    subtract_mean = MeanSubtractionNorm()   # This is used to center positions.
    # loading data
    dataset, subsets = get_dataset(
        config=config.data,
        transform=transform,
    )
    train_set, val_set = subsets['train'], subsets['test']
    # test_data_list = []
    # for data in val_set:
    #     one_pair = {}
    #     for k, v in data.items():
    #         one_pair[k] = v
    #     test_data_list.append(one_pair)
    # torch.save(test_data_list, 'test_data_list.pt')
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
    val_loader = DataLoader(val_set, 1, shuffle=False, follow_batch=FOLLOW_BATCH,
                            exclude_keys=collate_exclude_keys, pin_memory=True)
    del dataset, subsets

    sample_result_dir = os.path.join(log_dir, 'sample-results')
    ema_sample_result_dir = os.path.join(log_dir, 'ema_sample-results')
    os.makedirs(sample_result_dir, exist_ok=True)
    os.makedirs(ema_sample_result_dir, exist_ok=True)

    logger.info("building model...")
    model = Model_CoM_free(num_protein_element=len(ProteinElement2IndexDict), num_amino_acid=20,
                           dim_protein_element=protein_ele_dim, dim_amino_acid=protein_aa_dim,
                           dim_is_backbone=protein_is_backbone_dim,
                           num_ligand_element=len(MAP_ATOM_TYPE_AROMATIC_TO_INDEX_wo_h), depth=depth, # number of EGNN layers
                           hidden_dim=hidden_dim, n_steps=num_timesteps, use_mlp=use_mlp, gcl_layers=gcl_layers,
                           feat_time_dim=feat_t_dim, optimize_embedding=optimize_embedding,
                           agg_norm_factor=agg_norm_factor,
                           distance_sin_emb=distance_sin_emb, egnn_attn=egnn_attn, n_knn=n_knn, device=device,
                           add_bond_model=add_bond, opt_time_emb=opt_time_emb).to(device)

    logger.info(f"No. of parameters: {np.sum([p.numel() for p in model.parameters()])}")
    logger.info(
        f"No. of parameters using p.requires_grad: {np.sum([p.numel() for p in model.parameters() if p.requires_grad])}")

    optimizer = torch.optim.AdamW([{'params': model.parameters()},
                                  ], lr=init_lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_gamma, patience=config.train.lr_scheduler.patience,
                                                           threshold=0.0001, min_lr=1.0e-7, eps=1e-09, verbose=True)
    if config.train.checkpoint:
        logger.info("loading checkpoint " + str(config.train.ckp_path))
        ckp = torch.load(config.train.ckp_path, map_location=device)
        model.load_state_dict(ckp['model_ema'])
    ema = EMA(beta=ema_decay)
    model_ema = copy.deepcopy(model)
    epochs = int(config.train.epochs)
    model_ckp = os.path.join(log_dir, os.path.basename(current_file).split('.')[0] + '-' + current_time + ".pt")
    MSEloss_func = torch.nn.MSELoss(reduction='mean')
    MAEloss_func = torch.nn.L1Loss(reduction='mean')
    timestep_sampler = TimeStepSampler(num_timesteps, sampling_method=time_sampling_method)
    timestep_sampler_val = TimeStepSampler(num_timesteps, sampling_method='uniform')
    earlystop = EarlyStopping(patience=50, mode='min')
    n_steps_per_iter_val = 20
    for epoch in range(epochs):
        show_recent_loss = OnlineAveraging(averaging_range=len_history)  # show average loss in this epoch
        for step, batch in enumerate(train_loader):
            model.train()
            # Learning rate warmup
            num_current_steps = epoch * len(train_set) + step
            if num_current_steps < lr_warmup_steps:
                optimizer.param_groups[0]['lr'] = (num_current_steps + 1) * init_lr / lr_warmup_steps

            protein_pos, protein_ele, protein_amino_acid, protein_is_backbone, \
            Xt_pos, Xt_element_embedding, t, protein_batch, ligand_batch, bond_edges, \
            Xt_bond_embedding, target_pos, target_element_embedding, target_bond_features, X1_pos, t_bond = \
            prepare_inputs(batch, model, ProteinElement2IndexDict, num_timesteps, pos_scale, add_bond, timestep_sampler,
                           n_steps_per_iter=n_steps_per_iter, device=device)

            pred_pos, pred_element_embedding, pred_bond_embedding = \
                model(protein_pos, protein_ele, protein_amino_acid,
                      protein_is_backbone, Xt_pos, Xt_element_embedding,
                      t, batch_protein=protein_batch, batch_ligand=ligand_batch,
                      bond_indices=bond_edges, bond_features=Xt_bond_embedding,
                      num_atoms_per_ligand=ligand_batch.shape[0], num_proposals=1, t_bond=t_bond,
                      n_steps_per_iter=n_steps_per_iter)

            loss_pos = MAEloss_func(pred_pos, target_pos) if loss_type == 'MAE' else MSEloss_func(pred_pos, target_pos)
            loss_ele_embedding = MAEloss_func(pred_element_embedding, target_element_embedding) if loss_type == 'MAE' \
                else MSEloss_func(pred_element_embedding, target_element_embedding)
            if add_bond:
                loss_bond_embedding = MAEloss_func(pred_bond_embedding, target_bond_features) if loss_type == 'MAE' \
                    else MSEloss_func(pred_bond_embedding, target_bond_features)
            else:
                loss_bond_embedding = torch.tensor(0)

            optimizer.zero_grad()
            loss = loss_pos + loss_ele_embedding + loss_bond_embedding
            loss.backward()
            orig_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                            config.train.max_grad_norm)  # avoid gradient explosion
            optimizer.step()
            ema.update_model_average(model_ema, model)

            timestep_sampler.update_weights(t, loss.item())
            avg_result = show_recent_loss(loss=loss.item(), pos_loss=loss_pos.item(),
                                          ele_loss=loss_ele_embedding.item(), bond_loss=loss_bond_embedding.item())
            if step % print_interval == 0:
                avg_loss_str = [f'{k}:{v:.2f}' for k, v in avg_result.items()]
                logger.info(
                    'Ep %d/%d, step %d|avg: %s|Loss %.2f (pos %.3f|feat %.3f|bond %.3f)|grad_norm: %.1f|t:%s' % (
                        epoch + 1, epochs, step + 1, avg_loss_str, loss.item(), loss_pos.item(),
                        loss_ele_embedding.item(), loss_bond_embedding.item(), orig_grad_norm.item(),
                        timestep_sampler.last_t)
                )

                if step % (20 * print_interval) == 0:
                    logger.info(args.desc + ' ' + current_time + ' ' + current_file)
                    logger.info(
                        'Lr: %.7f|GradNorm: %.1f|hid_dim: %d|n_tsteps: %d|gcl_layers: %d|f_t_dim: %d|'
                        'p_scale: %.1f, bs: %d, rot: %s, bond: %s, agg_factor: %d' % (
                            optimizer.param_groups[0]['lr'], config.train.max_grad_norm,
                            hidden_dim, num_timesteps, gcl_layers, feat_t_dim, pos_scale, bs,
                            config.data.transform.random_rot, add_bond, agg_norm_factor)
                    )

            if (step + 1) % 20000 == 0 or (step + 1) >= len(train_loader):
                model.eval()
                logger.info("saving ema model")
                torch.save({'model_ema': model_ema.state_dict()}, os.path.join(log_dir, "model_ema.pt"))
                with torch.no_grad():
                    show_recent_val_loss = OnlineAveraging(averaging_range=100)
                    show_recent_val_loss_ema = OnlineAveraging(averaging_range=100)
                    show_recent_val_loss_mae = OnlineAveraging(averaging_range=100)
                    show_recent_val_loss_mae_ema = OnlineAveraging(averaging_range=100)
                    for val_step, batch in enumerate(val_loader):
                        protein_pos, protein_ele, protein_amino_acid, protein_is_backbone, \
                        Xt_pos, Xt_element_embedding, t, protein_batch, ligand_batch, bond_edges, \
                        Xt_bond_embedding, target_pos, target_element_embedding, target_bond_features, X1_pos, t_bond =\
                        prepare_inputs(batch, model, ProteinElement2IndexDict, num_timesteps, pos_scale, add_bond,
                                       timestep_sampler_val, device=device, n_steps_per_iter=n_steps_per_iter_val)
                        pred_pos, pred_element_embedding, pred_bond_embedding = \
                            model(protein_pos, protein_ele, protein_amino_acid,
                                  protein_is_backbone, Xt_pos, Xt_element_embedding,
                                  t, batch_protein=protein_batch, n_steps_per_iter=n_steps_per_iter_val,
                                  batch_ligand=ligand_batch, bond_indices=bond_edges,
                                  bond_features=Xt_bond_embedding, num_atoms_per_ligand=ligand_batch.shape[0],
                                  t_bond=t_bond)
                        pred_pos_ema, pred_element_embedding_ema, pred_bond_embedding_ema = \
                            model_ema(protein_pos, protein_ele, protein_amino_acid,
                                      protein_is_backbone, Xt_pos, Xt_element_embedding,
                                      t, batch_protein=protein_batch, n_steps_per_iter=n_steps_per_iter_val,
                                      batch_ligand=ligand_batch, bond_indices=bond_edges,
                                      bond_features=Xt_bond_embedding, num_atoms_per_ligand=ligand_batch.shape[0],
                                      t_bond=t_bond)

                        loss_pos = (target_pos - pred_pos).pow(2).mean()
                        loss_pos_ema = (target_pos - pred_pos_ema).pow(2).mean()
                        loss_ele_embedding = (target_element_embedding - pred_element_embedding).pow(2).mean()
                        loss_ele_embedding_ema = (target_element_embedding - pred_element_embedding_ema).pow(2).mean()
                        loss_pos_mae = MAEloss_func(pred_pos, target_pos)
                        loss_pos_mae_ema = MAEloss_func(pred_pos_ema, target_pos)
                        loss_ele_emb_mae = MAEloss_func(pred_element_embedding, target_element_embedding)
                        loss_ele_emb_mae_ema = MAEloss_func(pred_element_embedding_ema, target_element_embedding)
                        if add_bond:
                            loss_bond_embedding = (target_bond_features - pred_bond_embedding).pow(2).mean()
                            loss_bond_embedding_ema = (target_bond_features - pred_bond_embedding_ema).pow(2).mean()
                            loss_bond_emb_mae = MAEloss_func(pred_bond_embedding, target_bond_features)
                            loss_bond_emb_mae_ema = MAEloss_func(pred_bond_embedding_ema, target_bond_features)
                        else:
                            loss_bond_embedding = loss_bond_embedding_ema = loss_bond_emb_mae = loss_bond_emb_mae_ema = torch.tensor(0)

                        loss = loss_pos + loss_ele_embedding + loss_bond_embedding
                        loss_ema = loss_pos_ema + loss_ele_embedding_ema + loss_bond_embedding_ema
                        avg_result_val = show_recent_val_loss(loss=loss.item(), pos_loss=loss_pos.item(),
                                                              ele_loss=loss_ele_embedding.item(), bond_loss=loss_bond_embedding.item())
                        avg_result_val_ema = show_recent_val_loss_ema(loss=loss_ema.item(), pos_loss=loss_pos_ema.item(),
                                                                      ele_loss=loss_ele_embedding_ema.item(),
                                                                      bond_loss=loss_bond_embedding_ema.item())
                        loss_mae = loss_pos_mae + loss_ele_emb_mae + loss_bond_emb_mae
                        loss_mae_ema = loss_pos_mae_ema + loss_ele_emb_mae_ema + loss_bond_emb_mae_ema
                        avg_result_val_mae = show_recent_val_loss_mae(loss=loss_mae.item(), pos_loss=loss_pos_mae.item(),
                                                                      ele_loss=loss_ele_emb_mae.item(), bond_loss=loss_bond_emb_mae.item())
                        avg_result_val_mae_ema = show_recent_val_loss_mae_ema(loss=loss_mae_ema.item(), pos_loss=loss_pos_mae_ema.item(),
                                                                              ele_loss=loss_ele_emb_mae_ema.item(), bond_loss=loss_bond_emb_mae_ema.item())
                    logger.info(f"----val mse loss: {[f'{k}:{v:.2f}' for k, v in avg_result_val.items()]}")
                    logger.info(f"ema-val mse loss: {[f'{k}:{v:.2f}' for k, v in avg_result_val_ema.items()]}")
                    logger.info(f"----val mae loss: {[f'{k}:{v:.2f}' for k, v in avg_result_val_mae.items()]}")
                    logger.info(f"ema-val mae loss: {[f'{k}:{v:.2f}' for k, v in avg_result_val_mae_ema.items()]}")
                    metric = avg_result_val_mae_ema['loss'] if loss_type == 'MAE' else avg_result_val_ema['loss']
                    scheduler.step(metric)
                    if earlystop(path=model_ckp, logger=logger.info, metric=metric, model=model.state_dict(), model_ema=model_ema.state_dict()):
                        sys.exit()

        sample4val_eval(model_ema, val_loader, pos_scale, ema_sample_result_dir, logger=logger, device=device,
                        num_pockets=num_pockets, num_timesteps=num_timesteps, mode=mode,
                        ProteinElement2IndexDict=ProteinElement2IndexDict)
        sample4val_eval(model, val_loader, pos_scale, sample_result_dir, logger=logger, device=device,
                        num_pockets=num_pockets, num_timesteps=num_timesteps, mode=mode,
                        ProteinElement2IndexDict=ProteinElement2IndexDict)
        if avg_result_val_mae_ema.items() < avg_result_val_mae.items():
            logger.info("update model's parameters with parameters of EMA model")
            model.load_state_dict(model_ema.state_dict())