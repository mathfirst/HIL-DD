import torch, copy, time
from utils_toy import RectifiedFlow, MLP, make_data
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

'''
Note that the weight of the preference loss, i.e. the coefficient of the regularization term, is dynamically determined 
by the technique proposed by the following paper:
Gong, Chengyue and Liu, Xingchao and Liu, Qiang. Bi-objective Trade-off with Dynamic Barrier Gradient Descent, NeurIPS, 2021
'''

def plot_func(data, num_samples, color, fig_name=None):
    fig, ax = plt.subplots(1, num_samples, figsize=(2 * num_samples, 2))
    for j in range(num_samples):
        ax[j].scatter(*data[j].cpu().numpy().T, color=color, edgecolor="white")
        ax[j].set_axis_off()
    if fig_name is not None:
        print(f"saving {fig_name}")
        plt.savefig(fig_name, format='pdf')
    plt.show()

def test(rectified_flow, k, num_samples=8, plot_interval=10, color="purple", z0=None, fig_name=None):
    rectified_flow.model.eval()
    with torch.no_grad():
        if (k + 1) % plot_interval == 0:
            traj = rectified_flow.sample_ode(num_samples, z0=z0)
            # plot_func(traj[-1], num_samples, color, fig_name)
            fig, ax = plt.subplots(1, num_samples, figsize=(3 * num_samples, 3))
            for i in range(num_samples):
                ax[i].scatter(*traj[-1][i].cpu().numpy().T, color=color, edgecolor="white")
                ax[i].set_axis_off()
            if fig_name is not None:
                print(f'saving {fig_name}')
                plt.savefig(fig_name)
            plt.show()


def rectified_flow_pref(base_flow, rectified_flow, optimizer, target_samples, batchsize, inner_iters,
                        num_samples, num_proposals, device):
    loss_curve, inject_counter = {'regular_loss': [], 'pref_loss': []}, 0
    # target_samples = target_samples.to(device)
    num_steps = rectified_flow.N  # [num_steps]
    for i in range(inner_iters):
        if target_samples is not None:
            indices = torch.randperm(len(target_samples))[:batchsize]
            batch = target_samples[indices].to(device)
        else:
            batch = make_data(num_samples_each_class=batchsize, num_datapoints=base_flow.num_points)[0].to(device)

        z0 = rectified_flow.get_z0(batchsize=num_samples)
        if i == 0:
            test(rectified_flow, 0, num_samples=num_samples, fig_name=f"base.pdf", color='red', z0=z0, plot_interval=1)
            # plot_func(z0, num_samples, 'red', 'base_noise.pdf')

        rectified_flow.model.train()
        optimizer.zero_grad()
        x_t, t, target = rectified_flow.get_train_tuple(batch=batch)
        pred = rectified_flow.model(x_t.detach().clone(), t.detach().clone())
        loss = (target.detach().clone() - pred).view(pred.shape[0], -1).pow(2)  # .sum(dim=-1)
        loss_regular = loss.mean()
        loss_regular.backward()  #retain_graph=True
        f_grad = copy.deepcopy([p.grad.clone() for p in rectified_flow.model.parameters()])
        z0 = rectified_flow.get_z0(batchsize=num_proposals)
        # plot_func(z0, num_proposals, 'black', f'noise_{i}.pdf')
        sample_pair = base_flow.sample_ode(batchsize=num_proposals, z0=z0)[-1]
        plot_func(sample_pair, num_proposals, 'red', f'proposal_{i}.pdf')
        # Since sampling procedure will go over all time steps, adjusting parameters for all time steps will be
        # necessary to make preference learning effective. The downside is that this step can be very memory-consuming.

        t = torch.linspace(0, 1, steps=num_steps + 1)[:-1]  # torch.Size([num_steps]), default: 100
        t = t.reshape(num_steps, 1).repeat(1, num_proposals)  # [num_steps] -> [num_steps, 1] -> [num_steps, num_prop]
        t = t.reshape(-1, 1).unsqueeze(-1).to(
            device)  # [200, 1, 1], [num_steps, num_prop] -> [num_steps*num_prop, 1] -> [num_steps*num_prop, 1, 1]
        target_pair = sample_pair - z0  # [2, N, 2], [num_prop, N, 2]
        # dim of sample_pair: [num_prop, N, 2] -> [num_steps*num_prop, N, 2] (each sample has num_steps Xt's)
        x_t_pair = t * sample_pair.repeat(num_steps, 1, 1) + (1.0 - t) * z0.repeat(num_steps, 1, 1)  # [200, N, 2]
        # We predict v for every step of each sample.
        pred_pair = rectified_flow.model(x_t_pair, t)  # [200, 400, 2](a 2-sample proposal), [num_steps*num_prop, N, 2]
        # pref_logits is used to calculate preference loss.
        pref_logits = (target_pair.repeat(num_steps, 1, 1) - pred_pair).view(pred_pair.shape[0], -1).pow(2).mean(
            dim=-1)  # [num_steps*num_prop, N*2] -> [num_steps*num_prop, 1]
        pref_logits = pref_logits.reshape(-1, num_proposals).mean(
            dim=0)  # [2], [num_steps*num_prop, 1] -> [num_steps, num_prop] -> [num_prop]
        pref_logit_list = [f'{pref_logit:4.3f}' for pref_logit in pref_logits.data]
        raw_labels = input(f"input labels for logits of {pref_logit_list}")
        if bool(raw_labels) is False:
            print("You entered nothing, which means all proposals are not satisfying.")
            continue
        else:
            inject_counter += 1
            try:
                raw_labels = raw_labels.strip().split(' ')
                if raw_labels[0] == 'p' and len(raw_labels) < 2:
                    print(f"You chose the probability mode, but you only entered {raw_labels}. "
                          f"The correct format is p + space + prob, e.g. p 0.1 0.2 ... \n"
                          f"Please enter your preference again.")
            except Exception as err:
                print(f"{err}. Please give each sample a probability (a value in [0,1]) to indicate how much "
                      f"you prefer it or just input the indices (integers in [0,{num_proposals})) of the samples that you prefer.")
                raw_labels = input(f"Once again, input labels for logits of {pref_logit_list}")
            if raw_labels[0] == 'p':
                raw_labels = [float(raw_label) for raw_label in raw_labels[1:]]
                print(f'You assign the first {len(raw_labels[1:])} samples with probabilities {raw_labels[1:]} '
                      f'and 0 will be assigned to the rest.')
                if len(raw_labels) != num_proposals:
                    labels = torch.zeros(num_proposals)
                    labels[:len(raw_labels)] = torch.as_tensor(raw_labels)
                else:
                    labels = torch.as_tensor(raw_labels)
            else:
                print(f'You entered indices: {raw_labels}')
                raw_labels = torch.as_tensor([int(raw_label) for raw_label in raw_labels])
                labels = torch.zeros(num_proposals)
                labels[raw_labels] = 1.0 / len(raw_labels)
            print(f"The final labels are {labels}.")
            z0_copy = copy.deepcopy(z0)  # to show shared noise may generate more prefered shape
        pref_loss = F.cross_entropy(-pref_logits.reshape(1, num_proposals),
                                    labels.reshape(1, num_proposals).to(device))
        optimizer.zero_grad()
        pref_loss.backward()
        g_grad = copy.deepcopy([p.grad.clone() for p in rectified_flow.model.parameters()])
        g_grad_norm = np.sum([torch.sum(g_grad_i.detach().cpu() ** 2) for g_grad_i in g_grad])
        phi = np.minimum(pref_loss.detach().cpu().clone(), g_grad_norm.item()) * 1
        subtractor = np.sum([torch.sum(aa.cpu() * bb.cpu()) for aa, bb in zip(f_grad, g_grad)])
        coef = np.maximum((phi - subtractor) / g_grad_norm, 0)
        for f_grad_i, g_grad_i, p in zip(f_grad, g_grad, rectified_flow.model.parameters()):
            p.grad = f_grad_i + coef * g_grad_i
        optimizer.step()
        print(f"iter {i}, static (flow_loss: {loss_regular.item():.4f}, pref_loss: {pref_loss.item():.4f}), "
              f"coef: {coef:.4f}, injection times: {inject_counter}")
        loss_curve['regular_loss'].append(loss_regular.item())
        loss_curve['pref_loss'].append(pref_loss.item())
        z0 = rectified_flow.get_z0(batchsize=num_samples)
        # plot_func(z0, num_samples, 'green', f'noise_pref_{i}.pdf')
        test(rectified_flow, 0, num_samples=num_samples, color="green", fig_name=f"test_{i}.pdf", plot_interval=1, z0=z0)
        test(rectified_flow, 0, num_samples=num_proposals, color="green", z0=z0_copy, fig_name=f"pref_{i}.pdf", plot_interval=1)
    return loss_curve


if __name__ == '__main__':
    num_datapoints = 400   # number of datapoints in one sample
    num_total_samples = 200  # the size of dataset
    num_eval = 6
    num_proposals = 4
    feature_dim = 2
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    n_steps = 100  # number of time steps
    batch_size = 128  # training batch size
    n_units = 256  # number of hidden units of MLP
    time_dim = 4  # dimension of time embedding
    lr = 1e-3  # learning rate
    wd = 1e-3  # weight decay
    num_iters = 20  # number of training iterarions
    t0 = time.time()
    # ckp = torch.load('rectified_model_t-emb-dim_4_iter_140k.pt')#'rectified_model_ckp.pt')
    ckp = torch.load('rectified_model_ckp1.pt', map_location=device)
    rectified_flow = RectifiedFlow(method='ODE', input_dim=feature_dim, num_steps=n_steps, batchsize=batch_size,
                                   model=MLP(num_datapoints, feature_dim, hidden_num=n_units, time_dim=time_dim),
                                   num_points=num_datapoints, pi0_type='Gaussian', device=device)
    # rectified_flow.model.load_state_dict(ckp['model_state_dict_fixed_x1'])#(ckp['model_state_dict'])
    rectified_flow.model.load_state_dict(ckp['model_state_dict'])
    modelB = MLP(num_datapoints, feature_dim, hidden_num=n_units,
                                        time_dim=time_dim)
    modelB.load_state_dict(rectified_flow.model.state_dict())
    base_flow = RectifiedFlow(method='ODE', input_dim=feature_dim, num_steps=n_steps, batchsize=batch_size,
                              model=modelB,
                              num_points=num_datapoints, pi0_type='Gaussian', device=device)
    optimizer = torch.optim.Adam(rectified_flow.model.parameters(), lr=lr)#, weight_decay=wd)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=20,
    #                                                        threshold=0.0001, min_lr=1.0e-6, eps=1e-09, verbose=True)
    data = None # ckp['scaled_data']  # None
    loss_curve = rectified_flow_pref(base_flow, rectified_flow, optimizer, data, batch_size,
                                     num_iters, num_eval, num_proposals, device)
    time_consumed = time.time() - t0
    print(f"training time: {time_consumed / 60:.1f} minutes. saving model.state_dict()")
    plt.plot(loss_curve['regular_loss'], label='regular loss')
    plt.plot(loss_curve['pref_loss'], label='pref loss')
    plt.legend()
    plt.show()
