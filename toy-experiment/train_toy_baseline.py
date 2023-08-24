import matplotlib.pyplot as plt
import torch
import numpy as np
from utils_toy import make_data, MLP, RectifiedFlow, test

if __name__ == '__main__':
    device = torch.device('cuda:1' if torch.cuda.is_available() else "cpu")
    num_datapoints = 400
    num_iters = 150000
    batchsize = 128
    input_dim = 2
    time_dim = 4
    n_steps = 100
    n_units = 256
    lr = 1e-3
    wd = 1e-3
    show_plots = False  # If True, plots will be shown during the training process.

    rectified_flow = RectifiedFlow(method='ODE', device=device,
                                   model=MLP(num_datapoints, input_dim, hidden_num=n_units, time_dim=time_dim),
                                   input_dim=input_dim, num_steps=n_steps, batchsize=batchsize,
                                   num_points=num_datapoints, pi0_type='Gaussian')
    optimizer = torch.optim.Adam(rectified_flow.model.parameters(), lr=lr, weight_decay=wd)
    loss_curve, loss_curve1 = [], []
    # Experiments show that the number of samples should be large and the data do not necessarily have to be stored.
    target_samples, _ = make_data(num_samples_each_class=20000, num_datapoints=num_datapoints)
    for i in range(num_iters + 1):
        indices = torch.randperm(len(target_samples))[:batchsize]
        batch = target_samples[indices]
        batch = batch.to(device)

        x_t, t, target = rectified_flow.get_train_tuple(batch=batch)

        optimizer.zero_grad()
        pred = rectified_flow.model(x_t, t)
        loss = (target - pred).view(pred.shape[0], -1).pow(2).sum(dim=-1)
        loss = loss.mean()
        loss.backward()
        optimizer.step()

        if i % 500 == 0:
            loss_curve.append(np.log(loss.item()))  ## to store the loss curve
            print("iter:", i, "loss(fixed X1): %.2f" % loss.item())
        if show_plots and (i+1) % 5000:
            test(rectified_flow, num_samples=4, color='red')

    print("saving ckp")
    torch.save({'model_state_dict': rectified_flow.model.state_dict(),
                # 'scaled_data': target_samples,
                }, 'rectified_model_ckp1.pt')
