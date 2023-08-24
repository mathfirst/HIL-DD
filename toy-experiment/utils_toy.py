import torch
import numpy as np
import torch.nn as nn
from torch.distributions import Normal, Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
import matplotlib.pyplot as plt

# To make our code have better readability, we first define some helper functions.
# we need to make two datasets share the common value range of datapoints
def range_alignment(data, min_value=-1, max_value=1):
    # input: one dataset.
    # output: a new version of data whose minimum and maximum are min_value and max_value, respectively
    a, b = min_value, max_value
    c, d = np.min(data), np.max(data)
    if a != c or b != d:
        data = ((b - a) * data + a * d - b * c) / (d - c)
    return data


def make_data(num_samples_each_class=100, num_datapoints=400):
    from sklearn.datasets import make_s_curve, make_swiss_roll
    x = torch.zeros(num_samples_each_class * 2, num_datapoints, 2)
    y = torch.zeros(num_samples_each_class * 2)
    for i in range(num_samples_each_class):
        s_curve, _ = make_s_curve(num_datapoints, noise=0.1)  # make 10000 data points
        s_curve = s_curve[:, [0, 2]]  # take two features of the data
        s_curve = range_alignment(s_curve, min_value=-1, max_value=1)
        swiss_roll, _ = make_swiss_roll(num_datapoints, noise=0.1)  # make 10000 data points
        swiss_roll = swiss_roll[:, [0, 2]]  # take two features of the data
        swiss_roll = range_alignment(swiss_roll, min_value=-1, max_value=1)
        x[2 * i, :, :], x[2 * i + 1, :, :] = torch.from_numpy(s_curve), torch.from_numpy(swiss_roll)
        y[2 * i + 1] = 1

    return x, y


class MLP(nn.Module):
    def __init__(self, num_datapoints, input_dim=2, hidden_num=100, n_steps=100, time_dim=None):
        super().__init__()
        self.n_steps = n_steps
        if time_dim is None:
            self.time_dim = input_dim
        else:
            self.time_dim = time_dim
        self.fc1 = nn.Linear(input_dim + self.time_dim, hidden_num, bias=False)
        self.layernorm1 = nn.LayerNorm([num_datapoints, hidden_num])
        self.fc2 = nn.Linear(hidden_num, hidden_num, bias=False)
        self.layernorm2 = nn.LayerNorm([num_datapoints, hidden_num])
        self.fc3 = nn.Linear(hidden_num, hidden_num, bias=False)
        self.layernorm3 = nn.LayerNorm([num_datapoints, hidden_num])
        self.fc4 = nn.Linear(hidden_num, input_dim, bias=False)
        self.act = nn.LeakyReLU()

        self.time_embedding = nn.Embedding(n_steps, self.time_dim)

    def forward(self, x_input, t):
        t = (self.n_steps * t.squeeze()).int()
        t_embedding = self.time_embedding(t) * 0.2  # multiplied by 0.2 to prevent the learning process from being dominated by time embeddings
        inputs = torch.cat([x_input, t_embedding.unsqueeze(1).repeat(1, x_input.shape[1], 1)], dim=2)  # (bs, num_datapoints, dim_input + dim_time_emb)
        x = self.fc1(inputs)
        x = self.act(self.layernorm1(x))
        x = self.act(self.layernorm2(self.fc2(x)))
        x = self.act(self.layernorm3(self.fc3(x)))
        x = self.fc4(x)

        return x


# RectifiedFlow() is based on Xingchao Liu's code.
class RectifiedFlow():
    def __init__(self, method='ODE', model=None, input_dim=2, num_steps=100, batchsize=128, num_points=400,
                 pi0_type='Gaussian', device='cuda'):
        self.method = method
        self.model = model.to(device)
        self.input_dim = input_dim
        self.N = num_steps
        self.batchsize = batchsize
        self.pi0_type = pi0_type
        self.num_points = num_points
        self.init_model = None
        self.device = device

    def get_z0(self, batchsize=None):
        if batchsize is None:
            batchsize = self.batchsize

        if self.pi0_type == 'Gaussian':
            self.init_model = MultivariateNormal(torch.zeros(2), torch.eye(2))
            z0 = self.init_model.sample([batchsize, self.num_points])
        else:
            assert False, 'Not implemented'

        z0 = range_alignment(z0.numpy(), min_value=-1, max_value=1)
        z0 = torch.from_numpy(z0)
        return z0.to(self.device)

    def get_train_tuple(self, batch=None, z0=None):
        if self.method == 'ODE':
            if z0 is None:
                z0 = self.get_z0(batchsize=batch.shape[0])

            t = torch.rand((batch.shape[0], 1, 1)).to(self.device)
            t_ = t.repeat(1, self.num_points, 1)
            x_t = t_ * batch + (1. - t_) * z0
            target = batch - z0
        else:
            assert False, 'Not implemented'

        return x_t, t, target

    def sample_ode(self, batchsize=None, z0=None, N=None):
        ### NOTE: Use Euler method to sample from the learned flow
        if N is None:
            N = self.N
        dt = 1. / N
        traj = []  # to store the trajectory
        if z0 is not None:
            z = z0.detach().clone()
            batchsize = z.shape[0]
        else:
            if batchsize is None:
                batchsize = self.batchsize
            z = self.get_z0(batchsize=batchsize)

        traj.append(z.detach().clone())
        with torch.no_grad():
            for i in range(N):
                t = torch.ones((batchsize,)) * i / N
                pred = self.model(z.to(self.device), t.to(self.device))
                z = z.detach().clone() + pred * dt
                traj.append(z.detach().clone())

        return traj

    def spacing_sample_ode(self, batchsize=None, z0=None, N=None, interval=None):
        ### NOTE: Use Euler method to sample from the learned flow
        if N is None:
            N = self.N
        if interval is None:
            interval = 1

        dt = interval * 1. / N
        traj = []  # to store the trajectory
        if z0 is not None:
            z = z0.detach().clone()
            batchsize = z.shape[0]
        else:
            if batchsize is None:
                batchsize = self.batchsize
            z = self.get_z0(batchsize=batchsize)

        traj.append(z.detach().clone())
        with torch.no_grad():
            for i in range(0, N, interval if interval is not None else 1):
                t = torch.ones((batchsize,)) * i / N
                pred = self.model(z.to(self.device), t.to(self.device))
                z = z.detach().clone() + pred * dt
                traj.append(z.detach().clone())

        return traj


def test(rectified_flow, num_samples=8, color="purple", z0=None, fig_name=None):
    rectified_flow.model.eval()
    with torch.no_grad():
        traj = rectified_flow.sample_ode(num_samples, z0=z0)
        fig, ax = plt.subplots(1, num_samples, figsize=(3 * num_samples, 3))
        for i in range(num_samples):
            ax[i].scatter(*traj[-1][i].cpu().numpy().T, color=color, edgecolor="white")
            ax[i].set_axis_off()
        if fig_name is not None:
            print(f'saving {fig_name}')
            plt.savefig(fig_name)
        plt.show()

