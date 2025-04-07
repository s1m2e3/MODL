import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from mpc.mpc_nn import ResidualRNNController
from system_model import system_dynamics
# Training setup
state_dim, hidden_dim, output_dim = 2, 32, 1  # 2D state (position, velocity), control is 1D
rnn_controller = ResidualRNNController(state_dim, hidden_dim, output_dim)
rnn_controller.load_state_dict(torch.load('./outputs/model_clean.pth'))
data = {}

for init_conditions in range(30):
    # Initial conditions
    x0 = torch.empty(2).uniform_(-3,3).requires_grad_()
    h0 = torch.zeros(1, 1, hidden_dim)  # Initial hidden state for RNN
    T_horizon = 600  # MPC horizon    
    # Training loop
    x = x0
    h = h0.clone().detach()
    list_x = [x]
    list_u = []
    total_loss = 0
    for t in range(T_horizon):
        # Generate control from RNN
        u, h = rnn_controller(x.unsqueeze(0), h)
        
        # Simulate next state using ODE solver
        list_u.append(u.squeeze(1))
        x_next = system_dynamics(x, u)
        list_x.append(x_next)
        # Update state
        x = x_next
    prev = torch.stack(list_x).detach().numpy()
    prev_u = torch.stack(list_u).detach().numpy()
    data['sample {} x'.format(init_conditions)] = prev[:600,0]
    data['sample {} v'.format(init_conditions)] = prev[:600,1]
    data['sample {} u'.format(init_conditions)] = prev_u[:,0]

data = pd.DataFrame.from_dict(data)

data.to_csv('./outputs/test_clean.csv')