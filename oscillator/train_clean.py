import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from mpc.mpc_nn import ResidualRNNController
from system_model import system_dynamics,cost_function_state_following
# Training setup
state_dim, hidden_dim, output_dim = 2, 32, 1  # 2D state (position, velocity), control is 1D
rnn_controller = ResidualRNNController(state_dim, hidden_dim, output_dim)
optimizer = optim.Adam(rnn_controller.parameters(), lr=0.01)

for init_conditions in range(30):
    # Initial conditions
    init_position = torch.empty(1).uniform_(-1,1).requires_grad_()
    init_velocity = -init_position
    x0 = torch.stack([init_position,init_velocity]).squeeze(1)
    h0 = torch.zeros(1, 1, hidden_dim)  # Initial hidden state for RNN
    T_horizon = 100  # MPC horizon    
    # Training loop
    for epoch in range(100):
        optimizer.zero_grad()
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
            # Compute cost
            loss = cost_function_state_following(x[0],torch.zeros_like(x[0]),1,u,torch.zeros_like(u),0.001)
            total_loss += loss
            list_x.append(x_next)
            # Update state
            x = x_next
        total_loss.backward()
        print(total_loss,loss)
        optimizer.step()
        prev = torch.stack(list_x).detach().numpy()
        prev_u = torch.stack(list_u).detach().numpy()
        if epoch % 10 == 0 :
            print(f"Epoch {epoch}, Loss: {total_loss.item()}")
        

# Store model parameters
torch.save(rnn_controller.state_dict(), './outputs/model_clean.pth')