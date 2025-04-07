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
# Training setup
from system_model import system_dynamics,position_upper_bound,position_lower_bound,control_lower_bound,control_upper_bound,velocity_positive,ConstrainedResidualRNNController
# Training setup
state_dim, hidden_dim, output_dim = 2, 128, 1  # 2D state (position, velocity), control is 1D
rnn_controller = ConstrainedResidualRNNController(state_dim, hidden_dim, output_dim)

rnn_controller.load_state_dict(torch.load('./outputs/model_constrained.pth'))
data = {}

x_upper_bound = 1
u_upper_bound = 2
bigM = 10
steepness = 5

tanh_displacement = 1
softmax_temperature = 0.1

for init_conditions in range(30):
    # Initial conditions
    init_position = torch.empty(1).uniform_(-3,3).requires_grad_()
    init_velocity = -init_position
    x0 = torch.stack([init_position,init_velocity]).squeeze(1)
    h0 = torch.zeros(1, 1, hidden_dim)  # Initial hidden state for RNN
    T_horizon = 600  # MPC horizon    
    # Training loop
    x = x0
    h_prev = h0.clone().detach()
    u_prev = torch.zeros(1,1)
    list_x = [x]
    list_u = []
    for t in range(T_horizon):
        # Generate control from RNN
        u, h = rnn_controller(x.unsqueeze(0), h_prev)
        upper_bound_state = position_upper_bound(x,x_upper_bound).unsqueeze(0)
        lower_bound_state = position_lower_bound(x,x_upper_bound).unsqueeze(0)
        upper_bound_control = control_upper_bound(u_prev,u_upper_bound).squeeze(1)
        lower_bound_control = control_lower_bound(u_prev,u_upper_bound).squeeze(1)
        # upper_bound_velocity = velocity_positive(x).unsqueeze(0)

        # upper_bound_velocity_big_M = upper_bound_velocity*bigM
        upper_bound_constraints_rhs = torch.stack([upper_bound_state,upper_bound_control])
        # upper_bound_constraints_rhs_big_M = torch.stack([upper_bound_state,upper_bound_control,upper_bound_velocity_big_M])

        # upper_bound_constraints = u - upper_bound_constraints_rhs_big_M
        upper_bound_constraints = u - upper_bound_constraints_rhs
        lower_bound_constraints_rhs = torch.stack([lower_bound_state,lower_bound_control])
        # lower_bound_constraints_rhs_big_M = torch.stack([lower_bound_state,lower_bound_control,upper_bound_velocity_big_M])
        
        # lower_bound_constraints = -u-lower_bound_constraints_rhs_big_M
        lower_bound_constraints = -u+lower_bound_constraints_rhs
        normalized_constraints = torch.vstack([upper_bound_constraints,lower_bound_constraints])
        
        normalized_constraints_rhs = torch.vstack([upper_bound_constraints_rhs,lower_bound_constraints_rhs])
        # print(normalized_constraints)
        
        normalized_constraints = torch.relu(normalized_constraints)
        constraint = False
        # if normalized_constraints.sum(dim=0).item()>0:
            # constraint = True
        violated_constraints = 1/2+1/2*torch.tanh(steepness*(normalized_constraints.sum(dim=0))-tanh_displacement)
        normalized_constraints = normalized_constraints -torch.max(normalized_constraints,dim=0).values
        
        weights = rnn_controller.get_constraints_weights(normalized_constraints/softmax_temperature)
        
        # print(weights,violated_constraints,normalized_constraints_rhs,(weights*normalized_constraints_rhs).sum(dim=0).unsqueeze(0))
        # if constraint:
            # print(u)
        u = u + violated_constraints.unsqueeze(0)*((weights*normalized_constraints_rhs).sum(dim=0).unsqueeze(0)-u)
        # if constraint:
            # print(u)
            # input('hipi')
        u_prev = u
        # Simulate next state using ODE solver
        list_u.append(u.squeeze(1))
        x_prev = x
        x_next = system_dynamics(x, u)
        # Compute cost
        # loss_integral = cost_states_integral(x,torch.zeros_like(x)).unsqueeze(0)
        # total_integral += loss_integral
        list_x.append(x_next)
        # Update state
        x = x_next
        h_prev = h

    prev = torch.stack(list_x).detach().numpy()
    prev_u = torch.stack(list_u).detach().numpy()
    data['sample {} x'.format(init_conditions)] = prev[:T_horizon,0]
    data['sample {} v'.format(init_conditions)] = prev[:T_horizon,1]
    data['sample {} u'.format(init_conditions)] = prev_u[:T_horizon,0]

data = pd.DataFrame.from_dict(data)

data.to_csv('./outputs/test_constrained.csv')