import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import sys
import os
import math
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from mpc.mpc_nn import NStepResidualRNNController
from system_model import system_dynamics,get_inertial_target_state,get_random_target_state,cost_function_state_following
from system_model import angle_change_negative,angle_change_positive,binary_angle_change_negative,binary_angle_change_positive,steering_angle_rate_lower_bound,steering_angle_rate_upper_bound,acceleration_lower_bound,acceleration_upper_bound
# Training setup
state_dim, hidden_dim, output_dim = 5, 128, 2  # 2D state (position, velocity), control is 1D
n_steps = 3
rnn_controller = NStepResidualRNNController(state_dim, hidden_dim, output_dim, n_steps=n_steps)
optimizer = optim.Adam(rnn_controller.parameters(), lr=0.001)

x_min_change = 10
x_max_change = 30
y_min_change = 10
y_max_change = 30
delta_min_change = math.pi/4
delta_max_change = math.pi/2
rho_max = torch.tensor(0.2)
accel_max = torch.tensor(0.1)
steepness = 10
tanh_displacement = 1
softmax_temperature = 0.1


for init_condition in range(30):
    x0 = torch.rand(1)*5
    y0 = torch.rand(1)*5
    v0 = torch.tensor(0).unsqueeze(0)
    sigma0 = torch.tensor(0).unsqueeze(0)
    delta0 = torch.tensor(0).unsqueeze(0)
    x_init = torch.stack([x0,y0,v0,sigma0,delta0]).transpose(0,1)
    x_target = get_random_target_state(x_init.transpose(0,1),x_min_change,x_max_change,y_min_change,y_max_change,delta_min_change,delta_max_change).transpose(0,1)
    for epoch in range(1000):
        optimizer.zero_grad()
        x = x_init
        h = torch.zeros(1, 1, hidden_dim)  # Initial hidden state for RNN
        list_x = [x]
        list_u = []
        total_loss = 0
        for t in range(30):
            # Generate control from RNN
            u, h = rnn_controller(x-x_target, h)

            list_u.append(u.squeeze(1))
            for step in range(n_steps):
                
                acceleration_lower_bound_rhs =  acceleration_lower_bound(u[step],accel_max)
                acceleration_upper_bound_rhs =  acceleration_upper_bound(u[step],accel_max)
                acceleration_upper_bound_constraints_rhs = acceleration_upper_bound_rhs
                acceleration_lower_bound_constraints_rhs = acceleration_lower_bound_rhs
                
                acceleration_upper_bound_constraint = u[step,0]-acceleration_upper_bound_rhs
                acceleration_lower_bound_constraint = -u[step,0]+acceleration_lower_bound_rhs
                
                acceleration_normalized_constraints_rhs = torch.vstack([acceleration_upper_bound_rhs,acceleration_lower_bound_rhs])
                acceleration_normalized_constraints = torch.vstack([acceleration_upper_bound_constraint,acceleration_lower_bound_constraint])
                acceleration_normalized_constraints = torch.relu(acceleration_normalized_constraints)

                acceleration_violated_constraints = 1/2+1/2*torch.tanh(steepness*(acceleration_normalized_constraints.sum(dim=0))-tanh_displacement)
                acceleration_normalized_constraints = acceleration_normalized_constraints -torch.max(acceleration_normalized_constraints,dim=0).values
                acceleration_weights = rnn_controller.get_constraints_weights(acceleration_normalized_constraints/softmax_temperature)
                
                u[step,0] = u[step,0] + acceleration_violated_constraints.unsqueeze(0)*((acceleration_weights*acceleration_normalized_constraints_rhs).sum(dim=0).unsqueeze(0)-u[step,0])
                
                angle_change_positive_binary = torch.tanh(steepness*torch.relu(binary_angle_change_positive(x.squeeze(0),x_target.squeeze(0))))
                angle_change_negative_binary = torch.tanh(steepness*torch.relu(binary_angle_change_negative(x.squeeze(0),x_target.squeeze(0))))
                
                angle_change_positive_rhs = angle_change_positive(x.squeeze(0),u[step],angle_change_positive_binary)
                angle_change_negative_rhs = angle_change_negative(x.squeeze(0),u[step],angle_change_negative_binary)

                angle_change_negative_constraint = u[step,1]-angle_change_negative_rhs
                angle_change_positive_constraint = -u[step,1]+angle_change_positive_rhs
                
                steering_angle_rate_lower_bound_rhs =  steering_angle_rate_lower_bound(u[step],rho_max)
                steering_angle_rate_upper_bound_rhs =  steering_angle_rate_upper_bound(u[step],rho_max)
                
                steering_angle_rate_lower_bound_constraints_rhs = steering_angle_rate_lower_bound_rhs
                steering_angle_rate_upper_bound_constraints_rhs = steering_angle_rate_upper_bound_rhs
                
                steering_angle_rate_lower_bound_constraint = u[step,1]-steering_angle_rate_lower_bound_rhs
                steering_angle_rate_upper_bound_constraint = -u[step,1]+steering_angle_rate_upper_bound_rhs
                
                steering_angle_rate_normalized_constraints_rhs = torch.vstack([steering_angle_rate_upper_bound_constraints_rhs,
                                                                               steering_angle_rate_lower_bound_constraints_rhs,
                                                                               angle_change_positive_rhs,
                                                                               angle_change_negative_rhs])
                print(sigma0,x_target[0,3],u[step,1])
                print(steering_angle_rate_normalized_constraints_rhs)
                steering_angle_rate_normalized_constraints = torch.vstack([steering_angle_rate_upper_bound_constraint,
                                                                               steering_angle_rate_lower_bound_constraint,
                                                                               angle_change_positive_constraint,
                                                                               angle_change_negative_constraint,])
                steering_angle_rate_normalized_constraints = torch.relu(steering_angle_rate_normalized_constraints)
                print(steering_angle_rate_normalized_constraints)
                steering_angle_rate_violated_constraints = 1/2+1/2*torch.tanh(steepness*(steering_angle_rate_normalized_constraints.sum(dim=0))-tanh_displacement)
                steering_angle_rate_normalized_constraints = steering_angle_rate_normalized_constraints -torch.max(steering_angle_rate_normalized_constraints,dim=0).values
                steering_angle_rate_weights = rnn_controller.get_constraints_weights(steering_angle_rate_normalized_constraints/softmax_temperature)
                
                print(u[step,1])
                u[step,1] = u[step,1] + steering_angle_rate_violated_constraints.unsqueeze(0)*\
                    ((steering_angle_rate_weights*steering_angle_rate_normalized_constraints_rhs).sum(dim=0).unsqueeze(0)-u[step,1])
                print(u[step,1])
                input('hipi')
                new_x = system_dynamics(x[0], u[step])
                loss = cost_function_state_following(new_x[0],x_target[0],1,u[0],torch.zeros_like(u[0]),0.001)
                total_loss += loss
                x=new_x
            
        total_loss.backward()
        optimizer.step()
        prev = torch.stack(list_x).detach().numpy()
        prev_u = torch.stack(list_u).detach().numpy()
        if epoch %  10 ==0:
            final_distance = x
            final_x = final_distance[0,0]
            final_y = final_distance[0,2]
            final_delta = final_distance[0,4]
            target_x = x_target[0,0]
            target_y = x_target[0,2]
            target_delta = x_target[0,4]
            print(f"Epoch {epoch}, Loss: {total_loss.item()}, final state is: {final_x}, {final_y}, {final_delta}, target is: {target_x}, {target_y}, {target_delta}")
            

# Store model parameters
torch.save(rnn_controller.state_dict(), './outputs/model_clean.pth')