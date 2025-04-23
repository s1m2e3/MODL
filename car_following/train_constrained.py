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
from system_model import system_dynamics,get_random_target_state,cost_function_state_following
from system_model import angle_change_negative_lower_bound,angle_change_positive_upper_bound,\
                            binary_angle_change_positive,\
                            steering_angle_rate_lower_bound,steering_angle_rate_upper_bound,\
                            acceleration_lower_bound,acceleration_upper_bound,\
                            angle_change_negative_upper_bound,angle_change_positive_lower_bound,\
                            steering_angle_difference_lower_bound,steering_angle_difference_upper_bound,\
                            acceleration_difference_lower_bound,acceleration_difference_upper_bound,\
                            steering_like_previous_lower_bound,steering_like_previous_upper_bound,\
                            car_following_estimation,car_following_lower_bound,car_following_upper_bound

# Training setup
state_dim, hidden_dim, output_dim = 5, 256, 2  # 2D state (position, velocity), control is 1D
n_steps = 3
rnn_controller = NStepResidualRNNController(state_dim, hidden_dim, output_dim, n_steps=n_steps)
optimizer = optim.Adam(rnn_controller.parameters(), lr=0.01)

x_min_change = 15
x_max_change = 20
y_min_change = 30
y_max_change = 50
delta_min_change = math.pi/5
delta_max_change = math.pi/4

rho_difference_max = torch.tensor(0.2)
rho_max = torch.tensor(math.pi/2)
accel_max = torch.tensor(1)
speed_max = torch.tensor(10)
accel_difference_max = torch.tensor(0.3)
min_speed = accel_max/10
target_speed = 15
desired_headway = 1
steepness = 5
tanh_displacement = 1
softmax_temperature = 0.1
columns_to_select = [0,1,2,3,4]

for init_condition in range(100):
    x0 = torch.tensor(0).unsqueeze(0)
    y0 = torch.tensor(0).unsqueeze(0)
    v0 = torch.tensor(0).unsqueeze(0)
    sigma0 = torch.tensor(0).unsqueeze(0)
    delta0 = torch.tensor(0).unsqueeze(0)
    x_init = torch.stack([x0,y0,v0,sigma0,delta0]).transpose(0,1)
    x_target = get_random_target_state(x_init.transpose(0,1),x_min_change,x_max_change,y_min_change,y_max_change,delta_min_change,delta_max_change).transpose(0,1)
    
    for epoch in range(50):
        optimizer.zero_grad()
        x = x_init
        h = torch.zeros(1, 1, hidden_dim)  # Initial hidden state for RNN
        u_prev = torch.zeros(2)
        list_x = [x]
        list_u = []
        total_loss = 0
        for t in range(100):
            # Generate control from RNN
            u, h = rnn_controller(x-x_target, h)
            list_u.append(u.squeeze(1))
            for step in range(n_steps):
                # print("first controls are:",u[step])
                print("state and target heading are:",x,x_target)
                car_following = car_following_estimation(x.squeeze(0),x_target.squeeze(0),desired_headway,target_speed,accel_max)
                binary_car_following = torch.tanh(steepness*torch.relu(car_following))
                print(car_following,binary_car_following)
                car_following_lower_bound_rhs = car_following_lower_bound(car_following,binary_car_following)
                car_following_upper_bound_rhs = car_following_upper_bound(car_following,binary_car_following)

                car_following_lower_bound_constraint = -u[step,0]+car_following_lower_bound_rhs
                car_following_upper_bound_constraint = u[step,0]-car_following_upper_bound_rhs

                acceleration_normalized_constraints_rhs = torch.vstack([car_following_lower_bound_rhs,car_following_upper_bound_rhs])
                acceleration_normalized_constraints = torch.vstack([car_following_lower_bound_constraint,car_following_upper_bound_constraint])
                acceleration_normalized_constraints = torch.relu(acceleration_normalized_constraints)
                # print(acceleration_normalized_constraints)
                acceleration_violated_constraints = 1/2+1/2*torch.tanh(steepness*(acceleration_normalized_constraints.sum(dim=0))-tanh_displacement)
                acceleration_normalized_constraints = acceleration_normalized_constraints -torch.max(acceleration_normalized_constraints,dim=0).values
                acceleration_weights = rnn_controller.get_constraints_weights(acceleration_normalized_constraints/(softmax_temperature/2))
                print("acceleration before is:",u[step,0])
                u[step,0] = u[step,0] + acceleration_violated_constraints.unsqueeze(0)*((acceleration_weights*acceleration_normalized_constraints_rhs).sum(dim=0).unsqueeze(0)-u[step,0])
                print("bounded acceleration is:",u[step,0])
                # input('hipi')
                # if t%10==0:
                    
                #     input('hipi')
                acceleration_lower_bound_rhs =  acceleration_lower_bound(u[step],accel_max)
                acceleration_upper_bound_rhs =  acceleration_upper_bound(u[step],accel_max)
                
                acceleration_difference_lower_bound_rhs = acceleration_difference_lower_bound(u_prev[0],accel_difference_max)
                acceleration_difference_upper_bound_rhs = acceleration_difference_upper_bound(u_prev[0],accel_difference_max)
                acceleration_upper_bound_constraints_rhs = acceleration_upper_bound_rhs
                acceleration_lower_bound_constraints_rhs = acceleration_lower_bound_rhs
                # velocity_positive_constraint_rhs = velocity_positive_rhs
                
                acceleration_upper_bound_constraint = u[step,0]-acceleration_upper_bound_rhs
                acceleration_lower_bound_constraint = -u[step,0]+acceleration_lower_bound_constraints_rhs
                # velocity_positive_constraint = -u[step,0]+velocity_positive_rhs
                acceleration_difference_lower_bound_constraint = -u[step,0]+acceleration_difference_lower_bound_rhs
                acceleration_difference_upper_bound_constraint = u[step,0]-acceleration_difference_upper_bound_rhs

                acceleration_normalized_constraints_rhs = torch.vstack([acceleration_upper_bound_rhs,acceleration_lower_bound_constraints_rhs,
                                                                        acceleration_difference_lower_bound_rhs,acceleration_difference_upper_bound_rhs])
                acceleration_normalized_constraints = torch.vstack([acceleration_upper_bound_constraint,acceleration_lower_bound_constraint,
                                                                        acceleration_difference_lower_bound_constraint,acceleration_difference_upper_bound_constraint])
                acceleration_normalized_constraints = torch.relu(acceleration_normalized_constraints)
                # print(acceleration_normalized_constraints)
                acceleration_violated_constraints = 1/2+1/2*torch.tanh(steepness*(acceleration_normalized_constraints.sum(dim=0))-tanh_displacement)
                acceleration_normalized_constraints = acceleration_normalized_constraints -torch.max(acceleration_normalized_constraints,dim=0).values
                acceleration_weights = rnn_controller.get_constraints_weights(acceleration_normalized_constraints/(softmax_temperature/2))
                print("acceleration before is:",u[step,0])
                u[step,0] = u[step,0] + acceleration_violated_constraints.unsqueeze(0)*((acceleration_weights*acceleration_normalized_constraints_rhs).sum(dim=0).unsqueeze(0)-u[step,0])
                print("bounded acceleration is:",u[step,0])
                if t%10==0:
                    print("state and target heading are:",x,x_target)
                    input('hipi')
                # input('yipi')
                # print('steering before is',u[step,1])
                angle_change_positive_binary = torch.tanh(steepness*torch.relu(binary_angle_change_positive(x.squeeze(0),x_target.squeeze(0))))
                # print("binaries are:",angle_change_positive_binary)
                # print("angle change positive_lower bound")
                # input('yipi')
                angle_change_positive_lower_bound_rhs = angle_change_positive_lower_bound(x.squeeze(0),u[step],angle_change_positive_binary)
                angle_change_positive_upper_bound_rhs = angle_change_positive_upper_bound(x_target.squeeze(0),x.squeeze(0),u[step],angle_change_positive_binary)
                angle_change_negative_upper_bound_rhs = angle_change_negative_upper_bound(x.squeeze(0),u[step],angle_change_positive_binary)
                angle_change_negative_lower_bound_rhs = angle_change_negative_lower_bound(x_target.squeeze(0),x.squeeze(0),u[step],angle_change_positive_binary)
                angle_change_lower_bound_rhs = torch.vstack([angle_change_negative_lower_bound_rhs,angle_change_positive_lower_bound_rhs])
                angle_change_upper_bound_rhs = torch.vstack([angle_change_positive_upper_bound_rhs,angle_change_negative_upper_bound_rhs])
                # print("angle change rhs are:",angle_change_lower_bound_rhs,angle_change_upper_bound_rhs)
                angle_change_upper_bound_constraint = u[step,1]-angle_change_upper_bound_rhs
                angle_change_lower_bound_constraint = -u[step,1]+angle_change_lower_bound_rhs
                
                # print("angle change constraints are:",angle_change_upper_bound_constraint,angle_change_lower_bound_constraint)
                steering_angle_rate_normalized_constraints_rhs = torch.vstack([angle_change_lower_bound_rhs,
                                                                               angle_change_upper_bound_rhs])
                steering_angle_rate_normalized_constraints = torch.vstack([angle_change_lower_bound_constraint,
                                                                               angle_change_upper_bound_constraint])
                steering_angle_rate_normalized_constraints = torch.relu(steering_angle_rate_normalized_constraints)
                # print(steering_angle_rate_normalized_constraints)
                steering_angle_rate_violated_constraints = 1/2+1/2*torch.tanh(steepness*(steering_angle_rate_normalized_constraints.sum(dim=0))-tanh_displacement)
                steering_angle_rate_normalized_constraints = steering_angle_rate_normalized_constraints -torch.max(steering_angle_rate_normalized_constraints,dim=0).values
                steering_angle_rate_weights = rnn_controller.get_constraints_weights(steering_angle_rate_normalized_constraints/softmax_temperature)
                # print(steering_angle_rate_weights)
                u[step,1] = u[step,1] + steering_angle_rate_violated_constraints.unsqueeze(0)*\
                    ((steering_angle_rate_weights*steering_angle_rate_normalized_constraints_rhs).sum(dim=0).unsqueeze(0)-u[step,1])

                # print("controls over positive or negative is",u[step,1])
                # input('yipi')
                
                steering_angle_rate_lower_bound_rhs =  steering_angle_rate_lower_bound(u[step],rho_max)
                steering_angle_rate_upper_bound_rhs =  steering_angle_rate_upper_bound(u[step],rho_max)
               
                steering_angle_rate_upper_bound_constraint = u[step,1]-steering_angle_rate_upper_bound_rhs
                steering_angle_rate_lower_bound_constraint = -u[step,1]+steering_angle_rate_lower_bound_rhs

                # print(steering_angle_rate_lower_bound_rhs,steering_angle_rate_upper_bound_rhs)
                steering_angle_rate_difference_lower_bound_rhs =  steering_angle_difference_lower_bound(u_prev[1],rho_difference_max)
                steering_angle_rate_difference_upper_bound_rhs =  steering_angle_difference_upper_bound(u_prev[1],rho_difference_max)
                steering_angle_rate_difference_upper_bound_constraint = u[step,1]-steering_angle_rate_difference_upper_bound_rhs
                steering_angle_rate_difference_lower_bound_constraint = -u[step,1]+steering_angle_rate_difference_lower_bound_rhs

                steering_like_previous_lower_bound_rhs =  steering_like_previous_lower_bound(u_prev[1],angle_change_positive_binary)
                steering_like_previous_upper_bound_rhs =  steering_like_previous_upper_bound(u_prev[1],angle_change_positive_binary)
                steering_like_previous_upper_bound_constraint = u[step,1] - steering_like_previous_upper_bound_rhs
                steering_like_previous_lower_bound_constraint = -u[step,1] + steering_like_previous_lower_bound_rhs

                steering_angle_rate_normalized_constraints_rhs = torch.vstack([steering_angle_rate_upper_bound_rhs,
                                                                               steering_angle_rate_lower_bound_rhs,
                                                                               steering_angle_rate_difference_lower_bound_rhs,
                                                                               steering_angle_rate_difference_upper_bound_rhs,
                                                                               steering_like_previous_lower_bound_rhs,
                                                                               steering_like_previous_upper_bound_rhs])
                steering_angle_rate_normalized_constraints = torch.vstack([steering_angle_rate_upper_bound_constraint,
                                                                               steering_angle_rate_lower_bound_constraint,
                                                                               steering_angle_rate_difference_lower_bound_constraint,
                                                                               steering_angle_rate_difference_upper_bound_constraint,
                                                                               steering_like_previous_lower_bound_constraint,
                                                                               steering_like_previous_upper_bound_constraint])
                # steering_angle_rate_normalized_constraints_rhs = torch.vstack([steering_angle_rate_difference_lower_bound_rhs,
                #                                                         steering_angle_rate_difference_upper_bound_rhs])
                # steering_angle_rate_normalized_constraints = torch.vstack([steering_angle_rate_difference_lower_bound_constraint,
                #                                                                steering_angle_rate_difference_upper_bound_constraint])
                steering_angle_rate_normalized_constraints = torch.relu(steering_angle_rate_normalized_constraints)
                
                # print(steering_angle_rate_normalized_constraints_rhs,steering_angle_rate_normalized_constraints)
                steering_angle_rate_violated_constraints = 1/2+1/2*torch.tanh(steepness*(steering_angle_rate_normalized_constraints.sum(dim=0))-tanh_displacement)
                steering_angle_rate_normalized_constraints = steering_angle_rate_normalized_constraints -torch.max(steering_angle_rate_normalized_constraints,dim=0).values
                steering_angle_rate_weights = rnn_controller.get_constraints_weights(steering_angle_rate_normalized_constraints/softmax_temperature)
                # print(steering_angle_rate_violated_constraints,steering_angle_rate_weights)
                u[step,1] = u[step,1] + steering_angle_rate_violated_constraints.unsqueeze(0)*\
                    ((steering_angle_rate_weights*steering_angle_rate_normalized_constraints_rhs).sum(dim=0).unsqueeze(0)-u[step,1])
                # print("new_ratechange is:",u[step,1])
                # print(x[0])
                # print('final soft controls are:',u[step])
                u_prev = u[step]
                new_x = system_dynamics(x[0], u[step])
                # print("new state is:",new_x[0])]
             
                # loss_heading = 2*torch.nn.functional.l1_loss(new_x[0,3],x_target[0,3])
                # total_loss += loss_heading
                x=new_x
                # input('yipi')
               
        loss = cost_function_state_following(new_x[0,columns_to_select],x_target[0,columns_to_select],1,u[step],torch.zeros_like(u[step]),0.01)
        total_loss += loss
        total_loss.backward()
        optimizer.step()
        prev = torch.stack(list_x).detach().numpy()
        prev_u = torch.stack(list_u).detach().numpy()
        # if epoch %  10 ==0:
        final_distance = x
        final_x = final_distance[0,0]
        final_y = final_distance[0,1]
        final_delta = final_distance[0,3]
        target_x = x_target[0,0]
        target_y = x_target[0,1]
        target_delta = x_target[0,3]
        print(f"Epoch {epoch}, final position Loss: {loss.item()}")
        print(f"Epoch {epoch}, total Loss: {total_loss.item()}, final state is: {final_x}, {final_y}, {final_delta}, target is: {target_x}, {target_y}, {target_delta}")
        

# Store model parameters
torch.save(rnn_controller.state_dict(), './outputs/model_clean.pth')