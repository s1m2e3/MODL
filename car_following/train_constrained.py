import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import sys
import os
import math
import dill
dill.settings['recurse'] = True

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from mpc.mpc_nn import NStepResidualRNNController,ResidualRNNController
from system_model import system_dynamics,get_random_target_state,constraints_vector,quadratic_loss,\
                        constraints_vector_speed,constraints_vector_steering
from utils import CommandLineArgs,wolfe_line_search_torch,store_json

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

cmd_args = CommandLineArgs()
weights_name = cmd_args.args.weights_name
n_steps_alignment = cmd_args.args.n_steps_alignment
step_size_alignment = cmd_args.args.step_size_alignment
grad_clamp = cmd_args.args.grad_clamp
state_dim = cmd_args.args.state_dim
hidden_dim = cmd_args.args.hidden_dim
output_dim = cmd_args.args.output_dim
trajectory_optimization_epochs = cmd_args.args.trajectory_optimization_epochs
n_init_conditions = cmd_args.args.n_init_conditions
n_trajectory_steps = cmd_args.args.n_trajectory_steps
simulation_stats = {}

rnn_controller = ResidualRNNController(state_dim, hidden_dim, output_dim,device=device)
if os.path.isfile('outputs/'+weights_name+'.pt'):
    rnn_controller.load_state_dict(torch.load('outputs/'+weights_name+'.pt'))

optimizer = optim.Adam(rnn_controller.parameters(), lr=0.001)
torch.autograd.set_detect_anomaly(True)
torch.set_default_device(device)
x_min_change = 15
x_max_change = 15.5
y_min_change = 15.5
y_max_change = 15.5
delta_min_change = math.pi/4
delta_max_change = math.pi/4
mse =  torch.nn.MSELoss()

for init_condition in range(n_init_conditions):
    x_target = torch.tensor(0).unsqueeze(0)+x_min_change
    y_target = torch.tensor(0).unsqueeze(0)+y_min_change
    sigma_target = torch.tensor(0).unsqueeze(0)+delta_min_change
    x_init = torch.stack([torch.tensor(0).unsqueeze(0),
                          torch.tensor(0).unsqueeze(0),
                          torch.tensor(0).unsqueeze(0)]).transpose(0,1)
    # x_target = get_random_target_state(x_init.transpose(0,1),x_min_change,x_max_change,y_min_change,y_max_change,delta_min_change,delta_max_change).transpose(0,1).to(device)
    x_target = torch.stack([x_target,y_target,sigma_target]).transpose(0,1)
    simulation_stats[init_condition] = {}
    
    for trajectory_optimization in range(trajectory_optimization_epochs):
        simulation_stats[init_condition][trajectory_optimization] = {}
        x = x_init.to(device).float().requires_grad_(True)
        x_target = x_target.float().requires_grad_(True)
        u_prev = torch.zeros(2).unsqueeze(0).to(device).float().requires_grad_(True)
        h = torch.ones(1, 1, output_dim).to(device).float().requires_grad_(True)  # Initial hidden state for RNN     
        param_list = [p for p in rnn_controller.parameters() if p.requires_grad]        
        loss = 0
        
        for epoch in range(n_trajectory_steps):
            simulation_stats[init_condition][trajectory_optimization][epoch] = {}
            print(epoch,trajectory_optimization)
            pred, h,sat_loss = rnn_controller(x-x_target, h)
            pred= pred.squeeze(0)
            u = u_prev + pred.squeeze(0)
            if 'guided' in weights_name:
                u_init = u.clone()
                print('u init before any constraining',u_init)
                simulation_stats[init_condition][trajectory_optimization][epoch]['u_init'] = u_init.detach().cpu().numpy().tolist()
                constraints = constraints_vector_speed(u[0,0],u_prev[0,0])
                fixes = [0.05,-0.05,-0.05,0.05]
                counter = 0
                while (constraints>1).any().item() or counter < 50:
                    u[0,0]= u[0,0] + fixes[constraints.argmax().item()]
                    constraints = constraints_vector_speed(u[0,0],u_prev[0,0])
                    counter += 1
                    
                constraints = constraints_vector_steering(u[0,1],x_target[0,2]-x[0,2])
                fixes = [0.05,-0.05,0.05,-0.05]
                counter = 0
                while (constraints>1).any().item() or counter < 50:
                    u[0,1]= u[0,1] + fixes[constraints.argmax().item()]
                    constraints = constraints_vector_steering(u[0,1],x_target[0,2]-x[0,2])
                    counter += 1
                    
                c= constraints_vector(u,u_prev)
                counter = 0
                while torch.any(c>1.0).item() and counter < 10:
                    
                    cons_update_steer = torch.autograd.functional.jacobian(constraints_vector,(u,u_prev))[0].\
                        squeeze(1)
                    cons_update = cons_update_steer.sum(0).clamp(-grad_clamp,grad_clamp)
            
                    cons_update_steer = cons_update_steer[c.argmax().item()].clamp(-grad_clamp,grad_clamp) 
                    
                    step_size_alignment = wolfe_line_search_torch(
                        constraints_vector,
                        (u,
                        u_prev),
                        cons_update,
                        step_size_alignment
                    )
                    cons_update = cons_update_steer+cons_update
                    u=  u-step_size_alignment*(cons_update.clamp(-grad_clamp,grad_clamp))
                    c = constraints_vector(u,u_prev) 
                    counter += 1
                    
                print('before gradient heuristic',u)
                for i in range(n_steps_alignment):
                    
                    loss_func_update = torch.autograd.functional.jacobian(quadratic_loss,(x.float().requires_grad_(True),\
                                                                                            u.float().requires_grad_(True),\
                                                                                            x_target.float().requires_grad_(True)))[1]
                    
                    u = u-0.1*(loss_func_update[:,0].clamp(-grad_clamp,grad_clamp))
                print('after gradient heuristic')
                print(u)
                constraints = constraints_vector_speed(u[0,0],u_prev[0,0])
                fixes = [0.05,-0.05,-0.05,0.05]
                counter = 0
                while (constraints>1).any().item() or counter < 50:
                    u[0,0]= u[0,0] + fixes[constraints.argmax().item()]
                    constraints = constraints_vector_speed(u[0,0],u_prev[0,0])
                    counter += 1
                    
                constraints = constraints_vector_steering(u[0,1],x_target[0,2]-x[0,2])
                fixes = [0.05,-0.05,0.05,-0.05]
                counter = 0
                while (constraints>1).any().item() or counter < 50:
                    u[0,1]= u[0,1] + fixes[constraints.argmax().item()]
                    constraints = constraints_vector_steering(u[0,1],x_target[0,2]-x[0,2])
                    counter += 1
                    
                c= constraints_vector(u,u_prev)
                counter = 0
                while torch.any(c>1.0).item() and counter < 10:
                    
                    cons_update_steer = torch.autograd.functional.jacobian(constraints_vector,(u,u_prev))[0].\
                        squeeze(1)
                    cons_update = cons_update_steer.sum(0).clamp(-grad_clamp,grad_clamp)
            
                    cons_update_steer = cons_update_steer[c.argmax().item()].clamp(-grad_clamp,grad_clamp) 
                    
                    step_size_alignment = wolfe_line_search_torch(
                        constraints_vector,
                        (u,
                        u_prev),
                        cons_update,
                        step_size_alignment
                    )
                    cons_update = cons_update_steer+cons_update
                    u=  u-step_size_alignment*(cons_update.clamp(-grad_clamp,grad_clamp))
                    c = constraints_vector(u,u_prev)
                    counter +=1 

                print('final u',u)                    
                u_final = u.clone()
                loss_u = mse(u_init,u_final)
                
                simulation_stats[init_condition][trajectory_optimization][epoch]['u_final'] = u_final.detach().cpu().numpy().tolist()
                simulation_stats[init_condition][trajectory_optimization][epoch]['loss_u'] = loss_u.detach().cpu().numpy().tolist()

            if 'guided' in weights_name:
                constraint_loss = constraints_vector(u_init,u_prev).sum() 
                step_loss = quadratic_loss(x,u_init,x_target)*0.01
                loss = loss + step_loss + constraint_loss + loss_u + sat_loss
            else:
                constraint_loss = constraints_vector(u,u_prev).sum()*0.01 
                step_loss = quadratic_loss(x,u,x_target)*0.01
                loss = loss + step_loss + constraint_loss + sat_loss
            
            if loss.item()>1e15:
                print('in step constraint',constraint_loss)
                print('in step state loss',step_loss)
                print(u)
                print(constraints_vector(u_init,u_prev))
                print(constraints_vector(u,u_prev))
                input('loss too high')
                break

            simulation_stats[init_condition][trajectory_optimization][epoch]['constraint_loss'] = constraint_loss.detach().cpu().numpy().tolist()
            simulation_stats[init_condition][trajectory_optimization][epoch]['step_loss'] = step_loss.detach().cpu().numpy().tolist()
            simulation_stats[init_condition][trajectory_optimization][epoch]['x'] = x.detach().cpu().numpy().tolist()
            simulation_stats[init_condition][trajectory_optimization][epoch]['u'] = u.detach().cpu().numpy().tolist()
            new_x = system_dynamics(x= x, u=u).T.to(device)
            simulation_stats[init_condition][trajectory_optimization][epoch]['new_x'] = new_x.detach().cpu().numpy().tolist()
            u_prev = u
            x = new_x

        print('\n\n')
        print('stats in trajectory optimization epoch: ',trajectory_optimization)
        print('constraint',constraint_loss)
        print('step loss',step_loss)
        print('control is',u)
        print(x,x_target)
        print('\n\n')
        
        if loss.item()<1e25:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rnn_controller.parameters(), max_norm=5.0)
            optimizer.step()

        optimizer.zero_grad()
        torch.save(rnn_controller.state_dict(), 'outputs/'+weights_name+'.pt')

store_json('outputs/'+weights_name+'.json', simulation_stats)