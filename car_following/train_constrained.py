import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import sys
import os
import math
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from mpc.mpc_nn import NStepResidualRNNController,ResidualRNNController
from system_model import system_dynamics,get_random_target_state

import dill
dill.settings['recurse'] = True

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def load_function_lambdified(filename):
    """
    Loads a lambdified function from a specified file using dill.

    Args:
        filename (str): The path to the file from which to load the function.

    Returns:
        function: The lambdified function loaded from the specified file.
    """

    return dill.load(open(filename, "rb"))




def wolfe_line_search_torch(f, xk, gk,
                            alpha_init=1.0, 
                            c1=1e-4, c2=0.9, 
                            alpha_min=1e-8, 
                            max_iter=10,
                            grad_clamp=1):
    """
    Wolfe line search in PyTorch using automatic differentiation.
    
    Parameters:
        f        : callable, function taking tensor x and returning scalar
        xk       : torch.Tensor, current point (requires_grad=True)
        alpha_init : float, initial step size
        c1, c2   : float, Wolfe condition parameters
        alpha_min : float, minimum allowed step size
        max_iter  : int, maximum iterations
    
    Returns:
        alpha : float, step size satisfying Wolfe conditions
    """
    
    pk = -gk
    phi_0 = f(xk[0],xk[1]).sum().item()+f(xk[0],xk[1]).max().item()
    phi_prime_0 = torch.dot(gk, pk).item()

    assert phi_prime_0 < 0, "Negative gradient must be a descent direction."
    alpha = alpha_init

    for _ in range(max_iter):
        x_new = xk[0] + alpha * pk
        phi_alpha = f(x_new,xk[1]).sum().item()+f(x_new,xk[1]).max().item()
        grad_new = torch.autograd.functional.jacobian(f,(x_new,xk[1]))[0].squeeze(1)
        grad_new_steer = grad_new[f(x_new,xk[1]).argmax().item()].clamp(-grad_clamp,grad_clamp) 
        grad_new = grad_new.sum()+grad_new_steer
        phi_prime_alpha = torch.dot(grad_new, pk).item()
        
        # Wolfe conditions
        if phi_alpha > phi_0 + c1 * alpha * phi_prime_0:
            alpha *= 0.9  # reduce
        elif phi_prime_alpha < c2 * phi_prime_0:
            alpha *= 1.1  # increase slightly
        else:
            return alpha  # Wolfe satisfied

        if alpha < alpha_min:
            print("Step size below minimum threshold.")
            return alpha_min

    return alpha


def quadratic_loss(s, u, s_target,control_penalty=0.00001,dt=0.1,l=1):
    
    v,delta = u[0,:]
    x_target, y_target, theta_target = s_target[0,:]
    new_s = system_dynamics(s,u)
    
    new_x,new_y,new_theta = new_s[0],new_s[1],new_s[2,:]
    f = (x_target-new_x)**2+(y_target-new_y)**2+(theta_target-new_theta)**2+control_penalty*(v**2+delta**2)
    return f

def constraint_max_speed_upper_bound(v,v_max=30.0):
    return v-v_max
def constraint_max_speed_lower_bound(v,v_max=0.0):
    return -v-v_max
def constraint_max_acceleration_upper_bound(v,v_prev,a_max=3.0,dt=0.1):
    return (v-v_prev)/dt-a_max
def constraint_max_acceleration_lower_bound(v,v_prev,a_max=3.0,dt=0.1):
    return (-v+v_prev)/dt-a_max
def constraint_max_centrifugal_upper_bound(v,delta,l=1,a_centrifugal_max=10):
    return torch.pow(v,2)*torch.tan(delta)/l-a_centrifugal_max
def constraint_max_centrifugal_lower_bound(v,delta,l=1,a_centrifugal_max=10):
    return -torch.pow(v,2)*torch.tan(delta)/l-a_centrifugal_max
def constraint_max_steering_upper_bound(delta,delta_max=0.9):
    return delta-delta_max
def constraint_max_steering_lower_bound(delta,delta_max=0.9):
    return -delta-delta_max
def heuristic_turn_right(delta,delta_gap):
    return -delta*(1+torch.sign(delta_gap))
def heuristic_turn_left(delta,delta_gap):
    return delta*(1-torch.sign(delta_gap))

def update(constraint,control,fix=0.1):
    if constraint>1:
        return control + fix

def fix_speed(speed,prev_speed):
    constraints = constraints_vector_speed(speed,prev_speed)
    fixes = [0.05,-0.05,-0.05,0.05]
    while (constraints>1).any().item():
        speed = update(constraints.max().item(),speed,fix=fixes[constraints.argmax().item()])
        constraints = constraints_vector_speed(speed,prev_speed)
    
    return speed
def fix_steering(steering,delta_gap):
    constraints = constraints_vector_steering(steering,delta_gap)
    fixes = [0.05,-0.05,0.05,-0.05]
    
    while (constraints>1).any().item():
        steering = update(constraints.max().item(),steering,fix=fixes[constraints.argmax().item()])
        constraints = constraints_vector_steering(steering,delta_gap)
    return steering


def constraints_vector(u_curr,u_prev):
        # pack all six constraints into one tensor to avoid many small host syncs
        c = torch.stack((
            torch.exp(0.9*constraint_max_speed_lower_bound(u_curr[0,0])),
            torch.exp(0.9*constraint_max_speed_upper_bound(u_curr[0,0])),
            torch.exp(0.9*constraint_max_acceleration_upper_bound(u_curr[0,0], u_prev[0,0])),
            torch.exp(0.9*constraint_max_acceleration_lower_bound(u_curr[0,0], u_prev[0,0])),
            torch.exp(0.9*constraint_max_centrifugal_lower_bound(u_curr[0,0], u_curr[0,1])),
            torch.exp(0.9*constraint_max_centrifugal_upper_bound(u_curr[0,0], u_curr[0,1])),
            torch.exp(0.9*constraint_max_steering_lower_bound(u_curr[0,1])),
            torch.exp(0.9*constraint_max_steering_upper_bound(u_curr[0,1])),
        ))
        return c

def constraints_vector_speed(speed_curr,speed_prev):
        # pack all six constraints into one tensor to avoid many small host syncs
        c = torch.stack((
            torch.exp(1.5*constraint_max_speed_lower_bound(speed_curr)),
            torch.exp(1.5*constraint_max_speed_upper_bound(speed_curr)),
            torch.exp(1.5*constraint_max_acceleration_upper_bound(speed_curr, speed_prev)),
            torch.exp(1.5*constraint_max_acceleration_lower_bound(speed_curr, speed_prev)),
        ))
        return c

def constraints_vector_steering(steering,delta_gap):
        # pack all six constraints into one tensor to avoid many small host syncs
        c = torch.stack((
            torch.exp(1.5*constraint_max_steering_lower_bound(steering)),
            torch.exp(1.5*constraint_max_steering_upper_bound(steering)),
            torch.exp(1.5*heuristic_turn_right(steering,delta_gap)),
            torch.exp(1.5*heuristic_turn_left(steering,delta_gap)),
        ))
        return c

obj_jac_mod = torch.jit.load('outputs/obj_jac.pt')
cons_mod = torch.jit.load('outputs/constr_jac.pt')

obj_mod = torch.jit.load('outputs/obj.pt')

# Training setup
n_steps = 2
sympy_n_steps = 3
n_steps_alignment = 5
step_size_alignment = 0.1
grad_clamp = 1
num_constraints = 8

state_dim, hidden_dim, output_dim = 3, 128, 2  # 2D state (position, velocity), control is 1D

rnn_controller = ResidualRNNController(state_dim, hidden_dim, output_dim,device=device)
if os.path.isfile('outputs/weights_rnn.pt'):
    rnn_controller.load_state_dict(torch.load('outputs/weights_rnn.pt'))

optimizer = optim.Adam(rnn_controller.parameters(), lr=0.001)
torch.autograd.set_detect_anomaly(True)
torch.set_default_device(device)
x_min_change = 100
x_max_change = 200
y_min_change = 150
y_max_change = 300
delta_min_change = math.pi/5
delta_max_change = math.pi/4

mse =  torch.nn.MSELoss()
for init_condition in range(3000):
    x0 = torch.tensor(0).unsqueeze(0)
    y0 = torch.tensor(0).unsqueeze(0)
    v0 = torch.tensor(0).unsqueeze(0)
    sigma0 = torch.tensor(0).unsqueeze(0)
    delta0 = torch.tensor(0).unsqueeze(0)
    x_init = torch.stack([x0,y0,v0]).transpose(0,1)
    x_target = get_random_target_state(x_init.transpose(0,1),x_min_change,x_max_change,y_min_change,y_max_change,delta_min_change,delta_max_change).transpose(0,1).to(device)
    x_target = torch.abs(x_target)
    
    for trajectory_optimization in range(100):
        x = x_init.to(device).float().requires_grad_(True)
        x_target = x_target.float().requires_grad_(True)
        u_prev = torch.zeros(2).unsqueeze(0).to(device).float().requires_grad_(True)
        h = torch.ones(1, 1, output_dim).to(device).float().requires_grad_(True)  # Initial hidden state for RNN     
        param_list = [p for p in rnn_controller.parameters() if p.requires_grad]        
        loss = 0
        for epoch in range(5):
            # prev_h = h
            # for i in range(500):
            #     pred, h = rnn_controller(x-x_target, prev_h)
            #     pred= pred.squeeze(0)
            #     u = u_prev + pred.squeeze(0)
            #     c = constraints_vector(u,u_prev)
            #     constraint_loss = c.sum().clone()
            #     constraint_loss.backward(retain_graph=True)
            #     norm = []
            #     for param in param_list:
            #         norm.append(param.grad.norm().item())
            #     print(max(norm))
            #     torch.nn.utils.clip_grad_norm_(rnn_controller.parameters(), max_norm=1.0)
            #     optimizer.step()
            #     optimizer.zero_grad()
            # input('we are yippies')

            pred, h = rnn_controller(x-x_target, h)
            pred= pred.squeeze(0)
            u = u_prev + pred.squeeze(0)
            u_init = u.clone().detach()
            constraints = constraints_vector_speed(u[0,0],u_prev[0,0])
            fixes = [0.05,-0.05,-0.05,0.05]
            
            while (constraints>1).any().item():
                u[0,0]= u[0,0] + fixes[constraints.argmax().item()]
                constraints = constraints_vector_speed(u[0,0],u_prev[0,0])
            
            constraints = constraints_vector_steering(u[0,1],x_target[0,2]-x[0,2])
            fixes = [0.05,-0.05,0.05,-0.05]
            
            while (constraints>1).any().item():
                u[0,1]= u[0,1] + fixes[constraints.argmax().item()]
                constraints = constraints_vector_steering(u[0,1],x_target[0,2]-x[0,2])
            u_final = u.clone()
            loss_u = mse(u_init,u_final)
            # print(u_init,u_final)
            # print('loss u',loss_u,'epoch is:',epoch)
            constraint_loss = constraints_vector(u,u_prev).sum()
            
            # c= constraints_vector(u,u_prev)
            # while torch.any(c>1.2).item():
                
            #     cons_update_steer = torch.autograd.functional.jacobian(constraints_vector,(u,u_prev))[0].\
            #         squeeze(1)
            #     cons_update = cons_update_steer.sum(0).clamp(-grad_clamp,grad_clamp)
        
            #     cons_update_steer = cons_update_steer[c.argmax().item()].clamp(-grad_clamp,grad_clamp) 
                
            #     step_size_alignment = wolfe_line_search_torch(
            #         constraints_vector,
            #         (u,
            #         u_prev),
            #         cons_update,
            #         step_size_alignment
            #     )
            #     cons_update = cons_update_steer+cons_update
            #     u=  u-step_size_alignment*(cons_update.clamp(-grad_clamp,grad_clamp))
            #     c = constraints_vector(u,u_prev) 
            # print(u)
            # input('got out of gradient-based reduction')
            # print(c.norm())
            
            # print(c.norm())
            # input('yipi')
            # step_size_alignment = 0.01
            # pred, h = rnn_controller(x-x_target, h)
            # pred= pred.squeeze(0)
            
            # u = u_prev + pred.squeeze(0)
            # u_init = u.clone()

            # c = constraints_vector(u,u_prev)
            
            
            
            # u[0,1]=fix_steering(u[0,1],x_target[0,2]-x[0,2])
            

            # while torch.any(c>1.2).item():
                
            #     cons_update_steer = torch.autograd.functional.jacobian(constraints_vector,(u,u_prev))[0].\
            #         squeeze(1)
            #     cons_update = cons_update_steer.sum(0).clamp(-grad_clamp,grad_clamp)
        
            #     cons_update_steer = cons_update_steer[c.argmax().item()].clamp(-grad_clamp,grad_clamp) 
                
            #     step_size_alignment = wolfe_line_search_torch(
            #         constraints_vector,
            #         (u,
            #         u_prev),
            #         cons_update,
            #         step_size_alignment
            #     )
            #     cons_update = cons_update_steer+cons_update
            #     u=  u-step_size_alignment*(cons_update.clamp(-grad_clamp,grad_clamp))
            #     c = constraints_vector(u,u_prev)                        
            
            # for i in range(n_steps_alignment):
                
            #     loss_func_update = torch.autograd.functional.jacobian(quadratic_loss,(x.float().requires_grad_(True),\
            #                                                                             u.float().requires_grad_(True),\
            #                                                                             x_target.float().requires_grad_(True)))[1]
                
            #     u=  u-0.1*(loss_func_update[:,0].clamp(-grad_clamp,grad_clamp))
            
            # u[0,0]=fix_speed(u[0,0],u_prev[0,0])
            # u[0,1]=fix_steering(u[0,1],x_target[0,2]-x[0,2])    
            
            # c = constraints_vector(u,u_prev)
            
            # while torch.any(c>1.2).item():
            #     cons_update_steer = torch.autograd.functional.jacobian(constraints_vector,(u,u_prev))[0].\
            #         squeeze(1)
            #     cons_update = cons_update_steer.sum(0).clamp(-grad_clamp,grad_clamp)
            #     cons_update_steer = cons_update_steer[c.argmax().item()].clamp(-grad_clamp,grad_clamp) 
            #     step_size_alignment = wolfe_line_search_torch(
            #             constraints_vector,
            #             (u,
            #             u_prev),
            #             cons_update,
            #             step_size_alignment
            #         )
            #     cons_update = 2*cons_update_steer+cons_update
            #     u=  u-step_size_alignment*(cons_update.clamp(-grad_clamp,grad_clamp))
            #     c = constraints_vector(u,u_prev)
                
            # u_final = u.clone()
            
            step_loss = quadratic_loss(x,u,x_target)
            # print(u_final,u_init)
            # step_loss = mse(u_init,u_final)
            
            # loss = loss + step_loss + constraint_loss + loss_u
            loss = loss + loss_u
                
                #   print(torch.autograd.grad(u_init.sum(),param,retain_graph=True)[0].sum())
            #     input('yipi')
            #     print(torch.autograd.grad(mse(u_init,u_final),param,retain_graph=True)[0].sum())
            #     input('yipi')
            # input('yipi')
            
            new_x = system_dynamics(x= x, u=u).T.to(device)
            u_prev = u
            x = new_x
        # if trajectory_optimization % 2 == 0:
        print('constraint',constraint_loss)
        # print('step loss',step_loss)
        print('u loss',loss)
        print('control is',u)
        input('yipi')
       
        loss.backward()
        for param in rnn_controller.named_parameters():
            print(param[0],param[1].grad.norm().item())
        input('hipi')
        torch.nn.utils.clip_grad_norm_(rnn_controller.parameters(), max_norm=1.0)
        
        optimizer.step()
        optimizer.zero_grad()
        input('did step')
        # trajectory_loss = quadratic_loss(x,u,x_target).to(device)
        # if epoch % 5 == 0:
        #     print(u,loss,epoch,x,x_target)
        # # loss = loss + trajectory_loss

        # print(x,x_target)
        
        # print('performed step',loss)
        torch.save(rnn_controller.state_dict(), './outputs/weights_rnn.pt')
