# Define the system dynamics function (e.g., a simple mass-spring-damper system)
import torch

# Define cost function
def cost_function_state_following(x,x_target,lambda_x=1, u=torch.tensor([0]),u_target=torch.tensor([0]),lambda_u=torch.tensor([0])):
    """ Penalize deviation from zero state and large control inputs """
    return lambda_x*torch.sum((x-x_target)**2) + lambda_u*torch.sum((u-u_target)**2)

def binary_angle_change_positive(x,x_target):
    '''If the difference in angle is greater than zero, the change in angle is greater than zero, 
    and if the difference in angle is less than zero, the change in angle is less than zero'''
        
    _, _, _, angle_ego, _ = x
    _, _, _, angle_target, _ = x_target
    
    return angle_target-angle_ego

def binary_angle_change_negative(x,x_target):
    '''If the difference in angle is greater than zero, the change in angle is greater than zero, 
    and if the difference in angle is less than zero, the change in angle is less than zero'''

    _, _,_, angle_ego, _ = x
    _, _, _, angle_target, _ = x_target
    
    return -(angle_target-angle_ego)

def angle_change_positive(x,u,binary_angle_change_positive,l=1.0,big_m=1.0):
    _, _, velocity, _, steering_angle = x
    acceleration, _ = u
    
    angle_change_positive = (torch.atan((big_m*(1-binary_angle_change_positive)*l)/(velocity+acceleration))-steering_angle)

    return angle_change_positive

def angle_change_negative(x,u,binary_angle_change_negative,l=1.0,big_m=1.0):
    _, _, velocity, _, steering_angle = x
    acceleration, _ = u
    
    angle_change_negative = (torch.atan(-(big_m*(1-binary_angle_change_negative)*l)/(velocity+acceleration))-steering_angle)

    return angle_change_negative

def steering_angle_rate_upper_bound(u,steering_angle_rate_bound):
    _, steering_angle_rate = u
    return -steering_angle_rate_bound


def steering_angle_rate_lower_bound(u,steering_angle_rate_bound):
    _, steering_angle_rate = u
    return steering_angle_rate_bound

def acceleration_upper_bound(u,acceleration_bound):
    acceleration, _ = u
    return acceleration_bound

def acceleration_lower_bound(u,acceleration_bound):
    acceleration,_ = u
    return -acceleration_bound

def system_dynamics(x, u, dt=0.01,l=1.0):
    """ x = [position_x, position_y,velocity, angle,stering_angle], u = [acceleration,steering_angle_rate] """
    
    position_x, position_y, velocity, angle, steering_angle = x
    acceleration, steering_angle_rate = u
    

    position_x = position_x + velocity*torch.cos(angle) * dt
    position_y = position_y + velocity*torch.sin(angle) * dt
    angle = angle + velocity*torch.tan(steering_angle)/l * dt
    velocity = velocity + acceleration*dt
    steering_angle = steering_angle + steering_angle_rate * dt
    
    return torch.stack([position_x, position_y, velocity, angle, steering_angle]).unsqueeze(0)


def get_random_target_state(x,max_change_x,min_change_x,max_change_y,min_change_y,min_change_angle,max_change_angle):
    position_x, position_y,_, angle, _ = x

    sign_x = 1 if torch.rand(1).item() < 0.5 else -1
    sign_y = 1 if torch.rand(1).item() < 0.5 else -1
    sign_angle = 1 if torch.rand(1).item() < 0.5 else -1
    # curve = torch.rand(1).item() < 0.01
    curve = True
    delta_x_min = min_change_x * sign_x
    delta_x_max = max_change_x*torch.rand(1) * sign_x
    delta_y_min = min_change_y * sign_y
    delta_y_max = max_change_y*torch.rand(1) * sign_y
    delta_angle_min = min_change_angle * sign_angle
    delta_angle_max = max_change_angle*torch.rand(1) * sign_angle 

    new_position_x = position_x + min(delta_x_min,delta_x_max)
    new_position_y = position_y + min(delta_y_min,delta_y_max)
    new_angle = angle + min(delta_angle_min,delta_angle_max) if curve else angle
    return torch.stack([new_position_x,new_position_y,torch.tensor([0]),new_angle,torch.tensor([0])])
    
def get_inertial_target_state(x,target):
    return x-target

