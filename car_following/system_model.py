# Define the system dynamics function (e.g., a simple mass-spring-damper system)
import torch

# Define cost function
def cost_function_state_following(x,x_target,lambda_x=1, u=torch.tensor([0]),u_target=torch.tensor([0]),lambda_u=torch.tensor([0])):
    """ Penalize deviation from zero state and large control inputs """
    return lambda_x*torch.sum((x-x_target)**2) + lambda_u*torch.sum((u-u_target)**2)

def binary_angle_change_positive(x,x_target,dt=0.1,l=1,steepness=5):
    '''If the difference in angle is greater than zero, the change in angle is greater than zero, 
    and if the difference in angle is less than zero, the change in angle is less than zero'''
        
    _, _, velocity, angle_ego, delta = x
    velocity = torch.relu(velocity)
    _, _, _, angle_target, _ = x_target
    
    return steepness*(angle_target-(angle_ego+(velocity/l*torch.tan(delta))))

def binary_angle_change_negative(x,x_target):
    '''If the difference in angle is greater than zero, the change in angle is greater than zero, 
    and if the difference in angle is less than zero, the change in angle is less than zero'''

    _, _,_, angle_ego, _ = x
    _, _, _, angle_target, _ = x_target
    
    return -(angle_target-angle_ego)

def angle_change_positive_lower_bound(x,u,binary_angle_change_positive,l=1.0,big_m=10.0,dt=0.01):
    _, _, velocity, _, steering_angle = x
    acceleration, _ = u
    
    return -big_m*(1-binary_angle_change_positive)-steering_angle

def angle_change_positive_upper_bound(x_target,x,u,binary_angle_change_positive,l=1.0,big_m=10.0,dt=0.01):
    _, _,_, angle_target, _ = x_target
    _, _, velocity, angle, steering_angle = x
    velocity = torch.relu(velocity)
    acceleration, _ = u
    angle_delta = angle_target-angle
    inner_term = torch.atan(((angle_delta)*l*dt)/(velocity+1e-3))
    
    angle_change_positive = (inner_term-steering_angle+big_m*(1-binary_angle_change_positive))

    return angle_change_positive

def car_following_estimation(x,x_target,desired_headway,target_speed,accel_max,minimum_spacing=0.01):
    position_x, position_y, velocity, angle, _ = x
    position_x_target, position_y_target, _, angle_target, _ = x_target
    deceleration = accel_max/2
    
    position_modulus = (position_x**2+position_y**2)**(1/2)
    position_modulus_target = (position_x_target**2+position_y_target**2)**(1/2)
    delta_s = position_modulus_target-position_modulus
    delta_velocity = velocity
    ratio_velocity = velocity/target_speed
    s_star = minimum_spacing+velocity*desired_headway+velocity*delta_velocity/(2*(accel_max*deceleration)**(1/2))
    print(ratio_velocity,s_star/delta_s)
    return accel_max*(1-(ratio_velocity)**4-(s_star/delta_s)**2)

def car_following_upper_bound(car_following,binary_car_following,big_m=10.0):
    return car_following+big_m*binary_car_following

def car_following_lower_bound(car_following,binary_car_following,big_m=10.0):
    return car_following-big_m*(1-binary_car_following)


def binary_minimum_acceleration(x,speed_target,ratio_desired_speed=0.5):
    _, _, velocity, _, _ = x
    
    return ratio_desired_speed-velocity/speed_target

def steering_like_previous_upper_bound(steering_angle_prev,binary_angle_change_positive,big_m=10.0):
    return steering_angle_prev+big_m*(binary_angle_change_positive)

def steering_like_previous_lower_bound(steering_angle_prev,binary_angle_change_positive,big_m=10.0):
    return steering_angle_prev-big_m*(1-binary_angle_change_positive)

def minimum_acceleration(binary_minimum_acceleration,minimum_acceleration=0.3,big_m=10.0):
    return minimum_acceleration-big_m*(1-binary_minimum_acceleration)

def steering_angle_difference_upper_bound(steering_rate_prev,control_bound):
    return steering_rate_prev+control_bound
def steering_angle_difference_lower_bound(steering_rate_prev,control_bound):
    return steering_rate_prev-control_bound
def acceleration_difference_upper_bound(acceleration_prev,control_bound):
    return acceleration_prev+control_bound
def acceleration_difference_lower_bound(acceleration_prev,control_bound):
    return acceleration_prev-control_bound

def angle_change_negative_upper_bound(x,u,binary_angle_change_positive,l=1.0,big_m=10.0,dt=0.01):
    _, _, velocity, _, steering_angle = x
    velocity = torch.relu(velocity)
    
    return big_m*(binary_angle_change_positive)-steering_angle

def angle_change_negative_lower_bound(x_target,x,u,binary_angle_change_positive,l=1.0,big_m=10.0,dt=0.01):
    _, _,_, angle_target, _ = x_target
    _, _, velocity, angle, steering_angle = x
    velocity = torch.relu(velocity)
    acceleration, _ = u
    angle_delta = angle_target-angle
    inner_term = torch.atan(((angle_delta)*l*dt)/(velocity+1e-3))
    
    angle_change_negative = (inner_term-steering_angle)-big_m*(binary_angle_change_positive)

    return angle_change_negative


def steering_angle_rate_upper_bound(u,steering_angle_rate_bound):
    _, steering_angle_rate = u
    return steering_angle_rate_bound


def steering_angle_rate_lower_bound(u,steering_angle_rate_bound):
    _, steering_angle_rate = u
    return -steering_angle_rate_bound

def acceleration_upper_bound(u,acceleration_bound):
    acceleration, _ = u
    return acceleration_bound

def acceleration_lower_bound(u,acceleration_bound):
    acceleration,_ = u
    return -acceleration_bound

def system_dynamics(x, u, dt=0.1,l=1.0):
    """ x = [position_x, position_y,velocity, angle,stering_angle], u = [acceleration,steering_angle_rate] """
    
    position_x, position_y, velocity, angle, steering_angle = x
    acceleration, steering_angle_rate = u
    if abs(acceleration.item()) < 0.01:
        acceleration = torch.relu(acceleration)
    position_x = position_x + velocity*torch.sin(angle) * dt
    position_y = position_y + velocity*torch.cos(angle) * dt
    angle = angle + velocity*torch.tan(steering_angle)/l * dt
    velocity = velocity + acceleration*dt
    steering_angle = steering_angle + steering_angle_rate * dt
    
    return torch.stack([position_x, position_y, velocity, angle, steering_angle]).unsqueeze(0)


def get_random_target_state(x,max_change_x,min_change_x,max_change_y,min_change_y,min_change_angle,max_change_angle):
    position_x, position_y,_, angle, _ = x

    sign_angle = 1 if torch.rand(1).item() < 0.5 else -1
    curve = True
    delta_angle_min = min_change_angle * sign_angle
    new_angle = angle + min(delta_angle_min,0.5) if curve else angle
    sign_x = torch.sign(new_angle)
    sign_y = torch.sign(torch.tensor(1))
    delta_x_min = min_change_x *sign_x
    delta_x_max = max_change_x*torch.rand(1) * sign_x
    delta_y_min = min_change_y *sign_y
    delta_y_max = max_change_y*torch.rand(1) * sign_y
    

    new_position_x = position_x + max(delta_x_min,delta_x_max)
    new_position_y = position_y + max(delta_y_min,delta_y_max)
    new_angle = torch.atan(new_position_x/new_position_y)
    
    return torch.stack([new_position_x,new_position_y,torch.tensor([0]),new_angle,torch.tensor([0])])
    
def get_inertial_target_state(x,target):
    return x-target

