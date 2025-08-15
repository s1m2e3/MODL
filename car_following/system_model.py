# Define the system dynamics function (e.g., a simple mass-spring-damper system)
import torch



class BaseSympyModule(torch.nn.Module):
    def __init__(self, f,  default_v1=0.0, default_v2=0.0, default_delta1=0.0, default_delta2=0.0):
        super().__init__()
        self.f = f
        # register defaults as buffers so scripting can see them
        self.register_buffer('default_v1', torch.tensor(default_v1))
        self.register_buffer('default_v2', torch.tensor(default_v2))
        self.register_buffer('default_delta1', torch.tensor(default_delta1))
        self.register_buffer('default_delta2', torch.tensor(default_delta2))

    def forward(self,
                x0:   torch.Tensor,
                y0:   torch.Tensor,
                theta0:   torch.Tensor,
                v0:   torch.Tensor,
                delta0:   torch.Tensor,
                
               ) -> torch.Tensor:
        # now call the positional lambdified function
        device = x0.device
        nested = self.f(x0.to(device),y0.to(device),theta0.to(device),v0.to(device),delta0.to(device))
        # nested is a list-of-lists-of (Tensor OR int)
        rows = []
        for row in nested:
            # convert any int/float → Tensor on the right device/dtype
            trow = [
                e if torch.is_tensor(e)
                  else torch.as_tensor(e, dtype=x0.dtype, device=x0.device)
                for e in row
            ]
            
            rows.append(torch.stack(trow, dim=0))
        return torch.stack(rows, dim=-2).to(device)

class SympyFuncModule(BaseSympyModule):
    def __init__(self, f, default_x_target=0.0, default_y_target=0.0, default_theta_target=0.0,**kwargs):
        super().__init__(f, **kwargs)
        self.register_buffer('default_x_target', torch.tensor(default_x_target))
        self.register_buffer('default_y_target', torch.tensor(default_y_target))
        self.register_buffer('default_theta_target', torch.tensor(default_theta_target))

    def forward(self,
                x0:   torch.Tensor,
                y0:   torch.Tensor,
                theta0:   torch.Tensor,
                v0:   torch.Tensor,
                delta0:   torch.Tensor,
                x_target:   torch.Tensor,
                y_target:   torch.Tensor,
                theta_target:   torch.Tensor,
                v1:   torch.Tensor = None,
                v2:   torch.Tensor = None,
                delta1:   torch.Tensor = None,
                delta2:   torch.Tensor = None,
                
               ) -> torch.Tensor:
        device = x0.device
        # if y or z weren’t passed, fill from the buffers (and match x’s shape)            
        if v1 is None:
            v1 = self.default_v1.expand_as(x0)
        if v2 is None:
            v2 = self.default_v2.expand_as(x0)
        if delta1 is None:
            delta1 = self.default_delta1.expand_as(x0)
        if delta2 is None:
            delta2 = self.default_delta2.expand_as(x0)
        # now call the positional lambdified function
        nested = self.f(x0.to(device),y0.to(device),theta0.to(device),v0.to(device),delta0.to(device),
                        x_target.to(device),y_target.to(device),theta_target.to(device),v1.to(device),
                        v2.to(device),delta1.to(device),delta2.to(device))
        
        if torch.is_tensor(nested):
            return nested.to(device)
        # nested is a list-of-lists-of (Tensor OR int)
        rows = []
        for row in nested:
            # convert any int/float → Tensor on the right device/dtype
            trow = [
                e if torch.is_tensor(e)
                  else torch.as_tensor(e, dtype=x0.dtype, device=x0.device)
                for e in row
            ]
            
            rows.append(torch.stack(trow, dim=0))
        return torch.stack(rows, dim=-2).to(device)
        
class SympyConsModule(BaseSympyModule):
    def __init__(self, f, default_x10=0.0, default_y10=0.0,**kwargs):
        super().__init__(f, **kwargs)
        self.register_buffer('default_x10', torch.tensor(default_x10))
        self.register_buffer('default_y10', torch.tensor(default_y10))

        
    def forward(self,
                x0:   torch.Tensor,
                y0:   torch.Tensor,
                theta0:   torch.Tensor,
                v0:   torch.Tensor,
                delta0:   torch.Tensor,
                x10:   torch.Tensor = None,
                y10:   torch.Tensor = None,
                v1:   torch.Tensor = None,
                v2:   torch.Tensor = None,
                delta1:   torch.Tensor = None,
                delta2:   torch.Tensor = None,

               ) -> torch.Tensor:
        device = x0.device
        # if y or z weren’t passed, fill from the buffers (and match x’s shape)
        if x10 is None:
            x10 = self.default_x10.expand_as(x0)
        if y10 is None:
            y10 = self.default_y10.expand_as(x0)
        if v1 is None:
            v1 = self.default_v1.expand_as(x0)
        if v2 is None:
            v2 = self.default_v2.expand_as(x0)
        if delta1 is None:
            delta1 = self.default_delta1.expand_as(x0)
        if delta2 is None:
            delta2 = self.default_delta2.expand_as(x0)
        # now call the positional lambdified function
        nested = self.f(x0.to(device),y0.to(device),theta0.to(device),v0.to(device),delta0.to(device),
                        x10.to(device),y10.to(device),v1.to(device),v2.to(device),delta1.to(device),delta2.to(device))
        # nested is a list-of-lists-of (Tensor OR int)
        rows = []
        for row in nested:
            # convert any int/float → Tensor on the right device/dtype
            trow = [
                e if torch.is_tensor(e)
                  else torch.as_tensor(e, dtype=x0.dtype, device=x0.device)
                for e in row
            ]
            rows.append(torch.stack(trow, dim=0))
        return torch.stack(rows, dim=-2).to(device)

def quadratic_loss(s, u, s_target,control_penalty=0.01,dt=0.1,l=1):
    
    v,delta = u[0,:]
    x_target, y_target, theta_target = s_target[0,:]
    new_s = system_dynamics(s,u)
    
    new_x,new_y,new_theta = new_s[0],new_s[1],new_s[2,:]
    f = (x_target-new_x)**2+(y_target-new_y)**2+(theta_target-new_theta)**2+control_penalty*(v**2+delta**2*100)
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
            torch.exp(0.005*constraint_max_speed_lower_bound(u_curr[0,0])),
            torch.exp(0.005*constraint_max_speed_upper_bound(u_curr[0,0])),
            torch.exp(0.005*constraint_max_acceleration_upper_bound(u_curr[0,0], u_prev[0,0])),
            torch.exp(0.005*constraint_max_acceleration_lower_bound(u_curr[0,0], u_prev[0,0])),
            torch.exp(0.005*constraint_max_centrifugal_lower_bound(u_curr[0,0], u_curr[0,1])),
            torch.exp(0.005*constraint_max_centrifugal_upper_bound(u_curr[0,0], u_curr[0,1])),
            torch.exp(0.005*constraint_max_steering_lower_bound(u_curr[0,1])),
            torch.exp(0.005*constraint_max_steering_upper_bound(u_curr[0,1])),
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
            # torch.exp(1.5*heuristic_turn_right(steering,delta_gap)),
            # torch.exp(1.5*heuristic_turn_left(steering,delta_gap)),
        ))
        return c



def system_dynamics(x, u, dt=0.1,l=1.0):
    """ x = [position_x, position_y,velocity, angle,stering_angle], u = [acceleration,steering_angle_rate] """
    
    position_x, position_y, angle = x[0,:]
    speed, steering = u[0,:]
    
    position_x = position_x + speed*torch.cos(angle) * dt
    position_y = position_y + speed*torch.sin(angle) * dt
    angle = angle + speed*torch.tan(steering)/l * dt
    
    return torch.stack([position_x, position_y, angle]).unsqueeze(1)


def get_random_target_state(x,max_change_x,min_change_x,max_change_y,min_change_y,min_change_angle,max_change_angle):
    position_x, position_y, angle = x

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
    
    new_position_x = position_x + delta_x_max
    new_position_y = position_y + delta_y_max
    new_angle = torch.atan(new_position_x/new_position_y)
    
    return torch.stack([new_position_x,new_position_y,new_angle])
    
def get_inertial_target_state(x,target):
    return x-target

