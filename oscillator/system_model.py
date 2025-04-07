import torch
from mpc.mpc_nn import ResidualRNNController

# Define the system dynamics function (e.g., a simple mass-spring-damper system)
def system_dynamics(x, u, dt=0.01,mass=1.0,damping=0,stiffness=1.0):
    """ x = [position, velocity], u = control input """
    position, velocity = x
    
    # Numerically estimate position and velocity
    acceleration = (u - damping * velocity - stiffness * position) / mass
    next_velocity = velocity + acceleration * dt
    next_velocity = next_velocity.squeeze()
    next_position = position + velocity * dt
    return torch.stack([next_position, next_velocity])

def position_upper_bound(x,position_bound,mass=1,stiffness=1):
    position,velocity = x
    return mass*position_bound-mass*velocity+stiffness*position

def position_lower_bound(x,position_bound,mass=1,stiffness=1):
    position,velocity = x
    return -mass*position_bound-mass*velocity+stiffness*position

def control_upper_bound(u,control_bound):
    return -u+control_bound
def control_lower_bound(u,control_bound):
    return -u-control_bound

def velocity_positive(x):
    _,velocity = x
    return torch.relu(velocity)+(velocity-torch.relu(velocity))

# Define cost function
def cost_function_state_following(x,x_target,lambda_x=1, u=torch.tensor([0]),u_target=torch.tensor([0]),lambda_u=torch.tensor([0])):
    """ Penalize deviation from zero state and large control inputs """
    return lambda_x*torch.sum((x-x_target)**2) + lambda_u*torch.sum((u-u_target)**2)

def cost_states_integral(x,x_prev):
    return torch.sum((x-x_prev)**2)

class ConstrainedResidualRNNController(ResidualRNNController):
    def __init__(self, state_dim, hidden_dim, output_dim):
        super().__init__(state_dim, hidden_dim, output_dim)
        
    def bound_condition(self,x):
        return torch.relu(x)

    def smooth_bound(self,x, bound):
        return bound * torch.tanh(x / bound)

    def bound_jump(self,x,bound_lower=None,bound_upper=None):
        if bound_lower is not None:
            added_lower = self.bound_condition(x+bound_lower)
        
        if bound_upper is not None:
            added_upper = self.bound_condition(-x+bound_upper)
        
        if bound_lower is not None:
            x = torch.where(added_lower == 0, self.smooth_bound(x, -bound_lower), x)  # Set to lower bound
        
        if bound_upper is not None:
            x = torch.where(added_upper == 0, self.smooth_bound(x, bound_upper), x)  # Set to upper bound
        
        return x

    def get_constraints_weights(self,constraint):
        return torch.softmax(constraint,dim=0)

    def forward_constrain(self,u,constraints_upper,constraints_lower,beta=5):
        
        weights_contraints_upper = self.get_constraints_weights(constraints_upper,beta)
        weights_contraints_lower = self.get_constraints_weights(-constraints_lower,beta)
        
        u = u + weights_contraints_lower*(constraints_lower-u) + weights_contraints_upper*(constraints_upper-u)
