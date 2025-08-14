import sympy
import numpy as np 
import dill
import torch
dill.settings['recurse'] = True
from symbolic_functions import system_dynamics,quadratic_loss,generate_penalty_term\
    ,constraint_max_acceleration_lower_bound,constraint_max_acceleration_upper_bound,constraint_max_centrifugal_upper_bound,\
        constraint_max_centrifugal_lower_bound, constraint_max_speed_lower_bound,constraint_max_speed_upper_bound,constraint_vehicles_min_distance,\
        constraint_max_steering_lower_bound,constraint_max_steering_upper_bound\

num_steps = 4
num_constraints = 8
dt = 0.1
l = 1
control_penalty = 1e-3
v_max = 20
steering_max = 0.9
step_size = 0.01
n_steps_update = 5

x_vec = sympy.Matrix([])
f_vec = sympy.Matrix([])
u_vec = sympy.Matrix([])
x0 = sympy.Symbol('x0')
v0 = sympy.Symbol('v0')
y0 = sympy.Symbol('y0')

x10 = sympy.Symbol('x10')
y10 = sympy.Symbol('y10')

theta0 = sympy.Symbol('theta0')
delta0 = sympy.Symbol('delta0')

x_target = sympy.Symbol('x_target')
y_target = sympy.Symbol('y_target')
theta_target = sympy.Symbol('theta_target')

x_vec = x_vec.col_join(sympy.Matrix([x0,y0,theta0]))
u_vec = u_vec.col_join(sympy.Matrix([v0,delta0]))
x_target_vec = sympy.Matrix([x_target,y_target,theta_target])
one_step_dynamics = system_dynamics(x_vec[:,-1],u_vec[:,-1],dt,l)
one_step_loss = quadratic_loss(x_vec[:,-1],u_vec[:,-1],x_target_vec)

f_loss_vec = sympy.Matrix([])
constraints_vec = sympy.Matrix([])

for i in range(1,num_steps):
    
    s_attach = system_dynamics(x_vec[:,-1],u_vec[:,-1],dt,l)
    distance_constraint = generate_penalty_term(constraint_vehicles_min_distance(x_vec[0,-1],x_vec[1,-1],x10,y10))
    
    x_vec = x_vec.row_join(s_attach)
    f_loss_vec = f_loss_vec.col_join(sympy.Matrix([quadratic_loss(x_vec[:,-1],u_vec[:,-1],x_target_vec,control_penalty=0.00001)]))
    
    max_speed_constraint_upper = generate_penalty_term(constraint_max_speed_upper_bound(u_vec[0,-1],v_max=v_max))
    max_speed_constraint_lower = generate_penalty_term(constraint_max_speed_lower_bound(u_vec[0,-1],v_max=0.0))
    
    max_steering_constraint_upper = generate_penalty_term(constraint_max_steering_upper_bound(u_vec[1,-1],delta_max=steering_max))
    max_steering_constraint_lower = generate_penalty_term(constraint_max_steering_lower_bound(u_vec[1,-1],delta_max=steering_max))

    max_centrifugal_upper = generate_penalty_term(constraint_max_centrifugal_upper_bound(u_vec[0,-1],u_vec[1,-1]))
    max_centrifugal_lower = generate_penalty_term(constraint_max_centrifugal_lower_bound(u_vec[0,-1],u_vec[1,-1]))
    
    if i >1:
        f_loss_vec[-1] = f_loss_vec[-1]+f_loss_vec[-2]
    if i != num_steps-1:
        ut = sympy.Symbol('v'+str(i))
        deltat = sympy.Symbol('delta'+str(i))
        u_vec = u_vec.row_join(sympy.Matrix([ut,deltat]))

    max_acceleration_constraint_upper = generate_penalty_term(constraint_max_acceleration_upper_bound(u_vec[0,-1],u_vec[0,-2]))
    max_acceleration_constraint_lower = generate_penalty_term(constraint_max_acceleration_lower_bound(u_vec[0,-1],u_vec[0,-2]))
    constraints_vec = constraints_vec.row_join(sympy.Matrix([max_speed_constraint_lower,
                                                            max_speed_constraint_upper,
                                                            -max_acceleration_constraint_lower,
                                                            max_acceleration_constraint_upper,
                                                            -max_centrifugal_lower,
                                                            max_centrifugal_upper,
                                                            -max_steering_constraint_lower,
                                                            max_steering_constraint_upper
                                                            ]))
    
v_list = u_vec[0,1:].tolist()[0]
delta_list = u_vec[1,1:].tolist()[0]

u_vec = u_vec.reshape((num_steps-1)*2,1)
x_vec = x_vec.reshape((num_steps)*3,1)

system_dynamics_states = [x0,y0,theta0,v0,delta0]


init_states_loss = [x0,y0,theta0,v0,delta0,x_target,y_target,theta_target]
init_states_loss.extend(v_list)
init_states_loss.extend(delta_list)

init_states_cons = [x0,y0,theta0,v0,delta0,x10,y10]
init_states_cons.extend(v_list)
init_states_cons.extend(delta_list)


torch_mapping = {
    'sin':    torch.sin,
    'cos':    torch.cos,
    'tan':    torch.tan,
    'exp':    torch.exp,
    'log':    torch.log,
    'Abs':    torch.abs,
    'Pow':    torch.pow,
    # …add anything else you used (sqrt, atan2, etc.)…
}

system_dynamics_lambified = sympy.lambdify(system_dynamics_states,one_step_dynamics.tolist(),modules=[torch_mapping])
system_dynamics_lambified.__module__ = __name__

f_loss_lambdified = sympy.lambdify(init_states_loss,one_step_loss,modules=[torch_mapping])
f_loss_lambdified.__module__ = __name__

f_loss_jac = f_loss_vec.jacobian(u_vec).tolist()
constraints_jac = constraints_vec.reshape(num_constraints*(num_steps-1),1).jacobian(u_vec).tolist()


f_loss_jac_lambdified = sympy.lambdify(init_states_loss,f_loss_jac,modules=[torch_mapping])
f_loss_jac_lambdified.__module__ = __name__

constraints_jac_lambdified = sympy.lambdify(init_states_cons,constraints_jac,modules=[torch_mapping])
constraints_jac_lambdified.__module__ = __name__

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

mod = BaseSympyModule(system_dynamics_lambified)
scripted = torch.jit.trace(mod, (torch.tensor(0),torch.tensor(2),torch.tensor(0),torch.tensor(1),torch.tensor(0.1)))
torch.jit.save(scripted, 'outputs/system_dynamics.pt')

mod = SympyFuncModule(f_loss_lambdified)
scripted = torch.jit.trace(mod, (torch.tensor(0),torch.tensor(2),torch.tensor(0),torch.tensor(1),torch.tensor(0.1),torch.tensor(2),torch.tensor(3),torch.tensor(1)))
torch.jit.save(scripted, 'outputs/obj.pt')

mod = SympyFuncModule(f_loss_jac_lambdified)
scripted = torch.jit.trace(mod, (torch.tensor(0),torch.tensor(2),torch.tensor(0),torch.tensor(1),torch.tensor(0.1),torch.tensor(2),torch.tensor(3),torch.tensor(1)))
torch.jit.save(scripted, 'outputs/obj_jac.pt')


mod = SympyConsModule(constraints_jac_lambdified)
scripted = torch.jit.trace(mod,(torch.tensor(0),torch.tensor(2),torch.tensor(0),torch.tensor(1),torch.tensor(0.1)))
torch.jit.save(scripted, 'outputs/constr_jac.pt')