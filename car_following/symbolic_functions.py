import sympy as sp
import numpy as np

def generate_gaussian_term(f,f_0,Sigma):
    return sp.exp(-1/2*sp.Transpose(f-f_0)*Sigma*(f-f_0))
def generate_penalty_term(g,penalty_term=1):
    return sp.exp(g*penalty_term)

def generate_taylor_expansion(g,g_grad,f,f_0):
    g_at_f_0 = g.subs(dict(zip(f, f_0)))
    g_grad_at_f_0 = g_grad.subs(dict(zip(f, f_0)))
    delta_f = f - f_0
    return g_at_f_0 + (g_grad_at_f_0*delta_f)

def system_dynamics(s, u, dt=0.1,l=1.0):
    x,y,theta = s
    v,delta = u
    return sp.Matrix([x + v*sp.cos(theta) * dt, y + v*sp.sin(theta) * dt, theta + v*sp.tan(delta)/l * dt])

def quadratic_loss(s, u, s_target,control_penalty=0.00001):
    x,y,theta = s
    v,delta = u
    x_target, y_target, theta_target = s_target
    f = (x_target-x)**2+(y_target-y)**2+(theta_target-theta)**2+control_penalty*(v**2+delta**2)
    return f

def constraint_max_speed_upper_bound(v,v_max=30.0):
    return v-v_max
def constraint_max_speed_lower_bound(v,v_max=0.0):
    return -v-v_max
def constraint_max_steering_upper_bound(delta,delta_max=0.9):
    return delta-delta_max
def constraint_max_steering_lower_bound(delta,delta_max=0.9):
    return -delta-delta_max
def constraint_max_acceleration_upper_bound(v,v_prev,a_max=3.0,dt=0.1):
    return (v-v_prev)/dt-a_max
def constraint_max_acceleration_lower_bound(v,v_prev,a_max=3.0,dt=0.1):
    return (-v+v_prev)/dt-a_max
def constraint_max_centrifugal_upper_bound(v,delta,l=1,a_centrifugal_max=1):
    return sp.Pow(v,2)*sp.tan(delta)/l-a_centrifugal_max
def constraint_max_centrifugal_lower_bound(v,delta,l=1,a_centrifugal_max=1):
    return -sp.Pow(v,2)*sp.tan(delta)/l-a_centrifugal_max
def constraint_vehicles_min_distance(x1,y1,x2,y2,d_min=1.5):
    return sp.Pow(d_min,2)-sp.Pow(x1-x2,2)-sp.Pow(y1-y2,2)
