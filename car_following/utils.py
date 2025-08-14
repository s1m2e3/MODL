import torch
import dill
import argparse
import json

class CommandLineArgs:
    def __init__(self):
        self.args = self.parse_args()

    def parse_args(self):
        parser = argparse.ArgumentParser(description='Process command line arguments.')
        # Add arguments as needesd, for example:
        parser.add_argument('--weights_name', type=str, default='weights_rnn')
        parser.add_argument('--n_steps_alignment', type=int, default=10)
        parser.add_argument('--step_size_alignment', type=float, default=0.1)
        parser.add_argument('--grad_clamp', type=int, default=1)
        parser.add_argument('--state_dim', type=int, default=3)
        parser.add_argument('--hidden_dim', type=int, default=512)
        parser.add_argument('--output_dim', type=int, default=2)
        parser.add_argument('--trajectory_optimization_epochs', type=int, default=100)
        parser.add_argument('--n_trajectory_steps', type=int, default=10)
        parser.add_argument('--n_init_conditions', type=int, default=10)

        return parser.parse_args()

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



def store_json(filename, data):
    """Store data in a JSON file.

    Args:
        filename (str): The path to the file in which to store the data.
        data (dict): The data to store in the file.
    """
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
