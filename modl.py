import torch
from torchmin import minimize_constr
def collect_gradients(loss, model, subset=None):
    """
    Collect gradients of the loss with respect to a subset of parameters.

    Parameters
    ----------
    loss : torch.Tensor
        The computed loss for which gradients are calculated.
    model : torch.nn.Module
        The model to compute gradients for.
    subset : list, optional
        A list of parameter names for which to collect gradients. If None, gradients for all parameters are collected.

    Returns
    -------
    gradients : dict
        A dictionary where the keys are the parameter names and the values are the gradients.
    """
    torch.autograd.set_detect_anomaly(True)
    gradients = {}
    # If subset is None, collect gradients for all parameters
    subset = set(subset) if subset is not None else None
    for name, param in model.named_parameters():
        if subset is None or name in subset:
            grads = torch.autograd.grad(loss, param, retain_graph=True, allow_unused=True)[0]
            if grads is not None:
                gradients[name] = grads.clip(-0.5, 0.5)
    return gradients


def optimized_gradient(list_of_gradients):
    """
    Optimizes the aggregation of multiple gradients using constrained minimization.

    Given a list of gradients, this function computes a weighted aggregation of 
    these gradients by solving a constrained optimization problem. The aim is to 
    find optimal weights that minimize the norm of the aggregate gradient.

    Parameters
    ----------
    list_of_gradients : list of torch.Tensor
        A list containing gradients of model parameters with respect to different 
        objectives.

    Returns
    -------
    torch.Tensor
        The aggregated gradient optimized by the calculated weights.
    """

    def inner_norm(x):
        weighted_list_of_gradients = [list_of_gradients[i].flatten()*x[i] for i in range(len(list_of_gradients))]
        aggregated_gradients = torch.stack(weighted_list_of_gradients).sum(dim=0)
        dot_product = [torch.dot(aggregated_gradients, x[i]*list_of_gradients[i].flatten()) for i in range(len(list_of_gradients))]
        dot_product = torch.stack(dot_product).sum()
        return -dot_product
    bounds = {'lb':0,'ub':5}    
    res = minimize_constr(
        f = inner_norm,x0=torch.ones(len(list_of_gradients)),
        bounds=bounds
        )
    weighted_list_of_gradients = [list_of_gradients[i].flatten()*res.x[i] for i in range(len(list_of_gradients))]
    aggregated_gradients = torch.stack(weighted_list_of_gradients).sum(dim=0)
    return aggregated_gradients


def collect_non_zero_gradients(dictionary_of_gradients,name):
    gradients = []
    for i in dictionary_of_gradients:
        if name in dictionary_of_gradients[i]:
            if dictionary_of_gradients[i][name].abs().sum() > 0:
                gradients.append(dictionary_of_gradients[i][name])
    return gradients

def get_average_gradients(gradients,model):
    return torch.stack(gradients).to(model.device).sum(dim=0)/len(gradients)

def get_coordinated_lambda(dictionary_of_gradients,model,coordinated=True):

    """
    Calculates coordinated lambda values for model parameters based on gradients.

    This function examines the gradients of model parameters obtained from multiple
    loss functions, and determines if there is a conflict (i.e., negative dot product)
    between any pair of gradients. If there is a conflict, it applies an optimized
    strategy to determine the lambda value for each parameter; otherwise, it assigns
    a default lambda value of 1.

    Parameters
    ----------
    list_of_gradients : list
        A list of dictionaries containing gradients of the model parameters for
        different loss functions.
    model : torch.nn.Module
        The model whose parameters' lambda values are to be computed.

    Returns
    -------
    lambdas : dict
        A dictionary where keys are parameter names and values are the computed
        lambda values.
    """
        
    lambdas = {}
    for name, param in model.named_parameters():
        gradients = collect_non_zero_gradients(dictionary_of_gradients,name) 
        if len(gradients)>1:
            
            tuples = [torch.dot(gradients[i].flatten(),gradients[j].flatten())<0 
                          for i in range(len(gradients)) 
                          for j in range(len(gradients)) if i!=j ] 
            if any(tuples) and coordinated:
                lambdas[name] = optimized_gradient(gradients)
                print('Optimized Lambda')
            else:
                lambdas[name] = get_average_gradients(gradients,model)
                
        elif len(gradients)==1:
            lambdas[name] = gradients[0]
        
    return lambdas 

def gradient_descent_step(lambdas,model,optimizer,comparison_loss=1):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in lambdas:
                lambdas[name] = torch.reshape(lambdas[name],param.shape)
                param.grad = lambdas[name]
                if comparison_loss <= 0.01:
                    param.data -= param.grad*1e-3
                    param.grad = None
                
    optimizer.step()
    return model
