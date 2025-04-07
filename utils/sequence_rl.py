import torch


def difference_trajectory_loss(x,y):
    """
    Computes the loss between two sequences of states x and y.
    Parameters
    ----------
    x : torch.Tensor
        The sequence of states.
    y : torch.Tensor
        The target sequence of states.

    Returns
    -------
    loss : torch.Tensor
        The mean squared error loss between x and y.
    """
    loss = torch.nn.functional.l1_loss(x,y)
    return loss


def states_trajectory_loss(x,y):
    """
    Computes the loss between two sequences of states x and y.

    Parameters
    ----------
    x : torch.Tensor
        The sequence of states.
    y : torch.Tensor
        The target sequence of states.

    Returns
    -------
    loss : torch.Tensor
        The mean squared error loss between x and y.
    """
    loss = torch.nn.functional.mse_loss(x,y)
    return loss

def value_one_step_loss(rewards,V,V_at_n,gamma):
    """
    Computes the loss between a sequence of rewards and a value function at a single time step.

    Parameters
    ----------
    rewards : torch.Tensor
        The sequence of rewards.
    V : torch.Tensor
        The value function.
    V_at_n : torch.Tensor
        The target value function at the next time step.
    gamma : float
        The discount factor.

    Returns
    -------
    loss : torch.Tensor
        The Huber loss between the rewards and the value function.
    """
    device = V.device
    loss_metric = torch.nn.HuberLoss(delta=1)
    aggregated_rewards = rewards*gamma+V_at_n
    loss = loss_metric(V,V_at_n)
    return loss

def rewards_loss(x,y):
    """
    Computes the mean squared error loss between two sequences of rewards x and y.

    Parameters
    ----------
    x : torch.Tensor
        The sequence of rewards.
    y : torch.Tensor
        The target sequence of rewards.

    Returns
    -------
    loss : torch.Tensor
        The mean squared error loss between x and y.
    """
    loss = torch.nn.functional.mse_loss(x,y)
    return loss

def value_trajectory_loss(rewards,V,n,gamma):

    """
    Computes the loss between a sequence of rewards and a value function.

    Parameters
    ----------
    rewards : torch.Tensor
        The sequence of rewards.
    V_at_n_step : torch.Tensor
        The value of the state at the last step (n_step).
    V : torch.Tensor
        The value of the state at each step.
    n : int
        The number of steps.
    gamma : float
        The discount factor.

    Returns
    -------
    loss : torch.Tensor
        The mean squared error loss between the value function and the rewards.
    """
    device = V.device

    aggregated_rewards = torch.ones_like(rewards)+gamma*V[:,1:,:]
    loss_metric = torch.nn.functional.l1_loss(V[:,:-1,:],aggregated_rewards)
    
    return loss_metric

def compute_returns(rewards, gamma=0.9):
    returns = []
    G = 0
    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.insert(0, G)
    return returns

def proximal_policy_optimization_gradient_loss(probs,old_probs,V,accumulated_rewards,clip_epsilon=0.2):
    
    log_probs = torch.log(probs+1e-8)
    old_log_probs = torch.log(old_probs+1e-8)
    ratio = torch.exp(log_probs-old_log_probs)
    advantage = accumulated_rewards - V
    unclipped_loss = ratio*advantage
    clipped_ratio = torch.clamp(ratio,1-clip_epsilon,1+clip_epsilon)
    clipped_loss = clipped_ratio*advantage
    ratio_copy = ratio.clone().detach().cpu().numpy()
    clipped_ratio_copy = clipped_ratio.clone().detach().cpu().numpy()

    clipped_indexes = (ratio_copy > clipped_ratio_copy)
    unclipped_loss[clipped_indexes] = unclipped_loss[clipped_indexes]-unclipped_loss[clipped_indexes] + clipped_loss[clipped_indexes]
    policy_loss = -torch.sum(unclipped_loss)

    return policy_loss

def policy_entropy_loss(probs):
    log_probs = torch.log(probs+1e-8)
    entropy = -torch.sum(probs*log_probs)
    return -entropy*5

def policy_value_loss(V,accumulated_rewards):
    value_loss = torch.nn.functional.l1_loss(V,accumulated_rewards)
    return value_loss
