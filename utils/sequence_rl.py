import torch

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
    y=y[0]
    return torch.sum((x-y)**2)

def value_with_rewards_trajectory_loss(V_at_n_step,V,rewards,n,gamma):

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
    device = V_at_n_step.device
    aggregated_rewards = torch.sum(rewards*(gamma**torch.arange(n).to(device)))+gamma**n*V_at_n_step
    loss = torch.nn.HuberLoss(delta=1)
    return loss(V,aggregated_rewards)
