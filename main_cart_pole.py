from environment_simulation import run_sim
import numpy as np
from model import InterdependentResidualRNN
import gym
from gym.spaces import Discrete, Box
import matplotlib.pyplot as plt
import pandas as pd
from gym.wrappers.record_video import RecordVideo
from modl import collect_gradients, optimized_gradient, get_coordinated_lambda, gradient_descent_step
import torch
from utils.sequence_rl import value_trajectory_loss, states_trajectory_loss, difference_trajectory_loss

def train_for_controls_value(model,prev_states_step,next_states_step,rewards,bounds,rewards_bounds):
    outputs = model.forward_values_sequence(prev_states_step[:,0,:].unsqueeze(0),prev_states_step.shape[1],bounds,rewards_bounds*1.5)
    loss_value = value_trajectory_loss(rewards.unsqueeze(2).to(model.device),outputs,rewards.shape[1],1)
    subset = ['rnn2.'+name for name, _ in model.rnn2.named_parameters()]
    gradients = collect_gradients(loss=loss_value,model=model,subset=subset)
    print("loss value trajectories",loss_value)
    return gradients, loss_value.detach().cpu().numpy().item()

# def train_for_controls_prescribed(model,prev_states_step,next_states_step,rewards,bounds,target_states):
#     outputs = model.forward_actions(prev_states_step[:,0,:].unsqueeze(0),next_states_step.shape[1],bounds)
#     loss_controls = trajectory_target_loss(target_states.to(model.device),outputs)
#     subset = [name for name, _ in model.rnn2.named_parameters()]
#     gradients = collect_gradients(loss=loss_controls,model=model,subset=subset)
#     print("loss controls prescribed",loss_controls)
#     return gradients

def train_for_differences(model,prev_states_step,next_states_step,actions_step,bounds):
    """
    Trains the model by computing the gradients for the difference in state predictions.

    Parameters
    ----------
    model : object
        The model to be trained, with methods for forward propagation of state differences.
    prev_states_step : torch.Tensor
        The tensor containing the previous states for each step in the sequence.
    next_states_step : torch.Tensor
        The tensor containing the next states following each previous state.
    actions_step : torch.Tensor
        The tensor containing actions taken at each step.
    bounds : float
        The bounds used to constrain the model's predictions.

    Returns
    -------
    gradients : dict
        A dictionary of gradients for the parameters of the model's RNN1 network.
    loss_differences : float
        The computed loss value for the differences in state trajectories.
    """

    actions_step = actions_step.to(device=model.device).float()
    outputs = model.forward_states_differences(prev_states_step[:,0,:].unsqueeze(0), actions_step, actions_step.shape[1],bounds)
    differences = next_states_step - prev_states_step
    loss_differences = difference_trajectory_loss(differences.to(model.device),outputs)
    subset = ['rnn1.'+name for name, _ in model.rnn1.named_parameters()]
    gradients = collect_gradients(loss=loss_differences,model=model,subset=subset)
    print("loss differences",loss_differences)
    return gradients, loss_differences.detach().cpu().numpy().item()

def train_for_states(model,prev_states_step,next_states_step,actions_step,bounds):
    """
    Trains the model by computing the gradients for the states.

    Parameters
    ----------
    model : object
        The model to be trained, with methods for forward propagation of states.
    prev_states_step : torch.Tensor
        The tensor containing the previous states for each step in the sequence.
    next_states_step : torch.Tensor
        The tensor containing the next states following each previous state.
    actions_step : torch.Tensor
        The tensor containing actions taken at each step.
    bounds : float
        The bounds used to constrain the model's predictions.

    Returns
    -------
    gradients : dict
        A dictionary of gradients for the parameters of the model's RNN1 network.
    loss_states : float
        The computed loss value for the states.
    """
    
    actions_step = actions_step.to(device=model.device).float()
    outputs = model.forward_states_from_actions(prev_states_step[:,0,:].unsqueeze(0), actions_step, actions_step.shape[1],bounds)
    loss_states = states_trajectory_loss(next_states_step.to(model.device),outputs)
    print("loss states",loss_states)
    subset = ['rnn1.'+name for name, _ in model.rnn1.named_parameters()]
    gradients = collect_gradients(loss=loss_states,model=model,subset=subset)
    return gradients, loss_states.detach().cpu().numpy().item()


def run_and_train(model,num_episodes,env,num_steps,num_rewards,k,boolean,num_epochs,optimizer):
    """
    Runs simulations and trains the model on the generated trajectories.

    Parameters
    ----------
    model : object
        The model to be trained.
    num_episodes : int
        The number of episodes to run the simulation for.
    env : object
        The OpenAI Gym environment to be used.
    num_steps : int
        The maximum number of steps to run the simulation for.
    num_rewards : list
        A list to store the number of rewards collected in each episode.
    k : int
        The index in the num_rewards list to use for the current episode.
    boolean : bool
        A boolean indicating whether to use the value function or not.
    num_epochs : int
        The number of epochs to train the model for.
    optimizer : object
        The optimizer to use for training the model.

    Returns
    -------
    num_rewards : list
        The updated list of rewards collected in each episode.
    """
    epsilon = 0.9
    max_len = 0
    stored_trajectory = []
    stored_steps = [] 
    bounds = None
    rewards_bounds = None
    for i in range(num_episodes):
        prev_states, next_states, rewards, actions,epsilon,stored_steps= run_sim(env,model, num_steps,epsilon,stored_steps,optimizer)
        if len(rewards)>5:
            stored_trajectory.append([prev_states, next_states, rewards, actions,epsilon])
        num_rewards[k][i] = len(rewards)
        max_len = 5 if len(next_states)>5 else len(next_states)
                
        print('in episode {} we did {} steps'.format(i,len(next_states)))
        print("boolean is:",boolean)
        print('taken actions were:', actions)
        print("epsilon is:",epsilon)
        if len(rewards)>max_len:
            max_len = len(rewards)
        if len(stored_trajectory)>50:
            idx = np.random.choice(len(stored_trajectory),num_epochs if len(stored_trajectory)>num_epochs else len(stored_trajectory),replace=False)
            min_len_trajectory = min([torch.stack(stored_trajectory[i][0][-(max_len-k):]).shape[0] for i in idx])
            prev_states_step = torch.stack([torch.stack(stored_trajectory[i][0][-min_len_trajectory:]) for i in idx]).squeeze(2).squeeze(2)
            next_states_step = torch.stack([torch.stack(stored_trajectory[i][1][-min_len_trajectory:]) for i in idx]).squeeze(2).squeeze(2)
            rewards_step = torch.stack([torch.tensor(stored_trajectory[i][2][-min_len_trajectory:]) for i in idx])
            actions_step = torch.stack([torch.tensor(stored_trajectory[i][3][-min_len_trajectory:]) for i in idx]).unsqueeze(2)
            if bounds is None:
                bounds = torch.max(next_states_step - prev_states_step,dim=0).values.max(dim=0).values
            elif bounds is not None:
                new_bounds = torch.max(next_states_step - prev_states_step, dim=0).values.max(dim=0).values
                bounds = torch.max(bounds, new_bounds)
            if rewards_bounds is None:
                rewards_bounds = torch.max(rewards_step,dim=0).values.max(dim=0).values
            elif rewards_bounds is not None:
                new_rewards_bounds = torch.max(rewards_step,dim=0).values.max(dim=0).values
                rewards_bounds = torch.max(rewards_bounds, new_rewards_bounds)
            for _ in range(100):
                
                gradients_states,loss_states = train_for_states(model,prev_states_step,next_states_step,actions_step,bounds)
                gradients_differences,loss_differences = train_for_differences(model,prev_states_step,next_states_step,actions_step,bounds)
                if loss_states>0.5:
                    lambdas = get_coordinated_lambda({1:gradients_states,2:gradients_differences},model,False)
                elif loss_states<0.1:
                    gradients_value,loss_value = train_for_controls_value(model,prev_states_step,next_states_step,rewards_step,bounds,rewards_bounds)
                    print(gradients_value)
                    lambdas = get_coordinated_lambda({1:gradients_states,2:gradients_differences,3:gradients_value},model,True)
                else:
                    lambdas = get_coordinated_lambda({1:gradients_states,2:gradients_differences},model,True)
                # lambdas = get_coordinated_lambda({1:gradients_differences},model,False)
                model = gradient_descent_step(lambdas,model,optimizer,loss_states)
                # for j in range(min_len_trajectory):
                       # outputs_V = model.forward_value(prev_states_step[:,j,:]).max(dim=1).values.unsqueeze(1)
                        # outputs_V_at_n_step = model.forward_value(prev_states_step[:,min_len_trajectory-1,:]).max(dim=1).values.unsqueeze(1)
                        
                        # loss_value = value_with_rewards_trajectory_loss(outputs_V,outputs_V_at_n_step,rewards_step[:,j:].to(model.device),rewards_step[:,j:].shape[1],0.9 if boolean else 0.9)
                        # gradients[j]=collect_gradients(loss=loss_value,model=model)
                        # lambdas = get_coordinated_lambda([grads_states,grads_values],model)         
                        # prev_states, next_states, rewards, actions,epsilon= run_sim(env,model, num_steps,epsilon,seq_len=10)
                
    return num_rewards

def main(env_name, num_steps=1000,num_episodes=100,learning_rate = 1e-4,num_epochs = 4):
    # Create the environment
    env = gym.make(env_name, render_mode="rgb_array")
    state_dim = env.observation_space.shape[0] if isinstance(env.observation_space, Box) else env.observation_space.n 
    actions_dim = env.action_space.n if isinstance(env.action_space, Discrete) else env.action_space.shape[0]
    length = [i for i in range(4,5)]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for boolean in [True,False]:
        num_rewards = {}
        for k in length:
            model = InterdependentResidualRNN(input_size1=1, hidden_size1=state_dim, 
                                              input_size2=state_dim, hidden_size2=actions_dim,hidden_output_1 = 32,hidden_output_2=32)
            # Initialize Adam optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            num_rewards[k] = {}
            num_rewards = run_and_train(model,num_episodes,env,num_steps,num_rewards,k,boolean,num_epochs,optimizer)               
                       
        num_rewards = pd.DataFrame(num_rewards)
        for i in num_rewards.columns:
            linestyle = '--' if not boolean else '-'
            if boolean==True:
                ax1.plot(num_rewards[i], linestyle=linestyle)
            else:
                ax2.plot(num_rewards[i], linestyle=linestyle)
    ax1.set_ylim(0,300)
    ax2.set_ylim(0,300)
    plt.show()    
if __name__ == "__main__":
    main("CartPole-v1")
