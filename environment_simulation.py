import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import time


def run_sim(env,model,num_steps=1000,epsilon=0.9,stored_steps=[],optimizer=None):
    """
    Runs a simulation in a specified OpenAI Gym environment using a given model.

    Parameters
    ----------
    env : env
        The OpenAI Gym environment to be used.
    model : object
        The model used to predict actions based on the current state.
    num_steps : int, optional
        The maximum number of steps to run the simulation (default is 1000).

    Returns
    -------
    states : list
        A list of states observed during the simulation.
    next_states : list
        A list of states that follow each state in the `states` list.
    rewards : list
        A list of rewards received at each step of the simulation.
    actions : list
        A list of actions taken at each step of the simulation.
    """
    # Test the agent
    obs = env.reset()
    states = []
    rewards = []
    actions = []
    old_probs = []
    obs = torch.from_numpy(obs[0]).float().unsqueeze(0).unsqueeze(0).to(model.device)
    states.append(obs)
    for _ in range(num_steps):
        prob_actions  = model.forward_probs(obs)
        old_probs.append(prob_actions)
        action = np.random.choice(prob_actions.shape[2], p=prob_actions.squeeze(0).squeeze(0).detach().cpu().numpy())
        obs, reward, done, _, info = env.step(action)
        dist_to_goal = round(-abs(obs[0])-abs(obs[2])*3- abs(obs[1])-abs(obs[3])*3,5)                  
        reward += dist_to_goal
        obs = torch.tensor(obs).float().unsqueeze(0).unsqueeze(0).to(model.device)
        
        states.append(obs)
        rewards.append(reward)
        actions.append(action)
        stored_steps.append([states[-2],states[-1],rewards[-1],actions[-1]])
        # gradients = {}
        # if len(stored_steps)>3:
        #     idx = random.choice(range(len(stored_steps)))
        #     idxs = [i for i in range(idx,len(stored_steps))]
        #     if len(idxs)>3:
        #         next_states_step = torch.stack([stored_steps[i][1] for i in idxs]).squeeze(1)
        #         rewards_step = torch.tensor([stored_steps[i][2] for i in idxs])
        #         actions_step = torch.tensor([stored_steps[i][3] for i in idxs])
        #         accumulated_rewards = torch.stack(compute_returns(rewards_step,0.999)).unsqueeze(1).float()
        #         values = model.forward_value(next_states_step).squeeze(1).float()
        #         probs = model.forward_probs(next_states_step).squeeze(1)
        #         probs = probs[range(len(actions_step)), actions_step].unsqueeze(1).float()
        #         loss_policy = policy_gradient_loss(probs,values,accumulated_rewards.to(model.device))
        #         gradients_policy = collect_gradients(loss_policy,model)
        #         loss_entropy = policy_entropy_loss(probs)
        #         gradients_entropy = collect_gradients(loss_entropy,model)
        #         loss_value = policy_value_loss(values,accumulated_rewards.to(model.device))
        #         gradients_value = collect_gradients(loss_value,model)
        #         lambdas = get_coordinated_lambda({1:gradients_policy,2:gradients_entropy,3:gradients_value},model,False if random.random()>0.7 else True)
        #         model = gradient_descent_step(lambdas,model,optimizer)  
        #         print(loss_policy)
        #         print(loss_entropy)
        #         print(loss_value)

        if done: 
            # if _ < 100:
                # rewards[-1]=rewards[-1]-100
            break
    env.close()
    
    return states[0:-1],states[1:],rewards,actions,epsilon,stored_steps,old_probs