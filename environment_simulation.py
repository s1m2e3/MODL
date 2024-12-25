import torch
import numpy as np
import random
import matplotlib.pyplot as plt
def run_sim(env,model,num_steps=1000,epsilon=0.9,seq_len=10):
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
    obs = torch.from_numpy(obs[0]).float().unsqueeze(0).unsqueeze(0).to(model.device)
    states.append(obs)    
    for _ in range(num_steps):
        img = env.render()
        plt.imshow(img)
        plt.pause(0.01)
        pred_actions  = model.forward_value(obs)
        action = int(pred_actions[0,0,:].argmax().item())
        if np.random.rand() < epsilon:
            action = random.randint(0,1)
            epsilon = epsilon*0.999
            # print("chosen random action",action)
        # else:
            # print("chosen action",action)
        obs, reward, done, _, info = env.step(action)
        obs = torch.tensor(obs).float().unsqueeze(0).unsqueeze(0).to(model.device)
        
        states.append(obs)
        rewards.append(reward)
        actions.append(action)

        if done: 
            if _ < 100:
                rewards[-1]=rewards[-1]-100
            break
        plt.clf()
    env.close()
    plt.close()
    return states[0:-1],states[1:],rewards,actions,epsilon