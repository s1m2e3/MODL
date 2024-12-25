from environment_simulation import run_sim
from model import InterdependentResidualRNN
import gym
from gym.spaces import Discrete, Box

from gym.wrappers.record_video import RecordVideo
from modl import collect_gradients, optimized_gradient, get_coordinated_lambda, gradient_descent_step
import torch
from utils.sequence_rl import value_with_rewards_trajectory_loss, states_trajectory_loss
def main(env_name, num_steps=1000,num_episodes=500,learning_rate = 1e-4,num_epochs = 10):
    # Create the environment
    env = gym.make(env_name, render_mode="rgb_array")
    state_dim = env.observation_space.shape[0] if isinstance(env.observation_space, Box) else env.observation_space.n 
    actions_dim = env.action_space.n if isinstance(env.action_space, Discrete) else env.action_space.shape[0]
    
    model = InterdependentResidualRNN(input_size1=1, hidden_size1=state_dim, input_size2=state_dim, hidden_size2=actions_dim)
    epsilon = 0.9
    max_len = 0
    for i in range(num_episodes):
        prev_states, next_states, rewards, actions,epsilon= run_sim(env,model, num_steps,epsilon,seq_len=10)
        print('in episode {} we did {} steps'.format(i,len(next_states)))
        print('taken actions were:', actions)
        print("epsilon is:",epsilon)
        # if len(rewards)>max_len:
        # print(len(rewards),max_len)
        # max_len = len(rewards)
        # print(max_len)
        for i in range(num_epochs):
            gradients = {}
            max_len = 30 if len(next_states)>30 else len(next_states)
            for j in range(len(max_len)):
                if len(prev_states)>1:
                  
                    # actions_tensor = torch.tensor(actions).to(device=model.device).float().unsqueeze(1).unsqueeze(1)
                    # outputs = model.forward_states_from_actions(prev_states[0], actions_tensor, len(actions))
                    # loss_states = states_trajectory_loss(torch.stack(next_states).to(model.device).squeeze(1).squeeze(1),outputs)
                    # grads_states = collect_gradients(loss=loss_states,model=model)
                    outputs_V = model.forward_value(prev_states[j]).max()
                    outputs_V_at_n_step = model.forward_value(prev_states[len(next_states)-1]).max()
                    loss_value = value_with_rewards_trajectory_loss(outputs_V,outputs_V_at_n_step,torch.tensor(rewards).to(model.device),len(rewards),0.8)
                    gradients[j]=collect_gradients(loss=loss_value,model=model)
                    # lambdas = get_coordinated_lambda([grads_states,grads_values],model)         
                    # prev_states, next_states, rewards, actions,epsilon= run_sim(env,model, num_steps,epsilon,seq_len=10)
            lambdas = get_coordinated_lambda(gradients,model)
            model = gradient_descent_step(lambdas,model,learning_rate)
           
        # print("losses states and value respectively:",round(loss_states.detach().item(),3),round(loss_value.detach().item(),3))
        print("losses states and value respectively:",round(loss_value.detach().item(),3))
        
if __name__ == "__main__":
    main("CartPole-v1")
