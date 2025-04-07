import torch.nn as nn
import torch
import numpy as np
import pandas as pd
import torch as th
import numpy as np
from stable_baselines3 import PPO
from gymnasium import spaces
from torch.nn import functional as F
import torch
import torch.nn as nn
import random
from modl import get_coordinated_lambda, gradient_descent_step



class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size1, output_size, device):
        super(SimpleNN, self).__init__()
        self.device = device
        self.layer1 = nn.Linear(input_size, hidden_size1).to(device)
        self.output_layer = nn.Linear(hidden_size1, output_size).to(device)
        self.tanh = nn.Tanh().to(device)

    def forward(self, x):
        x = self.tanh(self.layer1(x))
        x = self.output_layer(x)
        return x



class InterdependentResidualRNN(nn.Module):
    def __init__(self, input_size1, hidden_size1, input_size2, hidden_size2,hidden_size_V=256,hidden_output_1=256,hidden_output_2=256):
        super(InterdependentResidualRNN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size1 = input_size1
        self.hidden_size1 = hidden_size1
        self.input_size2 = input_size2
        self.hidden_size2 = hidden_size2

        # # Define RNNs for h_t^{(1)} and h_t^{(2)}
        # self.rnn1 = nn.RNN(input_size1, hidden_size1, batch_first=True,device=self.device)
        # self.rnn2 = nn.RNN(input_size2, hidden_size2, batch_first=True,device=self.device)
        

        # # Residual connection weights
        # self.W1 = nn.Linear(hidden_size1, hidden_output_1,device=self.device)  # h_t^{(2)} -> h_{t+1}^{(1)}
        # self.W1_out = nn.Linear(hidden_output_1, hidden_size1,device=self.device)
        # self.W2 = nn.Linear(hidden_size1, hidden_size2,device=self.device)  # h_t^{(1)} -> h_{t+1}^{(2)}
        self.V_value_1 = nn.Linear(hidden_size1, hidden_size_V,device=self.device)
        self.V_value_2 = nn.Linear(hidden_size_V, hidden_size_V,device=self.device)
        self.V_value_3 = nn.Linear(hidden_size_V, 1,device=self.device)

        self.V_policy_1 = nn.Linear(hidden_size1, hidden_size_V,device=self.device)
        self.V_policy_2 = nn.Linear(hidden_size_V, hidden_size_V,device=self.device)
        self.V_policy_3 = nn.Linear(hidden_size_V, hidden_size2,device=self.device)
    
        # self.R1 = nn.Linear(input_size2, hidden_size_V,device=self.device)
        # self.R2 = nn.Linear(hidden_size_V, hidden_size_V,device=self.device)
        # self.R3 = nn.Linear(hidden_size_V, 1,device=self.device)

    def bound_condition(self,x):
        return torch.relu(x)

    def smooth_bound(self,x, bound):
        return bound * torch.tanh(x / bound)

    def forward_reward(self,x):
        x = torch.relu(self.R1(x))
        x = torch.relu(self.R2(x))
        x = self.R3(x)
        return x

    def bound_jump(self,x,bound):
        added_upper = self.bound_condition(-x+bound)
        added_lower = self.bound_condition(x+bound)
        
        # Use torch.where to apply conditions without in-place operations
        x = torch.where(added_upper == 0, self.smooth_bound(x, bound), x)  # Set to upper bound
        x = torch.where(added_lower == 0, self.smooth_bound(x, -bound), x)  # Set to lower bound
        return x
    def forward_states_from_actions(self, h1_0, x1, seq_len,bounds,batch_size=1):
        """
        h1_0: Initial hidden state for RNN1 that represents the dynamics predictions (1, batch_size, hidden_size1)
        x1: Input sequence of actions (seq_len, batch_size, hidden_size2)
        """
        h1 = h1_0.to(self.device).contiguous()
        new_states = []
        for t in range(x1.shape[1]):
            h1_new = self.bound_jump(self.rnn1(x1[:,t,:].unsqueeze(-1), h1.squeeze(-1))[0],bounds)+h1.squeeze(0).unsqueeze(1)
            new_states.append(h1_new)
            h1 = h1_new.squeeze(1).unsqueeze(0)
        
        new_states = torch.cat(new_states,dim=1).to(self.device)
        return new_states 
    
    def forward_states_differences(self, h1_0, x1, seq_len,bounds,batch_size=1):
        """
        h1_0: Initial hidden state for RNN1 that represents the dynamics predictions (1, batch_size, hidden_size1)
        x1: Input sequence of actions (seq_len, batch_size, hidden_size2)
        """
        
        h1 = h1_0.to(self.device).contiguous()
        differences = []
        for t in range(x1.shape[1]):
            diff = self.bound_jump(self.rnn1(x1[:,t,:].unsqueeze(-1), h1.squeeze(-1))[0],bounds)
            h1_new = diff+h1.squeeze(0).unsqueeze(1)
            # +h1.squeeze(-1)
            differences.append(diff)
            h1 = h1_new.squeeze(1).unsqueeze(0)
        
        differences = torch.cat(differences,dim=1).to(self.device)
        return differences 

    def forward_actions_sequence(self, h1_0, seq_len,bounds,batch_size=1):
        """
        h1_0: Initial hidden state for RNN1 that represents the dynamics predictions (1, batch_size, hidden_size1)
        h2_0: Initial hidden state for RNN2 that represents the policy predictions (1, batch_size, hidden_size2)
        """
        h1_0 = h1_0.to(self.device)
        x2 = h1_0.squeeze(0).unsqueeze(1)# Initial input passed to policy network which is the initial state 
        h1 = h1_0 # Initial hidden state for dynamics network which is the initial state
        h2 = self.forward_value(h1)
        actions = []  # Store the outputs of both RNNs
        for t in range(seq_len-1):
            # With initial state compute the output of the policy network
            h2_new = self.rnn2(x2, h2)[0]+h2.squeeze(0).unsqueeze(1)
            max_index = torch.argmax(h2_new, dim=2).unsqueeze(2)
            
            actions.append(max_index.squeeze(2))
            x1 = max_index.float().squeeze(0).to(self.device)
            x2 = self.forward_states_from_actions(x2.squeeze(1).unsqueeze(0), x1, 1,bounds,batch_size)
            h2 = h2_new.squeeze(1).unsqueeze(0)

        # Concatenate outputs along the time dimension
        actions = torch.stack(actions, dim=1)  # (batch_size, seq_len, hidden_size2)
        return actions

    def forward_values_sequence(self, h1_0, seq_len,bounds,rewards_bounds,batch_size=1):
        """
        h1_0: Initial hidden state for RNN1 that represents the dynamics predictions (1, batch_size, hidden_size1)
        h2_0: Initial hidden state for RNN2 that represents the policy predictions (1, batch_size, hidden_size2)
        """
        h1_0 = h1_0.to(self.device)
        x2 = h1_0.squeeze(0).unsqueeze(1)# Initial input passed to policy network which is the initial state 
        h1 = h1_0 # Initial hidden state for dynamics network which is the initial state
        h2 = self.forward_value(h1)
        max_index = torch.argmax(h2, dim=2).unsqueeze(2)
        values = [torch.gather(h2,2,max_index).squeeze(0)]  # Store the outputs of both RNNs
        actions = [max_index.squeeze(0)]
        for t in range(seq_len):
            x1 = max_index.float().squeeze(0).unsqueeze(1).to(self.device)
            x2 = self.forward_states_from_actions(x2.squeeze(1).unsqueeze(0), x1, 1,bounds,batch_size)
            # With initial state compute the output of the policy network
            h2_new = h2.squeeze(0).unsqueeze(1)+self.bound_jump(self.rnn2(x2, h2)[0],rewards_bounds)
            max_index = torch.argmax(h2_new, dim=2).unsqueeze(2)
            actions.append(max_index.squeeze(1))
            values.append(torch.gather(h2_new,2,max_index).squeeze(2))
            h2 = h2_new.squeeze(1).unsqueeze(0)
            max_index = max_index.squeeze(1).unsqueeze(0)
            # Store outputs for the current timestep
        # Concatenate outputs along the time dimension
        values = torch.stack(values,dim=1)  # (batch_size, seq_len, hidden_size2)
        actions = torch.stack(actions,dim=1)
        print(pd.DataFrame(actions.squeeze(2).detach().cpu().numpy()))
        return values

    def forward_value(self, x):
        """
        x: State (1, batch_size, hidden_size1)
        """
        outputs = torch.relu(self.V_value_1(x))
        outputs = torch.relu(self.V_value_2(outputs))
        outputs = self.V_value_3(outputs)
        return outputs
    
    def forward_probs(self, x):
        """
        Computes the probability distribution over actions given the input state.

        Parameters
        ----------
        x : torch.Tensor
            The input state tensor with shape (1, batch_size, hidden_size1).

        Returns
        -------
        torch.Tensor
            A tensor representing the probability distribution over actions, obtained 
            by applying a softmax function to the output layer.
        """
        outputs = torch.relu(self.V_policy_1(x))
        outputs = torch.relu(self.V_policy_2(outputs))
        outputs = self.V_policy_3(outputs)
        return torch.softmax(outputs, dim=2)
    


class CustomPPO(PPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                print(rollout_data)
                input('hipi')
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                
                entropy_losses.append(entropy_loss.item())
                if random.random() >1:
                    grads = {1:{},2:{},3:{}}

                    self.policy.optimizer.zero_grad()
                    policy_loss.backward(retain_graph=True)
                    for name,param in self.policy.named_parameters():
                        grads[1][name] = param.grad

                    self.policy.optimizer.zero_grad()
                    entropy_loss.backward(retain_graph=True)
                    for name,param in self.policy.named_parameters():
                        grads[2][name] = param.grad

                    self.policy.optimizer.zero_grad()
                    value_loss.backward(retain_graph=True)
                    for name,param in self.policy.named_parameters():
                        grads[3][name] = param.grad
                    
                    lambdas = get_coordinated_lambda(grads,self.policy,True)
                    self.policy = gradient_descent_step(lambdas,self.policy,self.policy.optimizer)
                else:
                    loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                    
                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                    with th.no_grad():
                        log_ratio = log_prob - rollout_data.old_log_prob
                        approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                        approx_kl_divs.append(approx_kl_div)

                    if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                        continue_training = False
                        if self.verbose >= 1:
                            print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                        break
                     # Optimization step
                    self.policy.optimizer.zero_grad()
                    loss.backward()
                    # Clip grad norm
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.policy.optimizer.step()
                print(policy_loss,entropy_loss,value_loss)

            self._n_updates += 1
            if not continue_training:
                break

        # explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        # self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

# class CustomDQN(OffPolicyAlgorithm):
#      def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#     def train(self) -> None:
    