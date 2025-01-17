import torch.nn as nn
import torch
import numpy as np
import pandas as pd
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


import torch
import torch.nn as nn

class InterdependentResidualRNN(nn.Module):
    def __init__(self, input_size1, hidden_size1, input_size2, hidden_size2,hidden_size_V=64,hidden_output_1=64,hidden_output_2=64):
        super(InterdependentResidualRNN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size1 = input_size1
        self.hidden_size1 = hidden_size1
        self.input_size2 = input_size2
        self.hidden_size2 = hidden_size2

        # Define RNNs for h_t^{(1)} and h_t^{(2)}
        self.rnn1 = nn.RNN(input_size1, hidden_size1, batch_first=True,device=self.device)
        self.rnn2 = nn.RNN(input_size2, hidden_size2, batch_first=True,device=self.device)
        

        # Residual connection weights
        self.W1 = nn.Linear(hidden_size1, hidden_output_1,device=self.device)  # h_t^{(2)} -> h_{t+1}^{(1)}
        self.W1_out = nn.Linear(hidden_output_1, hidden_size1,device=self.device)
        self.W2 = nn.Linear(hidden_size1, hidden_size2,device=self.device)  # h_t^{(1)} -> h_{t+1}^{(2)}
        self.V_value_1 = nn.Linear(hidden_size1, hidden_size_V,device=self.device)
        self.V_value_2 = nn.Linear(hidden_size_V, hidden_size_V,device=self.device)
        self.V_value_3 = nn.Linear(hidden_size_V, 1,device=self.device)

        self.V_policy_1 = nn.Linear(hidden_size1, hidden_size_V,device=self.device)
        self.V_policy_2 = nn.Linear(hidden_size_V, hidden_size_V,device=self.device)
        self.V_policy_3 = nn.Linear(hidden_size_V, hidden_size2,device=self.device)
    
        self.R1 = nn.Linear(input_size2, hidden_size_V,device=self.device)
        self.R2 = nn.Linear(hidden_size_V, hidden_size_V,device=self.device)
        self.R3 = nn.Linear(hidden_size_V, 1,device=self.device)

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