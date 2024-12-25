import torch.nn as nn
import torch
import numpy as np

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
    def __init__(self, input_size1, hidden_size1, input_size2, hidden_size2,hidden_size_V=64):
        super(InterdependentResidualRNN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size1 = input_size1
        self.hidden_size1 = hidden_size1
        self.input_size2 = input_size2
        self.hidden_size2 = hidden_size2

        # Define RNNs for h_t^{(1)} and h_t^{(2)}
        self.rnn1 = nn.RNN(input_size1, hidden_size1, batch_first=True,device=self.device)
        
        # Residual connection weights
        self.W1 = nn.Linear(hidden_size2, hidden_size1,device=self.device)  # h_t^{(2)} -> h_{t+1}^{(1)}
        self.W2 = nn.Linear(hidden_size1, hidden_size2,device=self.device)  # h_t^{(1)} -> h_{t+1}^{(2)}
        self.V1 = nn.Linear(hidden_size1, hidden_size_V,device=self.device)
        self.V2 = nn.Linear(hidden_size_V, hidden_size_V,device=self.device)
        self.V3 = nn.Linear(hidden_size_V, hidden_size2,device=self.device)
    
    def forward_states_from_actions(self, h1_0, x1, seq_len,batch_size=1):
        """
        h1_0: Initial hidden state for RNN1 that represents the dynamics predictions (1, batch_size, hidden_size1)
        x1: Input sequence of actions (seq_len, batch_size, hidden_size2)
        """
        h1 = h1_0.to(self.device)
        next_states = [] # Store the outputs of both RNNs
        for t in range(seq_len):
            # With initial state compute the output of the policy network
            h1_new = self.rnn1(x1[t], h1[0])[0].squeeze(1) + h1.squeeze(0)
            h1 = h1_new.unsqueeze(0)  # Add batch dimension for the next step
            # Store outputs for the current timestep
            next_states.append(h1)  # Keep the time dimension
        
        # Concatenate outputs along the time dimension
        next_states = torch.cat(next_states, dim=1).squeeze(0)  # (batch_size, seq_len, hidden_size1)
        return next_states
    
    def forward_actions(self, h1_0, seq_len,batch_size=1):
        """
        h1_0: Initial hidden state for RNN1 that represents the dynamics predictions (1, batch_size, hidden_size1)
        h2_0: Initial hidden state for RNN2 that represents the policy predictions (1, batch_size, hidden_size2)
        """
        h1_0 = h1_0.to(self.device)
        x2 = h1_0# Initial input passed to policy network which is the initial state 
        h1 = h1_0 # Initial hidden state for dynamics network which is the initial state
        h2 = torch.zeros(1, batch_size, self.hidden_size2).to(self.device)
            
        actions = []  # Store the outputs of both RNNs
        for t in range(seq_len):
            # With initial state compute the output of the policy network
            value = self.forward_value(x2)
            x1 = value.argmax().unsqueeze(0).unsqueeze(0).unsqueeze(0).float().to(self.device)
            h1_new = self.rnn1(x1, h1)[0].squeeze(1) + h1.squeeze(0)
            x2 = h1_new.unsqueeze(0)
            # Update hidden states for the next timestep
            h1 = h1_new.unsqueeze(0)  # Add batch dimension for the next step
            # Store outputs for the current timestep
            actions.append(x1)
        
        # Concatenate outputs along the time dimension
        actions = torch.cat(actions, dim=1).squeeze(0)  # (batch_size, seq_len, hidden_size2)
        return actions

    def forward_value(self, x):
        """
        x: State (1, batch_size, hidden_size1)
        """
        outputs = torch.tanh(self.V1(x))
        outputs = torch.tanh(self.V2(outputs))
        outputs = self.V3(outputs)
        return outputs