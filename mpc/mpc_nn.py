import torch
import torch.nn as nn
import math 

# Define RNN-based MPC controller
class ResidualRNNController(nn.Module):
    def __init__(self, state_dim, hidden_dim, output_dim):
        super().__init__()
        self.rnn = nn.RNN(state_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h_prev):
        h_out, h_next = self.rnn(x.unsqueeze(1), h_prev)  # RNN output h_t

        # Residual connection on hidden state
        h_res = h_out + h_prev[0]  # assuming h_prev is a tuple (h_n, ...)
        
        u = self.fc(h_res).squeeze(1)  # Compute control output from residual state
        
        return u, h_next


class NStepResidualRNNController(ResidualRNNController):
    def __init__(self, state_dim, hidden_dim, output_dim, n_steps=1):
        super().__init__(state_dim, hidden_dim, output_dim)
        self.n_steps = min(n_steps,5)
        self.rnn = [nn.RNN(state_dim, hidden_dim, batch_first=True) for _ in range(n_steps)]
        self.fc = [nn.Linear(hidden_dim, output_dim) for _ in range(n_steps)]
    
    def forward(self, x, h_prev):
        if self.n_steps >= 1:
            h_res_list = []
            control_outputs = []

            for i in range(self.n_steps):
                h_out, _ = self.rnn[i](x.unsqueeze(1), h_prev)

                if i == 0:
                    h_res = h_out + h_prev[0]
                else:
                    # Compute weighted sum using binomial coefficients
                    coeffs = torch.tensor([math.comb(i, k) for k in range(i+1)])  # Pascal's triangle row
                    residual_sum = sum(c * h_res_list[-(k+1)] for k, c in enumerate(coeffs[:-1]))
                    h_res = h_out + h_prev[0] + residual_sum

                h_res_list.append(h_res)
                u = self.fc[0](h_res).squeeze(1)
                control_outputs.append(u)

        return torch.stack(control_outputs), _
