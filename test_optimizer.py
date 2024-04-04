import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# Define a simple neural network architecture
class NeuralNetwork(nn.Module):
    def __init__(self,layers:list=[],kind:str="binary",device:str="cpu"):
        """
        layers: list of layer sizes, indicating the shape of the input and output
        """
        if len(layers)<3:
            raise ValueError("The number of layers must be greater than 2.")
        for i in range(len(layers)-1):
            if layers[i][1] == layers[i+1][0]:
                pass
            else:
                raise ValueError("The shape of the network of one layer has to match the next one.")
            
        super(NeuralNetwork, self).__init__()
        self.layers = layers
        self.number_parameters = sum([np.product(pair)+pair[1] for pair in layers])
        self.weight_bias = torch.nn.Parameter(torch.randn(size=(self.number_parameters,1))).to(device)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.threshold = torch.nn.Parameter(torch.randn(1,1).to(device))
        self.k = 10
        self.kind = kind
    def forward(self, x):
        prev_layer = 0
        for layer in self.layers:
            layer_index = self.layers.index(layer)
            weigths = self.weight_bias[prev_layer:np.product(layer)+prev_layer].view(layer[0],layer[1])
            bias = self.weight_bias[np.product(layer)+prev_layer:np.product(layer)+prev_layer+layer[1]]
            x =  (torch.matmul(x,weigths).transpose(0,1)+bias).transpose(0,1)
            
            if layer_index < len(self.layers)-1 :
                x = self.relu(x)
            elif layer_index == len(self.layers)-1 and self.kind == "binary":
                x = self.sigmoid(x)
            elif layer_index == len(self.layers)-1 and self.kind == "regression":
                x = self.relu(x)
            prev_layer = np.product(layer)+layer[1]
        return x
    def heaviside(self,x):
        return 1/2 + 1/2*torch.tanh(x*self.k)
    def forward_threshold(self,x):
        x = self.forward(x)
        return  x-(x)*self.heaviside(x-self.threshold)-(x-1)*self.heaviside(self.threshold-x) 
    
# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNetwork(layers=[(10,10),(10,10),(10,1)],kind="binary",device = device)

# Sample data
X_train = torch.randn(100, 10).to(device)  # Input data
y_train = torch.randint(0, 2, (100, 1)).float().to(device)  # Binary labels

# Define three different objective functions
def true_positive(output, target):
    return (output * target).sum()
def true_negative(output, target):
    return ((1-output) * (1 - target)).sum()
def false_positive(output, target):
    return ((1-output) * target).sum()
def false_negative(output, target):
    return (output * (1 - target)).sum()

def bce_loss(output, target):
    return nn.BCEWithLogitsLoss()(output, target)

def precision_loss(output, target):
    tp = true_positive(output, target)
    fp = false_positive(output, target)
    return -tp/(tp+fp)
    
def accuracy_loss(output, target):
    tp = true_positive(output, target)
    tn = true_negative(output, target)
    fp = false_positive(output, target)
    fn = false_negative(output, target)
    return -(tp+tn)/(tp+tn+fp+fn)

def recall_loss(output, target):
    tp = true_positive(output, target)
    fn = false_negative(output, target)
    return -tp/(tp+fn)
    
def f1_loss(output, target):
    precision = precision_loss(output, target)
    recall = recall_loss(output, target)
    return 2*precision*recall/(precision+recall)
    
# Define optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
epochs = 1
model.weight_bias.retain_grad()
for epoch in range(epochs):
    gradients = []
    # Forward pass
    output_logits = model.forward(X_train)
    output_labels = model.forward_threshold(X_train)
    optimizer.zero_grad()

    # Compute gradients and update weights for loss2
    loss_val = bce_loss(output_logits, y_train)
    loss_val.backward(retain_graph=True)
    gradients.append(model.weight_bias.grad)
    
    # Compute gradients and update weights for loss1
    gradients = []
    for loss in [precision_loss,accuracy_loss,recall_loss,f1_loss]:
        optimizer.zero_grad()
        loss_val = loss(output_labels, y_train)
        loss_val.backward(retain_graph=True)
        gradients.append(model.weight_bias.grad)
    gradients = torch.stack(gradients)
    print(gradients)
    
    
    
    
