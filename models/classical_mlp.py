import torch

class MLP(torch.nn.Module):
    def __init__(self, input_size=784, hidden_sizes=[128,64], num_classes=10):
        super(MLP, self).__init__()
        self.input_size     = input_size
        self.hidden_sizes   = hidden_sizes
        self.num_classes    = num_classes


        # Layers
        self.fc1  = torch.nn.Linear(input_size, hidden_sizes[0])
        self.fc2  = torch.nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3  = torch.nn.Linear(hidden_sizes[1], num_classes[0])
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # Reshapes the input tensor x into input_size
        x           = x.view(-1,self.input_size)

        hidden1     = self.relu(self.fc1(x))
        hidden2     = self.relu(self.fc2(hidden1))
        output      = self.fc3(hidden2)

        return output 