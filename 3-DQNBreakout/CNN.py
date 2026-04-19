import torch

class Model(torch.nn.Module):
    def __init__(self, architecture):
        super().__init__()
        self.architecture = architecture
        self.layers = torch.nn.ModuleList()

        for layer in architecture:
            layer_type = layer[0]

            if layer_type == "Conv2d":
                self.layers.append(torch.nn.Conv2d(layer[1], layer[2], layer[3], layer[4]))
            elif layer_type == "ReLU":
                self.layers.append(torch.nn.ReLU())
            elif layer_type == "Flatten":
                self.layers.append(torch.nn.Flatten())
            elif layer_type == "Linear":
                self.layers.append(torch.nn.Linear(layer[1], layer[2]))
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
