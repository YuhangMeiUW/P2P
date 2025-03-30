import torch
import torch.nn as nn

class score_nn(torch.nn.Module):
    def __init__(self, x_dim, u_dim, hidden_dim):
        super(score_nn, self).__init__()
        self.x_dim = x_dim
        self.hidden_dim = hidden_dim
        self.u_dim = u_dim
        
        self.activation = nn.ELU()
        
        self.layer_input = nn.Linear(self.x_dim + 1, self.hidden_dim, bias=True)

        self.layer_1 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.layer_2 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.layer_3 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.layer_4 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.layer_5 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.layer_6 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)

        self.layerout = nn.Linear(self.hidden_dim, self.u_dim, bias=True)
                
    def forward(self, x,t):
        z_in = torch.concat((x,t),dim=1)

        h = self.layer_input(z_in)
        h_temp = self.activation(self.layer_1(h)) 
        h_temp = self.layer_2(h_temp)
        h = self.activation(h_temp + h) 

        h_temp = self.activation(self.layer_3(h))
        h_temp = self.layer_4(h_temp)
        h = self.activation(h_temp + h)

        h_temp = self.activation(self.layer_5(h))
        h_temp = self.layer_6(h_temp)
        h = self.activation(h_temp + h)
        
        z_out = self.layerout(h) 
        return z_out