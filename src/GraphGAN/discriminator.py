import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class Discriminator(nn.Module):
    def __init__(self, n_node, node_emd_init):
        super(Discriminator, self).__init__()
        self.n_node = n_node
        self.node_emd_init = torch.from_numpy(node_emd_init)
        self.embedding_matrix = Parameter(torch.FloatTensor(node_emd_init.shape))
        self.bias_vector = Parameter(torch.FloatTensor(n_node))

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding_matrix.data = self.node_emd_init
        self.bias_vector.data = torch.zeros([self.n_node])

    def forward(self, node_id, node_neighbor_id):
        node_embedding = self.embedding_matrix[node_id]
        node_neighbor_embedding = self.embedding_matrix[node_neighbor_id]
        bias = self.bias_vector[node_neighbor_id]
        score = torch.sum(node_embedding * node_neighbor_embedding, dim=1) + bias
        score = torch.clamp(score, -10, 10)
        
        return score, node_embedding, node_neighbor_embedding, bias

    def get_reward(self, score):
        reward = torch.log(1 + torch.exp(score))

        return reward
