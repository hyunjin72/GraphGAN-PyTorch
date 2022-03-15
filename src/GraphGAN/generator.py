import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class Generator(nn.Module):
    def __init__(self, n_node, node_emd_init):
        super(Generator, self).__init__()
        self.n_node = n_node
        self.node_emd_init = torch.from_numpy(node_emd_init)
        self.embedding_matrix = Parameter(torch.FloatTensor(node_emd_init.shape))
        self.bias_vector = Parameter(torch.FloatTensor(n_node))

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding_matrix.data = self.node_emd_init
        self.bias_vector.data = torch.zeros([self.n_node])

    def forward(self, node_id, node_neighbor_id):
        # write down loss, optimizers in graph_gan_torch.py
        node_embedding = self.embedding_matrix[node_id]
        node_neighbor_embedding = self.embedding_matrix[node_neighbor_id]
        bias = self.bias_vector[node_neighbor_id]
        score = torch.sum(node_embedding * node_neighbor_embedding, dim=1) + bias
        prob = torch.sigmoid(score)
        prob = torch.clamp(prob, 1e-5, 1)

        return node_embedding, node_neighbor_embedding, prob

    def get_all_score(self):
        all_score = torch.matmul(self.embedding_matrix, self.embedding_matrix.t()) + self.bias_vector

        return all_score
