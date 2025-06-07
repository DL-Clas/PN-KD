"""
Differential Knowledge Distillation by Samples Graphical Representation
"""
import torch
import torch.nn as nn
import torch_geometric.data as geom_data

# node_cos + edge_cos

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def features_to_graph(features):
    num_nodes = features.shape[0]
    # Calculate the similarity between all pairs of nodes
    norm_features = nn.functional.normalize(features, p=2, dim=1)
    sim_matrix = torch.mm(norm_features, norm_features.t())
    # Exclude the similarity of its own nodes
    mask = ~torch.eye(num_nodes, dtype=torch.bool, device=device)
    edge_weights = sim_matrix[mask]
    # Generate an edge index
    edge_index = torch.nonzero(mask).t().contiguous()
    return geom_data.Data(x=features, edge_index=edge_index, edge_weight=edge_weights)


class Loss_compute:
    def __init__(self):
        self.gama = 5

    def __call__(self, teacher, student, images, labels, device):
        # forward propagation
        teacher.eval()  # Teachers' networks are not updated on a gradient
        teacher_output = teacher(images)
        student_output = student(images)

        teacher_graph = features_to_graph(teacher_output)
        student_graph = features_to_graph(student_output)

        # Loss of computed side weights
        edge_cos_sim = nn.functional.cosine_similarity(student_graph.edge_weight.unsqueeze(0), teacher_graph.edge_weight.unsqueeze(0))
        edge_weight_loss = 1 - edge_cos_sim.mean()

        # Calculate node loss
        node_cos_sim = nn.functional.cosine_similarity(student_graph.x.unsqueeze(0), teacher_graph.x.unsqueeze(0))
        node_feature_loss = 1 - node_cos_sim.mean()

        # Define a hard loss function, using cross-entropy loss
        criterion_ce = nn.CrossEntropyLoss()
        hard_loss = criterion_ce(student_output, labels)

        loss_tal = hard_loss + self.gama * (edge_weight_loss + node_feature_loss)

        return loss_tal
