import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch


class LayerGT(nn.Module):
    """
    Parameters:
    - out_channels: The number of output channels (features) for each node.
    - num_channels: The number of parallel GCNConv layers in this layer.
    - dr: Dropout rate applied to the output of the GCNConv layers.
    - device: The device (CPU/GPU) on which the model is run.

    This class:
    - Applies Leaky ReLU activation and dropout to the output.
    - Combines the outputs of multiple GCNConv layers to form a comprehensive feature representation.
    """

    def __init__(self, out_channels, num_channels, dr=12, device=None):
        super(LayerGT, self).__init__()
        self.device = device
        # Initialize multiple GCNConv layers based on the number of channels
        self.convs = nn.ModuleList([GCNConv(out_channels, out_channels) for _ in range(num_channels)])

        # Initialize two fully connected layers for feature transformation
        self.fc1 = nn.Linear(out_channels * num_channels, out_channels * num_channels, bias=False)
        self.fc2 = nn.Linear(out_channels * num_channels, out_channels * num_channels, bias=False)
        self.dropout = nn.Dropout(dr)
        self.out_channels = out_channels
        self.num_channels = num_channels

        # Move the model to the specified device
        self.to(self.device)

    def forward(self, x, edge_index):
        """
        Forward pass for the LayerGT.

        Parameters:
        - x: Node features tensor, expected to be of shape (num_nodes, out_channels * num_channels).
        - edge_index: Edge index tensor defining the graph structure.

        Returns:
        - output: The transformed node features after applying GCNConv layers, activations,
          dropout, and linear transformations.

        The function:
        - Reshapes the input features.
        - Combines the results, applies Leaky ReLU activation and dropout.
        - Applies linear transformations and computes the final output.
        """
        x_reshaped = x.reshape(-1, self.out_channels, self.num_channels)
        z = []
        for i, conv in enumerate(self.convs):
            z.append(F.leaky_relu(conv(x_reshaped[:, :, i], edge_index)))
        z = torch.cat(z, dim=1)
        z = F.leaky_relu(z)
        z = self.dropout(z)

        z_fc1 = self.fc1(z)
        z_fc2 = self.fc2(z)

        s = torch.sum(z_fc1.T @ z_fc2, dim=0)
        output = z * F.softmax((s / torch.sqrt(torch.tensor([s.shape[0]]).to(self.device))).unsqueeze(0), dim=0).expand(
            z.shape[0], -1)

        return output


class MGHCT(nn.Module):
    """
    Parameters:
    - in_channels: Number of input channels (features) for each node.
    - out_channels: Number of output channels (features) for each node in each GCNConv layer.
    - num_layers: Number of LayerGNN layers in the model.
    - num_channels: Number of parallel GCNConv layers within each LayerGNN layer.
    - dr: Dropout rate applied in each LayerGNN layer.
    - device: The device (CPU/GPU) on which the model is run.

    This class:
    - Initializes a series of GCNConv layers for initial processing of node features.
    - Stacks multiple LayerGNN layers for deeper feature extraction.
    - Applies residual connections to combine features from different layers.
    """

    def __init__(self, in_channels, out_channels=128, num_layers=8, num_channels=8, dr=0.5, device=None):
        super(MGHCT, self).__init__()
        self.device = device

        self.initial_convs = nn.ModuleList([GCNConv(in_channels, out_channels) for _ in range(num_channels)])
        self.fc1 = nn.Linear(out_channels * num_channels, out_channels * num_channels, bias=False)
        self.fc2 = nn.Linear(out_channels * num_channels, out_channels * num_channels, bias=False)
        self.dropout = nn.Dropout(dr)
        self.out_channels = out_channels

        self.layer_gnns = nn.ModuleList(
            [LayerGT(out_channels, num_channels, dr=dr, device=self.device) for _ in range(num_layers - 1)])

        # Initialize residual connections as additional GCNConv layers
        self.res_gnn = nn.ModuleList(
            [GCNConv(out_channels * num_channels, out_channels * num_channels).to(self.device) for _ in
             range(num_layers)])

        self.res_beta = 1  # Residual scaling factor

        # Move the model to the specified device
        self.to(self.device)

    def forward(self, data):
        """
        Parameters:
        - data: The input data containing node features (x) and graph structure (edge_index).

        Returns:
        - output: The combined node features after passing through all layers of the model.

        The function:
        - Applies LayerGT layers iteratively with residual connections.
        - Combines the outputs from all layers to form the final node representations.
        """
        x, edge_index = data.x, data.edge_index  # Extract node features and edge index
        initial_z = []

        # Apply initial GCNConv layers to process node features
        for conv in self.initial_convs:
            initial_z.append(F.leaky_relu(conv(x, edge_index)))

        # Concatenate the outputs from initial GCNConv layers
        initial_z = torch.cat(initial_z, dim=1)
        initial_z = F.leaky_relu(initial_z)
        initial_z = self.dropout(initial_z)

        # Apply fully connected layers and compute softmax for weighting
        z_fc1 = self.fc1(initial_z)
        z_fc2 = self.fc2(initial_z)
        s = torch.sum(z_fc1.T @ z_fc2, dim=0)
        c = initial_z * F.softmax((s / torch.sqrt((torch.tensor([s.shape[0]]).to(self.device)))).unsqueeze(0),
                                  dim=0).expand(initial_z.shape[0], -1)

        outputs = [c]  # Start with the initial weighted output

        # Iteratively apply LayerGT layers with residual connections
        for i, layer_gnn in enumerate(self.layer_gnns):
            outputs.append(layer_gnn(outputs[-1], edge_index) + F.softmax(self.res_gnn[i + 1](outputs[-1], edge_index),
                                                                          dim=-1) * self.res_beta)

        # Concatenate the outputs from all layers to form the final node representations
        return torch.cat(outputs, dim=1)
