import torch.nn.functional as F
import torch
from utils import get_metrics
from model import MGHCT


def train(model, data, optimizer, pos_edge_index, neg_edge_index, device):
    """
    Train the model for one epoch.

    Parameters:
    - model: The PyTorch model to be trained.
    - data: The input data containing node features and graph structure.
    - optimizer: The optimizer used to update the model parameters.
    - pos_edge_index: The edge index tensor for positive samples (i.e., real edges in the graph).
    - neg_edge_index: The edge index tensor for negative samples (i.e., randomly sampled non-existent edges).
    - device: The device (CPU/GPU) on which to perform training.

    Returns:
    - loss.item(): The scalar loss value after one training step.

    The function:
    - Passes the input data through the model to obtain node embeddings.
    - Computes dot products of embeddings for positive and negative edge pairs.
    - Calculates binary cross-entropy loss for positive and negative samples.
    - Performs backpropagation and updates the model parameters.
    """
    model.train()
    optimizer.zero_grad()

    z = model(data)

    # Calculate the dot products for positive and negative edge pairs
    pos_out = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=1)
    neg_out = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)

    # Compute binary cross-entropy loss
    pos_loss = F.binary_cross_entropy_with_logits(pos_out, torch.ones(pos_out.size(0)).to(device))
    neg_loss = F.binary_cross_entropy_with_logits(neg_out, torch.zeros(neg_out.size(0)).to(device))
    loss = pos_loss + neg_loss

    # Perform backpropagation and update model parameters
    loss.backward()
    optimizer.step()

    return loss.item()


def test(model, data, pos_edge_index, neg_edge_index, device):
    """
    Evaluate the model on test data.

    Parameters:
    - model: The PyTorch model to be evaluated.
    - data: The input data containing node features and graph structure.
    - pos_edge_index: The edge index tensor for positive samples (i.e., real edges in the graph).
    - neg_edge_index: The edge index tensor for negative samples (i.e., randomly sampled non-existent edges).
    - device: The device (CPU/GPU) on which to perform evaluation.

    Returns:
    - true_score: Tensor of true labels (1 for positive edges, 0 for negative edges).
    - predict_score: Tensor of predicted scores from the model.

    The function:
    - Passes the input data through the model to obtain node embeddings.
    - Computes dot products for positive and negative edge pairs, followed by a sigmoid activation.
    - Concatenates the results to form true and predicted scores for further evaluation.
    """
    model.eval()
    with torch.no_grad():
        z = model(data)
        pos_out = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=1).sigmoid()
        neg_out = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1).sigmoid()
        pos_out2 = (z[pos_edge_index[1]] * z[pos_edge_index[0]]).sum(dim=1).sigmoid()
        neg_out2 = (z[neg_edge_index[1]] * z[neg_edge_index[0]]).sum(dim=1).sigmoid()

        # Concatenate true labels and predicted scores
        true_score = torch.cat([torch.ones(pos_out.size(0) * 2), torch.zeros(neg_out.size(0) * 2)]).cpu()
        predict_score = torch.cat([pos_out, pos_out2, neg_out, neg_out2]).cpu()

        return true_score, predict_score


def evaluate_model(model, data, pos_edge_index, neg_edge_index):
    """
    Evaluate the model using various metrics.

    Parameters:
    - model: The PyTorch model to be evaluated.
    - data: The input data containing node features and graph structure.
    - pos_edge_index: The edge index tensor for positive samples (i.e., real edges in the graph).
    - neg_edge_index: The edge index tensor for negative samples (i.e., randomly sampled non-existent edges).

    Returns:
    - metrics: A list of calculated metrics.
    - metric_names: A list of names corresponding to the calculated metrics.

    The function:
    - Calls the `test` function to obtain true and predicted scores.
    - Computes various metrics using the `get_metrics` function.
    - Returns the metrics and their corresponding names.
    """
    true_score, predict_score = test(model, data, pos_edge_index, neg_edge_index)
    metrics, metric_names = get_metrics(true_score, predict_score)
    return metrics, metric_names


def main_training_loop(data_tuple, params, device):
    """
    Main training loop for training and evaluating the model.

    Parameters:
    - data_tuple: A tuple containing training and test data, as well as edge information.
    - params: The configuration object containing model and training parameters.
    - device: The device (CPU/GPU) on which to perform training.

    Returns:
    - save_list: A list containing metrics for each evaluated epoch.

    The function:
    - Initializes the model and optimizer based on provided parameters.
    - Iteratively trains the model for a specified number of epochs.
    - Evaluates the model on test data at specified intervals.
    - Tracks and saves the best AUC score during training.
    """
    save_list = []
    (data, train_data, test_data,
     train_pos_edges, train_neg_edges,
     test_pos_edges, test_neg_edges) = data_tuple  # Unpack the data tuple
    num_epochs = params.epochs  # Number of epochs to train
    max_auc = 0  # Variable to store the maximum AUC score

    model = MGHCT(in_channels=data.num_features,
                  out_channels=params.out_channels,
                  num_layers=params.num_layers,
                  num_channels=params.num_channels,
                  dr=params.dr,
                  device=device).to(device)

    # Initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)

    # Training loop
    for epoch in range(num_epochs):
        loss = train(model, train_data, optimizer, train_pos_edges, train_neg_edges, device=device)

        # Evaluate the model at the specified print interval
        if epoch % params.print_epoch == 0:
            true_score, predict_score = test(model, test_data, test_pos_edges, test_neg_edges, device=device)
            metrics, metric_names = get_metrics(true_score, predict_score)
            auc_score = metrics[2]  # Extract AUC score from metrics
            max_auc = max(max_auc, auc_score)  # Track the maximum AUC score
            print(f'Epoch {epoch}, Loss: {loss:.4f}, Test AUC: {auc_score:.4f}')

            # Append additional information to the metrics for logging
            metrics.extend([
                data.num_features, params.num_layers,
                params.num_channels, params.out_channels,
                params.dr, params.repeat, epoch])
            save_list.append(metrics)

    print(f'Max AUC: {max_auc:.4f}')  # Print the best AUC score achieved during training

    # Clean up to free memory
    del model
    del optimizer
    torch.cuda.empty_cache()  # Clear unused CUDA memory cache

    return save_list  # Return the list of metrics collected during training
