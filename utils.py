from torch_geometric.utils import from_networkx, negative_sampling, add_self_loops
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, accuracy_score, recall_score, \
    precision_score
import pandas as pd
import networkx as nx


def get_data_func(file_name='NPInter2_55', device=None):
    """
    Load and process the training and test datasets.

    Parameters:
    - file_name: Name of the dataset directory containing the train and test data files.
    - device: The device (CPU/GPU) on which the data will be loaded.

    Returns:
    - data_tuple: A tuple containing the processed graph data, training and test data,
      and positive/negative edge indices for training and testing.

    The function:
    - Loads the train and test datasets from text files.
    - Constructs a bipartite graph with lncRNA and protein nodes.
    - Converts the graph into PyTorch Geometric format and prepares it for GNN processing.
    - Generates positive and negative edge samples for training and testing.
    - Clones the graph data for separate use in training and testing.
    """
    # Load the training and test data from files
    train_file_path = f'./data/{file_name}/train.txt'
    test_file_path = f'./data/{file_name}/test.txt'

    train_df = pd.read_csv(train_file_path, header=None, sep='\t', names=['lncRNA', 'protein', 'interaction'])
    test_df = pd.read_csv(test_file_path, header=None, sep='\t', names=['lncRNA', 'protein', 'interaction'])

    # Create an empty undirected graph
    G = nx.Graph()

    # Extract unique lncRNA and protein nodes from train and test datasets
    lncRNA_nodes = sorted(set(train_df['lncRNA']).union(set(test_df['lncRNA'])))
    protein_nodes = sorted(set(train_df['protein']).union(set(test_df['protein'])))

    # Assign unique indices to each lncRNA and protein node
    lncRNA_to_index = {node: i for i, node in enumerate(lncRNA_nodes)}
    protein_to_index = {node: i + len(lncRNA_nodes) for i, node in enumerate(protein_nodes)}

    # Add lncRNA nodes to the graph and mark them as part of the first bipartite set
    G.add_nodes_from(lncRNA_to_index.values(), bipartite=0)
    # Add protein nodes to the graph and mark them as part of the second bipartite set
    G.add_nodes_from(protein_to_index.values(), bipartite=1)

    # Add edges from the training set (only those with interaction = 1)
    train_edges = [(lncRNA_to_index[row['lncRNA']], protein_to_index[row['protein']]) for _, row in train_df.iterrows()
                   if
                   row['interaction'] == 1]
    G.add_edges_from(train_edges)

    # Convert the graph into PyTorch Geometric data format
    data = from_networkx(G)
    data.x = torch.eye(data.num_nodes)  # Create an identity matrix for node features
    data.batch = torch.zeros(data.num_nodes, dtype=torch.long)  # Assign all nodes to the same batch

    # Prepare positive and negative edges for training and testing
    train_pos_edges = torch.tensor(
        [(lncRNA_to_index[row['lncRNA']], protein_to_index[row['protein']]) for _, row in train_df.iterrows() if
         row['interaction'] == 1], dtype=torch.long).t().contiguous()

    # Generate negative samples for training
    train_neg_edges = negative_sampling(edge_index=train_pos_edges, num_nodes=data.num_nodes,
                                        num_neg_samples=train_pos_edges.size(1)).to(device)

    # Prepare positive edges for testing
    test_pos_edges = torch.tensor(
        [(lncRNA_to_index[row['lncRNA']], protein_to_index[row['protein']]) for _, row in test_df.iterrows() if
         row['interaction'] == 1], dtype=torch.long).t().contiguous()

    # Prepare negative edges for testing
    test_neg_edges = torch.tensor(
        [(lncRNA_to_index[row['lncRNA']], protein_to_index[row['protein']]) for _, row in test_df.iterrows() if
         row['interaction'] == 0], dtype=torch.long).t().contiguous()

    # Clone the data for separate training and testing datasets
    train_data = data.clone()
    train_data.edge_index = train_pos_edges

    test_data = data.clone()
    test_data.edge_index = test_pos_edges

    # Move all data to the specified device (CPU/GPU)
    train_data = train_data.to(device)
    test_data = test_data.to(device)
    train_pos_edges = train_pos_edges.to(device)
    test_pos_edges = test_pos_edges.to(device)
    test_neg_edges = test_neg_edges.to(device)

    return (data, train_data, test_data, train_pos_edges, train_neg_edges, test_pos_edges, test_neg_edges)


def get_metrics(true_score, predict_score):
    """
    Compute evaluation metrics for the model's predictions.

    Parameters:
    - true_score: Ground truth binary labels (1 for positive, 0 for negative).
    - predict_score: Predicted scores (e.g., probabilities or logits) from the model.

    Returns:
    - metrics: A list of computed metric values including AUC, AUPR, F1-score, accuracy, recall, specificity, and precision.
    - metric_names: A list of corresponding metric names.

    The function:
    - Computes AUC (Area Under the ROC Curve) and AUPR (Area Under the Precision-Recall Curve).
    - Determines the best threshold for binary classification based on F1-score.
    - Calculates additional metrics including accuracy, recall, specificity, and precision using the best threshold.
    """
    true_score = np.array(true_score)
    predict_score = np.array(predict_score)

    # Compute AUC (Area Under the ROC Curve)
    auc_score = roc_auc_score(true_score, predict_score)

    # Compute Precision-Recall Curve and AUPR (Area Under the Precision-Recall Curve)
    precision, recall, _ = precision_recall_curve(true_score, predict_score)
    aupr_score = auc(recall, precision)

    # Find the best threshold based on F1 score
    f1_scores = []
    thresholds = []
    for t in range(0, 10):
        threshold = t / 10.0
        binary_predict_score = (predict_score > threshold).astype(int)
        f1 = f1_score(true_score, binary_predict_score)
        f1_scores.append(f1)
        thresholds.append(threshold)

    # Determine the threshold that yields the highest F1-score
    best_threshold = thresholds[f1_scores.index(max(f1_scores))]

    # Use the best threshold for final binary classification
    binary_predict_score = (predict_score > best_threshold).astype(int)
    f1_score_value = f1_score(true_score, binary_predict_score)
    accuracy = accuracy_score(true_score, binary_predict_score)
    recall_value = recall_score(true_score, binary_predict_score)
    specificity = recall_score(1 - true_score, 1 - binary_predict_score)
    precision_value = precision_score(true_score, binary_predict_score)

    # Print out the computed metrics
    print(
        'auc:{:.4f}, aupr:{:.4f}, f1_score:{:.4f}, accuracy:{:.4f}, recall:{:.4f}, specificity:{:.4f}, precision:{:.4f}, best_threshold:{:.2f}'.format(
            auc_score, aupr_score, f1_score_value, accuracy, recall_value, specificity, precision_value,
            best_threshold))

    return [true_score, predict_score, auc_score, aupr_score, f1_score_value, accuracy, recall_value, specificity,
            precision_value], \
        ['y_true', 'y_score', 'auc', 'prc', 'f1_score', 'acc', 'recall', 'specificity', 'precision']
