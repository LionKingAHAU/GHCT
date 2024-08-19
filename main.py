import setproctitle
import copy
import multiprocessing
from sklearn.model_selection import ParameterGrid
import os
import datetime
import torch
import numpy as np
import warnings
import argparse
from functools import partial

from utils import get_data_func
from trian_model import main_training_loop

# Parsing command-line arguments for configuring the script
parser = argparse.ArgumentParser(description='script')
parser.add_argument('--gpu', type=int, default=0, help='GPU number to be used for training')
parser.add_argument('--repeat', type=int, default=10,
                    help='Number of times to repeat the training for each configuration')
parser.add_argument('--use_multiprocessing', type=bool, default=False, help='Flag to enable multiprocessing')
args = parser.parse_args()
device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')  # Select device: GPU or CPU

# Set the start method for multiprocessing to 'spawn', which is safe for use with CUDA
multiprocessing.set_start_method('spawn')

# Set the process title to identify this process easily in the system
setproctitle.setproctitle("Bio_gsearch")


class Config:
    """Configuration class to store model and training parameters."""

    def __init__(self):
        # File paths for data and saving results
        self.datapath = './data/'  # Path where datasets are stored
        self.save_file = './save_result/'  # Directory to save results

        # Training parameters
        self.epochs = 200  # Number of training epochs
        self.print_epoch = 10  # Frequency of printing training status

        # Hyperparameters for training the model
        self.lr = 0.01  # Learning rate
        self.reg = 0.0005  # Regularization parameter
        self.decay = 0.985  # Learning rate decay factor
        self.decay_step = 1  # Step size for learning rate decay
        self.patience = 30  # Patience for early stopping

        # Hyperparameters for model architecture
        self.out_channels = 128  # Number of output channels in the model
        self.num_layers = 8  # Number of layers in the model
        self.num_channels = 8  # Number of channels in each layer
        self.dr = 0.5  # Dropout rate

        # Number of times to repeat the experiment (for stability)
        self.repeat = 1


def set_attr(config, param_search):
    """
    Generate configurations based on the parameter grid for hyperparameter search.

    Parameters:
    config (Config): The initial configuration object.
    param_search (dict): A dictionary defining the grid of parameters to search.

    Yields:
    Config: A new configuration object with specific parameters set according to the grid search.
    """
    # Create a list of all possible parameter combinations
    param_grid_list = list(ParameterGrid(param_search))
    for param in param_grid_list:
        # Make a deep copy of the base configuration to avoid modifying the original
        new_config = copy.deepcopy(config)
        new_config.other_args = {'arg_name': [], 'arg_value': []}  # Store additional arguments for logging or debugging

        # Iterate over each parameter and set it in the new configuration
        for key, value in param.items():
            setattr(new_config, key, value)  # Dynamically set the attribute in the config
            new_config.other_args['arg_name'].append(key)  # Store the name of the argument
            new_config.other_args['arg_value'].append(value)  # Store the value of the argument
            print(f"{key}: {value}")  # Print the parameter and its value for tracking

        yield new_config  # Yield the newly created configuration


def set_seed(seed):
    """
    Set the seed for all relevant random number generators to ensure reproducibility.

    Parameters:
    seed (int): The seed value to set.

    This function sets the seed for:
    - Torch (for PyTorch operations)
    - NumPy (for NumPy operations)
    - Python's built-in hash function (for environment stability)
    - CUDA (for GPU operations if available)
    """
    torch.manual_seed(seed)  # Set seed for PyTorch
    np.random.seed(seed)  # Set seed for NumPy
    os.environ['PYTHONHASHSEED'] = str(seed)  # Set seed for Python environment (hash functions)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # Set seed for all CUDA devices
        # Ensure deterministic operations for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def print_execution_time(start_time):
    """
    Print the execution time of the task.

    Parameters:
    start_time (datetime): The start time of the task.

    This function calculates the elapsed time since the start and prints it in a human-readable format (hours, minutes, seconds).
    """
    execution_time = datetime.datetime.now() - start_time
    hours, remainder = divmod(execution_time.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Total execution time: {hours} hours, {minutes} minutes, {seconds} seconds")


def mul_func(config, best_param_search):
    """
    Function to handle model training and evaluation for a single configuration.

    Parameters:
    config (tuple): A tuple containing the file name, configuration object, and data tuple.
    best_param_search (dict): A dictionary of the best parameters to be searched over.

    This function:
    - Sets the seed for reproducibility.
    - Constructs a file name based on the parameters for saving the results.
    - Calls the main training loop with the provided configuration and data.
    - Saves the results to disk.
    - Prints the execution time for the training process.
    """
    start_time = datetime.datetime.now()  # Record start time for tracking execution time
    warnings.filterwarnings('ignore', message='TypedStorage is deprecated.')  # Suppress specific warnings

    # Unpack the configuration parameters
    file_name, params, data_tuple = config
    set_seed(getattr(params, 'repeat'))  # Set seed based on the 'repeat' parameter for reproducibility

    save_file = file_name  # Start constructing the save file name with the dataset name

    # Append parameter settings to the file name for differentiation
    for sf in best_param_search.keys():
        if sf == 'repeat':  # Ensure that the 'repeat' parameter is included in the file name
            save_file += sf + '_' + str(getattr(params, sf))
    print(f'-----Task {save_file} started-----')  # Log the start of the task

    # Call the main training loop with the provided data and parameters
    save_list = main_training_loop(data_tuple, params, device)  # Perform cross-validation training
    save_array = np.array(save_list, dtype=object)  # Convert the list of results to a NumPy array

    # Ensure the save directory exists
    if not os.path.exists(params.save_file):
        os.makedirs(params.save_file, exist_ok=True)

    # Save the results array to a file
    np.save(os.path.join(params.save_file, f"{save_file}.npy"), save_array)

    print(f'-----Task {save_file} completed-----')  # Log the completion of the task

    print_execution_time(start_time)  # Print the execution time for this task
    del save_array  # Clean up memory by deleting the results array
    del save_list  # Clean up memory by deleting the list of results


if __name__ == '__main__':
    # Define dataset information and corresponding hyperparameters
    datasets_info = {
        'NPInter2_55': (128, 4, 8, 0.5),  # Dataset: NPInter2_55, with recommended hyperparameters
        'NPInter3.0_human_55': (128, 4, 8, 0.4),  # Dataset: NPInter3.0_human_55, with recommended hyperparameters
        'NPInter3.0_mouse_55': (128, 8, 8, 0.4)  # Dataset: NPInter3.0_mouse_55, with recommended hyperparameters
    }

    # Loop over each dataset configuration to set up hyperparameter search
    for k, v in datasets_info.items():
        best_param_search = {
            'out_channels': [v[0]],  # Set the number of output channels
            'num_layers': [v[1]],  # Set the number of layers
            'num_channels': [v[2]],  # Set the number of channels in each layer
            'dr': [v[3]],  # Set the dropout rate
            'repeat': list(range(args.repeat))  # Set the number of repetitions
        }
        data_names = k
        print(best_param_search)  # Print the parameter search space for tracking

        # Initialize the configuration object with base parameters
        params_all = Config()

        # Generate all configurations based on the parameter grid
        param_generator = set_attr(params_all, best_param_search)
        params_list = []

        file_name = data_names
        # Retrieve the dataset based on file name and device
        data_tuple = get_data_func(file_name, device=device)

        # Collect all configurations to be processed
        for params in param_generator:
            params_list.append((file_name, params, data_tuple))
            print(f"Configuration set {len(params_list)} prepared...")

        print(f"A total of {len(params_list)} configurations will be processed...")

        # Determine if multiprocessing should be used based on user input
        use_multiprocessing = args.use_multiprocessing
        if use_multiprocessing:
            # Use multiprocessing to parallelize the training process
            with multiprocessing.Pool(processes=min(len(params_list), int(2))) as pool:
                # Use partial to pass additional arguments to mul_func
                func = partial(mul_func, best_param_search=best_param_search)
                pool.map(func, params_list)  # Execute the function in parallel
        else:
            # Process each configuration sequentially without multiprocessing
            for config in params_list:
                mul_func(config, best_param_search)  # Execute the function in sequence

        print("All tasks completed.")  # Log the completion of all tasks
