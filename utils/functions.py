import os
import torch
import torch.nn as nn
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import yaml

def set_seed(seed: int = None) -> None:
    """
    Optionally, set seed for reproducibility.
        
    Parameters
    ----------
        seed: Optional[int]
            Seed.
    """
    assert seed is not None, "Invalid seed!"
    
    os.environ["PYTHONHASHSEED"] = str(seed)       # set seed for Python hashes
    np.random.seed(seed)                           # set seed for NumPy
    random.seed(seed)                              # set seed for random module
    torch.manual_seed(seed)                        # set seed for PyTorch on CPU
    if torch.cuda.is_available():                  # if GPU is available
        torch.cuda.manual_seed(seed)               # set seed for CUDA
        torch.cuda.manual_seed_all(seed)           # set seed for all GPUs
        torch.backends.cudnn.deterministic = True  # ensure reproducibility
        torch.backends.cudnn.benchmark = False     # disable non-deterministic optimisations


def plot(data: pd.DataFrame, path: str) -> None:
    # Creazione della figura e dei sottografi (subplots)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8)) #(12, 8)

    # calculate the rolling average with a window size
    window_size = 10 #5
    data['Running Average'] = data['Score'].rolling(window=window_size).mean()
    # fit a linear trend line (polynomial of degree 1)
    trend = np.polyfit(data['Episode'], data['Score'], 1)  # Linear regression
    trend_line = np.poly1d(trend)

    
    # Grafico 1: Punteggi
    axs[0, 0].plot(data['Episode'], data['Score'], label="Score", alpha=0.7, color="orange")
    axs[0, 0].plot(data['Episode'], data['Running Average'], label=f'Running Avg. (Window={window_size})', color="darkorange", linewidth=2)
    axs[0, 0].plot(data['Episode'], trend_line(data['Episode']), label='Trend', color="orangered", linestyle='--', alpha=0.8)
    axs[0, 0].set_title("Score")
    axs[0, 0].set_xlabel("Episode")
    axs[0, 0].set_ylabel("Score")
    axs[0, 0].grid(True)
    axs[0, 0].legend()
    
    # Grafico 2: Loss
    axs[0, 1].plot(data['Episode'], data['Loss'], label="Loss", color="red")
    axs[0, 1].set_title("Loss")
    axs[0, 1].set_xlabel("Episode")
    axs[0, 1].set_ylabel("Loss")
    axs[0, 1].grid(True)
    #axs[0, 1].legend()
    
    # Grafico 3: Epsilon
    axs[1, 0].plot(data['Episode'], data['Epsilon'], label="Epsilon", color="green")
    axs[1, 0].set_title("Epsilon")
    axs[1, 0].set_xlabel("Episode")
    axs[1, 0].set_ylabel("Epsilon")
    axs[1, 0].grid(True)
    #axs[1, 0].legend()
    
    # Grafico 4: Learning Rate
    axs[1, 1].plot(data['Episode'], data['LearningRate'],  label="Learning Rate", color="blue")
    axs[1, 1].set_title("Learning Rate")
    axs[1, 1].set_xlabel("Episode")
    axs[1, 1].set_ylabel("Learning Rate")
    axs[1, 1].grid(True)
    #axs[1, 1].legend()
    
    # Miglioriamo il layout e salviamo il grafico
    plt.tight_layout()
    plt.savefig(path, dpi=300)


def hard_update(main_model: nn.Module, target_model: nn.Module) -> None:
    """
    Update target model parameters:
        
    θ_target = θ_main
            
    Parameters
    ----------
        main_model: nn.Module
            Weights will be copied from.
        target_model: nn.Module
            Weights will be copied to.
    """
    target_model.load_state_dict(main_model.state_dict())
    
    
def soft_update(main_model: nn.Module, target_model: nn.Module, tau: float) -> None:
    """
    Soft update target model parameters:
        
    θ_target = τ*θ_main + (1 - τ)*θ_target
        
    Parameters
    ----------
        main_model: nn.Module
            Weights will be copied from.
        target_model: nn.Module
            Weights will be copied to.
        tau: float
            Interpolation parameter.
    """    
    for target_param, main_param in zip(target_model.parameters(), main_model.parameters()):
        target_param.data.copy_(tau*main_param.data + (1.0-tau)*target_param.data)


def read_hyperparameters() -> dict:
    """
    Read the hyperparameters from .yml file.

    Returns
    -------
        dict
            Dictionary of hyperparameters.
    """
    with open('hyperparameters.yml', 'r') as file:
        args = yaml.safe_load(file)
    return args
    #LR, type=float, required=True, help="Learning Rate"
    #LR_STEP_SIZE, type=int, default=None, help="Period of learning rate decay. If None, no lr decayment is performed"
    #LR_GAMMA, type=float, default=None, help="Multiplicative factor of learning rate decay. If None, no lr decayment is performed"
    #BUFFER_SIZE, type=int, required=True, help="Size of the Experience Replay buffer"
    #BATCH_SIZE, type=int, required=True, help="Size of the mini-batch sampled from the Replay buffer"
    #GAMMA, type=float, required=True, help="Discount factor"
    #N_EPISODES, type=int, default=None, help="Number of episodes for training"
    #HARD_UPDATE_EVERY, type=int, default=None, help="After this number of steps/updates, hard update the target model. If None, soft update is performed"
    #TAU, type=float, required=True, help="Multiplicative factor for soft update."
    #EPSILON_START, type=float, required=True, help="Initial epsilon"
    #EPSILON_END, type=float, required=True, help="Final epsilon"
    #EPSILON_DECAY, type=float, default=None, help="Factor of epsilon decayment. If None, epsilon decays linearly w.r.t the number of episodes"
    #STATE_SIZE, type=int, required=True, help="State dimension"
    #ACTION_SIZE, type=int, required=True, help="Number of actions"
    #HIDDEN_LAYERS, type=int, nargs="+", required=True, help="Number of neurons (e.g. 256 128 64)") # nargs="+" per avere una lista da 1 a più elementi
    #SEED, type=int, default=None, help="Set seed if deterministic behaviour is needed, ignore otherwise"
    #TRAIN_MODE, type=str2bool, default=True, help='True: Train mode; False: Evaluation mode'
    #BEST_MODEL_PATH, type=str, required=True, help='Path (with extension .pt) for saving the best model'
    #LOG_PATH, type=str, required=True, help='Path for saving the best model'
    #PLOT_PATH, type=str, required=True, help='Path for saving the learning curves plot'