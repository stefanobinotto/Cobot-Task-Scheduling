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


def single_run_plot(data: pd.DataFrame, path: str) -> None:
    # Figure and subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 8))

    # calculate the rolling average with a window size
    window_size = 10 #5
    data['Running Average'] = data['Score'].rolling(window=window_size).mean()
    # fit a linear trend line (polynomial of degree 1)
    trend = np.polyfit(data['Episode'], data['Score'], 1)  # Linear regression
    trend_line = np.poly1d(trend)

    # filter only model saving episodes
    saved_episodes = data[data['Model saving'] == 'Saved']
    
    # Grafico 1: Punteggi
    axs[0, 0].plot(data['Episode'], data['Score'], label="Score", alpha=0.7, color="orange")
    axs[0, 0].plot(data['Episode'], data['Running Average'], label=f'Running avg. (window={window_size})', color="darkorange", linewidth=2)
    axs[0, 0].plot(data['Episode'], trend_line(data['Episode']), label='Trend', color="orangered", linestyle='--', alpha=0.8)
    # model saving marker
    axs[0, 0].scatter(
        saved_episodes['Episode'], 
        saved_episodes['Score'], 
        color='green', 
        label='Model saved', 
        s=50, 
        marker='*',
        linewidth=0
    )
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
    axs[1, 1].plot(data['Episode'], data['Learning rate'],  label="Learning Rate", color="blue")
    axs[1, 1].set_title("Learning Rate")
    axs[1, 1].set_xlabel("Episode")
    axs[1, 1].set_ylabel("Learning Rate")
    axs[1, 1].grid(True)
    #axs[1, 1].legend()
    
    # Miglioriamo il layout e salviamo il grafico
    plt.tight_layout()
    plt.savefig(path, dpi=300)


def plot(all_scores: list, all_losses: list, epsilons: list, path: str) -> None:
    # Figure and subplots
    #_, axs = plt.subplots(3, 1, figsize=(8, 6))
    _, axs = plt.subplots(1, 1, figsize=(6, 4))

    # mean and std
    df = pd.DataFrame()
    df['mean_scores'] = pd.DataFrame(np.mean(np.array(all_scores), axis=0))
    df['std_scores'] = pd.DataFrame(np.std(np.array(all_scores), axis=0))
    df['mean_losses'] = pd.DataFrame(np.mean(np.array(all_losses), axis=0))
    df['std_losses'] = pd.DataFrame(np.std(np.array(all_losses), axis=0))
    # Compute rolling averages
    window_size = 10
    df['rolling_mean_scores'] = df['mean_scores'].rolling(window=window_size).mean()
    df['rolling_std_scores'] = df['std_scores'].rolling(window=window_size).mean()
    df['rolling_mean_losses'] = df['mean_losses'].rolling(window=window_size).mean()
    df['rolling_std_losses'] = df['std_losses'].rolling(window=window_size).mean()

    # Score
    axs[0].plot(df['rolling_mean_scores'])#, label=f"Rolling avg. (w={window_size})", color="blue")   uncomment for label
    axs[0].fill_between(
        range(len(df['rolling_mean_scores'])),
        df['rolling_mean_scores'] - df['rolling_std_scores'],
        df['rolling_mean_scores'] + df['rolling_std_scores'],
        color='blue',
        alpha=0.2,
        #label='Std dev',         uncomment for label
        linewidth=0
    )
    #axs[0].set_title("Score")
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Score (min)")
    axs[0].grid(True)
    axs[0].legend()
    
    # Loss
    #axs[1].plot(df['rolling_mean_losses'], label=f"Rolling avg. (w={window_size})", color="red")
    #axs[1].fill_between(
    #    range(len(df['rolling_mean_losses'])),
    #    df['rolling_mean_losses'] - df['rolling_std_losses'],
    #    df['rolling_mean_losses'] + df['rolling_std_losses'],
    #    color='red',
    #    alpha=0.2,
    #    label='Std dev',
    #    linewidth=0
    #)
    #axs[1].set_title("Loss")
    #axs[1].set_xlabel("Episode")
    #axs[1].set_ylabel("Loss")
    #axs[1].grid(True)
    #axs[1].legend()

    # Grafico 3: Epsilon
    #axs[2].plot(epsilons, label="Epsilon", color="green")
    #axs[2].set_title("Epsilon")
    #axs[2].set_xlabel("Episode")
    #axs[2].set_ylabel("Epsilon")
    #axs[2].grid(True)
    #axs[1, 0].legend()
    
    # Miglioriamo il layout e salviamo il grafico
    plt.tight_layout()
    plt.savefig(path, dpi=600)


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