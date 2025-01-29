import os
import torch
import random
import numpy as np

def set_seed(seed: int = None) -> None:
    """
    Optionally, set seed for reproducibility.
        
    Parameters
    ----------
        seed: Optional[int]
            Seed.
    """
    assert seed is not None, "Invalid seed!"
    
    os.environ["PYTHONHASHSEED"] = str(seed)  # set seed for Python hashes
    np.random.seed(seed)                     # set seed for NumPy
    random.seed(seed)                        # set seed for random module
    torch.manual_seed(seed)                  # set seed for PyTorch on CPU
    if torch.cuda.is_available():            # if GPU is available
        torch.cuda.manual_seed(seed)         # set seed for CUDA
        torch.cuda.manual_seed_all(seed)     # set seed for all GPUs
        torch.backends.cudnn.deterministic = True  # ensure reproducibility
        torch.backends.cudnn.benchmark = False     # disable non-deterministic optimisations


def read_hyperparameters() -> dict:
    """
    Read the hyperparameters from textfile.

    Returns
    -------
        dict
            Dictionary of hyperparameters.
    """
    parser = argparse.ArgumentParser(description="Read hyperparameters")

    # Configuration to read arguments from a file with a specific prefix (e.g. '@')
    parser.fromfile_prefix_chars = "@"

    # Hyperparameter Setting
    parser.add_argument("--LR", type=float, required=True, help="Learning Rate")
    parser.add_argument("--LR_STEP_SIZE", type=int, default=None, help="Period of learning rate decay. Ignore if no lr decay is performed")
    parser.add_argument("--LR_GAMMA", type=float, default=None, help="Multiplicative factor of learning rate decay. Ignore if no lr decayment is performed")
    parser.add_argument("--BUFFER_SIZE", type=int, required=True, help="Size of the Experience Replay buffer")
    parser.add_argument("--BATCH_SIZE", type=int, required=True, help="Size of the mini-batch sampled from the Replay buffer")
    parser.add_argument("--GAMMA", type=float, required=True, help="Discount factor")
    parser.add_argument("--TAU", type=float, default=None, help="Multiplicative factor for soft update. Ignore if hard update is chosen")
    parser.add_argument("--EPSILON_START", type=float, required=True, help="Initial epsilon")
    parser.add_argument("--EPSILON_END", type=float, required=True, help="Final epsilon")
    parser.add_argument("--EPSILON_DECAY", type=float, required=True, help="Factor of epsilon decayment")
    parser.add_argument("--STATE_SIZE", type=int, required=True, help="State dimension")
    parser.add_argument("--ACTION_SIZE", type=int, required=True, help="Number of actions")
    parser.add_argument("--HIDDEN_LAYERS", nargs="+", type=int, required=True, help="Number of neurons (e.g. 256 128 64)") # nargs="+" per avere una lista da 1 a più elementi
    parser.add_argument("--SEED", type=int, default=None, help="Set seed if deterministic behaviour is needed, ignore otherwise")
#    parser.add_argument('--DDQN', type=str2bool, default=True, help='True:DDQN; False:DQN')

    # hyperparameters parsing
    args = parser.parse_args()
    
    return vars(args)


def str2bool(v) -> bool:
    '''
    Transfer str to bool for argparse:
    ('yes', 'True','true','TRUE', 't', 'y', '1') -> True
    ('no', 'False','false','FALSE', 'f', 'n', '0') -> False

    Returns
    -------
        bool
    '''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes','True','true','TRUE','t','y','1'):
        return True
    elif v.lower() in ('no','False','false','FALSE','f','n','0'):
        return False
    else:
        raise TypeError("Only 'yes', 'True', 'true', 'TRUE', 't', 'y', '1' or 'no', 'False', 'false', 'FALSE', 'f', 'n', '0' are allowed!")