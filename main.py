import argparse
from utils.functions import read_hyperparameters, set_seed
import Agent

def main():
    # read hyperparams from file using '@' in CLI (e.g. python main.py @args.txt)
    hyperparameters = read_hyperparameters()

    # deterministic behaviour
    if hyperparameters['SEED'] is not None:
        set_seed(hyperparameters['SEED'])
    
    agent = Agent(hyperparameters)
    agent.train()


if __name__ == "__main__":
    main()