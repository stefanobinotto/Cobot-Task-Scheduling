from utils.functions import read_hyperparameters, set_seed
from agent import Agent

def main():
    # import hyperparameters
    hyperparameters = read_hyperparameters()
    # deterministic behaviour
    if hyperparameters['SEED'] is not None:
        set_seed(hyperparameters['SEED'])
    
    agent = Agent(hyperparameters)
    if hyperparameters["TRAIN_MODE"]:
        agent.train(hyperparameters["N_EPISODES"] )
    else:
        print("entra qui!")
        #agent.evaluate()


if __name__ == "__main__":
    main()