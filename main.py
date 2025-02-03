from utils.functions import read_hyperparameters, set_seed
from agent import Agent

def main():
    # read hyperparams from file using '@' in CLI (e.g. python main.py @args.txt)
    hyperparameters = read_hyperparameters()
    print(hyperparameters["HIDDEN_LAYERS"])
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