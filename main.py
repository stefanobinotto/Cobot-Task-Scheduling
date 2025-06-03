from utils import read_hyperparameters, set_seed, plot
from agent import Agent
import torch


def main():
    # import hyperparameters
    hyperparameters = read_hyperparameters()

    ### TEST
    
    if hyperparameters['MODEL_PATH'] is not None:
        agent = Agent(hyperparameters)
        agent.test()
        exit(0)

    ### TRAIN
    
    # deterministic behaviour
    if hyperparameters['SEED'] is not None and hyperparameters['N_EXPERIMENTS'] == 1:
        set_seed(hyperparameters['SEED'])
        print("Setting seed!")

    all_scores = []
    all_losses = []
    for i in range(hyperparameters['N_EXPERIMENTS']):
        agent = Agent(hyperparameters)
        logs = agent.train()
        logs.to_csv(hyperparameters['LOG_PATH']+"op_"+str(hyperparameters['ID_OPERATOR'])+"_training_run_"+str(i+1)+".csv", index=False)
        all_scores.append(logs['Score'])
        all_losses.append(logs['Loss'])
        # free CUDA memory
        del agent
        torch.cuda.empty_cache()

    plot(all_scores, all_losses, logs['Epsilon'], hyperparameters['LOG_PATH']+"op_"+str(hyperparameters['ID_OPERATOR'])+"_curves.pdf")  


if __name__ == "__main__":
    main()