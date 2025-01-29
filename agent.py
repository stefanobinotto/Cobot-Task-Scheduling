import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import numpy as np
import tqdm

from models.dqn import DQN
from memory.replay import ReplayBuffer
from env.env_v2 import CobotEnv

class Agent:
    def __init__(self, hyperparameters: dict):

        self.hyp = hyperparameters
        #self.gamma = hyperparameters['GAMMA']
        #self.tau = hyperparameters['TAU']
        #self.state_size = hyperparameters['STATE_SIZE']
        #self.action_size = hyperparameters['ACTION_SIZE']
        #self.epsilon_start = hyperparameters['EPSILON_START']
        #self.epsilon_end = hyperparameters['EPSILON_END']
        #self.epsilon_decay = hyperparameters['EPSILON_DECAY']
        #self.lr = hyperparameters['LR']
        #self.lr_step_size = hyperparameters['LR_STEP_SIZE']
        #self.lr_gamma = hyperparameters['LR_GAMMA']
        #self.buffer_size = hyperparameters['BUFFER_SIZE']
        #self.batch_size = hyperparameters['BATCH_SIZE']
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Policy network and Target Network
        self.policy_net = DQN(hyp['STATE_SIZE'], hyp['HIDDEN_LAYERS'], hyp['ACTION_SIZE']).to(self.device)
        self.target_net = DQN(hyp['STATE_SIZE'], hyp['HIDDEN_LAYERS'], hyp['ACTION_SIZE']).to(self.device)
        
        for p in self.target_net.parameters(): # freeze params
            p.requires_grad = False
            
        hard_update(self.policy_net, self.target_net)

        # loss function and optimization algorithm
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=hyp['LR'])
        
        if (self.lr_gamma is not None) and (self.lr_step_size is not None):
            self.scheduler = StepLR(self.optimizer, step_size=hyp['LR_STEP_SIZE'], gamma=hyp['LR_GAMMA']) # PROVARE ANCHE REDUCE LR ON PLATEAU

        # Experience Replay
        self.memory = ReplayBuffer(max_capacity=hyp['BUFFER_SIZE'], batch_size=hyp['BATCH_SIZE'], device=self.device)


    def act(self, state, deterministic) -> int:


    def learn():
        #prendi un batch ed imparalo


    def hard_update(self, main_model: nn.Module, target_model: nn.Module) -> None:
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
    
    
    def soft_update(self, main_model: nn.Module, target_model: nn.Module, tau: float) -> None:
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

    
    def train(n_episodes: int):

        # instance of the environment
        env = CobotEnv()

        # list to keep track of rewards collected per episode
        rewards_history = []
        
        # list to keep track of epsilon decay
        epsilon_history = []

        epsilon = self.hyp['EPSILON_START']

        step = 0

        for e in tqdm(range(1, n_episodes+1), desc = "Episode #:"):
            state = env.reset()
            rewards = 0
            done = False
            while not done:
                action = agent.act(state, deterministic=False) # select action
                
            

            







        