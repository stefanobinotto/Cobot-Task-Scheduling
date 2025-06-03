import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import numpy as np
from tqdm import tqdm
import pandas as pd
import random

from collections import deque

# my packages
from models import DQN
from env import CobotEnv
from utils import soft_update, hard_update
from memory import ReplayBuffer


class Agent:
    def __init__(self, hyperparameters: dict):
        
        self.hp = hyperparameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device used:",self.device)

        # policy network
        self.policy_net = DQN(self.hp['INPUT_SIZE'], self.hp['HIDDEN_LAYERS'], self.hp['ACTION_SIZE']).to(self.device)
        # target network
        self.target_net = DQN(self.hp['INPUT_SIZE'], self.hp['HIDDEN_LAYERS'], self.hp['ACTION_SIZE']).to(self.device)
        hard_update(self.policy_net, self.target_net)
        for p in self.target_net.parameters(): # freeze parameters
            p.requires_grad = False
        
        # loss function
        self.loss_fn = nn.MSELoss()
        # optimization algorithm
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.hp['LR'])
        # reduce lr
        if self.hp['LR_STEP_SIZE'] is not None or self.hp['LR_FACTOR'] is not None:
            # STEP LR
            #self.scheduler = StepLR(self.optimizer, step_size=self.hp['LR_STEP_SIZE'], gamma=self.hp['LR_FACTOR'])
            # REDUCE LR ON PLATEAU
            self.scheduler = ReduceLROnPlateau(self.optimizer, factor=self.hp['LR_FACTOR'], patience=self.hp['LR_STEP_SIZE'])
            
        # experience Replay
        self.memory = ReplayBuffer(max_capacity=self.hp['BUFFER_SIZE'], batch_size=self.hp['BATCH_SIZE'])


    def convert_state(self, state: tuple) -> torch.Tensor:
        """
        Convert the environment state (tuple) into a flat tensor as the input for model.

        Parameters
        ----------
            state: tuple
                Tuple containing the state of the environment.

        Returns
        -------
            torch.float32
                Flat tensor of the state.          
        """
        # robot cumulative execution time
        r = np.sum(state[0]*state[2])
        # operator cumulative execution time
        t = np.sum(state[3]*state[5])

        s0 = torch.tensor(state[0], dtype=torch.float32)            # robot done
        s1 = torch.tensor([state[1]], dtype=torch.float32)          # robot scheduled
        s2 = torch.tensor(state[2], dtype=torch.float32)            # robot execution time
        s3 = torch.tensor(state[3], dtype=torch.float32)            # operator done
        s4 = torch.tensor([state[4]], dtype=torch.float32)          # operator scheduled
        #s5 = torch.tensor([max(r,t)-min(r,t)], dtype=torch.float32) # elapsed time of the task still in progress
        
        #return torch.cat((s0,s1,s2,s3,s4,s5)).to(self.device)
        return torch.cat((s0,s1,s2,s3,s4)).to(self.device)

    
    def act(self, state: torch.Tensor, mask: torch.Tensor, epsilon: float) -> torch.Tensor:
        """
        Select a valid (non-masked) action according to an epsilon greedy policy.

        Parameters
        ----------
            state: torch.Tensor
                State of the environment.
            mask: torch.Tensor
                Mask of valid actions from the given state.
            epsilon: float
                For epsilon greedy policy; if 0, act greedily.

        Returns
        -------
            torch.float32
                Flat tansor of the state             
        """  
        # select a random action
        if random.random() < epsilon:
            valid_actions = torch.where(mask == 1)[0]  # available actions indeces
            action = torch.tensor(random.choice(valid_actions.tolist()), dtype=torch.int64) # tensor(a)            
        # select the greedy action
        else:
            with torch.no_grad():
                # unsqueeze to add the batch dimension as input for the net
                # we get tensor([[q1], [q2], [q3], [q4]]), so squeeze() to get tensor([q1, q2, q3, q4])
                # argmax gives tensor(a)
                self.policy_net.eval()
                q_values = self.policy_net(state.unsqueeze(dim=0)).squeeze().cpu()
                large = -1000.
                action = (q_values+large*(1-mask)+large*(1-mask)).argmax()
                self.policy_net.train()
        return action
    

    def learn(self, batch: tuple) -> float:
        """
        
        """
        # separate the single elements of the transitions
        states, actions, rewards, next_states, dones, next_masks = zip(*batch)
        # stack tensors to create batch-like tensors
        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        next_states = torch.stack(next_states)
        dones = torch.stack(dones)
        next_masks = torch.stack(next_masks)
        
        # compute target Q values
        with torch.no_grad():
            large = -1000.
            # Double DQN
            if self.hp['DOUBLE']:
                # masking and action selection
                argmax_a = (self.policy_net(next_states) + large*(1-next_masks) + large*(1-next_masks)).argmax(dim=1)
                # action evaluation
                max_next_q_value = self.target_net(next_states).gather(1, index=argmax_a.unsqueeze(dim=1)).squeeze()
            # DQN
            else:
                # masking and maxing
                max_next_q_value = (self.target_net(next_states) + large*(1-next_masks) + large*(1-next_masks)).max(dim=1)[0]
            # Q target
            target_q = rewards + (1-dones)*self.hp['GAMMA'] * max_next_q_value   # shape [batch size]        
        # compute current Q values
        current_q = self.policy_net(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze() # shape [batch size]

        # loss and optimization
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()  # clear gradients
        loss.backward()             # compute gradients
        self.optimizer.step()       # update policy network
        return loss.item()

    
    def train(self) -> pd.DataFrame:
        """
        
        """

        # (mu, dev std) operator data for sampling execution times
        operator = pd.read_csv("operators_data.csv")
        # list of means for sampling execution times
        mu = operator["mu_"+str(self.hp['ID_OPERATOR'])].to_list()
        # dev std for sampling execution times
        sigma = operator["sigma_"+str(self.hp['ID_OPERATOR'])][0] # just one float needed

        print("mu operator "+str(self.hp['ID_OPERATOR'])+":",mu)
        print("sigma operator "+str(self.hp['ID_OPERATOR'])+":",sigma)

        # instance of the environment
        env = CobotEnv(n_operators=1,
                       mu_operators=(mu,),
                       std=sigma)

        # list to keep track of return, loss, epsilon, lr and model saving collected from every episode
        scores, losses, epsilons, lrs = [], [], [], []

        # to track best return for model saving
        score_window = deque(maxlen=10)
        best_score = -np.inf
        update_count, epsilon_count = 0, 0
        
        # initialize epsilon
        epsilon = self.hp['EPSILON_START']
        
        for episode in tqdm(range(self.hp['N_EPISODES'])):
            # even episode: expert operator
            #if episode%2 == 0:
            #    state = env.reset(1)
            # odd episode: slow operator
            #else:
            #    state = env.reset(0)
            state = env.reset(0)
            state = self.convert_state(state) # convert from env state to NN input state

            episode_score, episode_loss, loss_count = 0.0, 0.0, 0
            done = False

            # until episode is not over
            while not done:
                # get valid actions
                mask = env.get_valid_actions()
                # select best action, according to epsilon greedy policy and valid actions
                action = self.act(state, mask, epsilon).to(self.device)
                # execute action
                next_state, reward, done = env.step(action.item() + 1) # +1 makes the action range between 1 and ACTION_SIZE
                # cumulate reward
                episode_score += reward
                # get next state mask
                next_mask = env.get_valid_actions().to(self.device)
                # convert next state, reward and done to tensors
                next_state = self.convert_state(next_state) # convert from env state to dqn input state
                reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
                done = torch.tensor(done, dtype=torch.float32).to(self.device)
                # store transition into replay buffer
                self.memory.push((state, action, reward, next_state, done, next_mask))

                state = next_state
                
                # if enough samples have been collected into the replay buffer
                if self.memory.size() >= self.hp['BATCH_SIZE']:
                    # update counters
                    loss_count += 1
                    update_count += 1
                    
                    # sample batch
                    batch = self.memory.sample()
                    # train
                    loss = self.learn(batch)
                    episode_loss += loss

                    # update target network
                    if self.hp['HARD_UPDATE_EVERY'] is None:
                        soft_update(self.policy_net, self.target_net, self.hp['TAU'])
                    else:
                        if update_count > self.hp['HARD_UPDATE_EVERY']:
                            update_count = 0
                            hard_update(self.policy_net, self.target_net)

                    # Decay epsilon after every update
                    # exponential decay
                    #epsilon = self.hp['EPSILON_END'] + (self.hp['EPSILON_START'] - self.hp['EPSILON_END']) * math.exp(-1. * episode / self.hp['EPSILON_DECAY'])
                    # dynamic decay
                    epsilon = self.hp['EPSILON_END']+(self.hp['EPSILON_START']-self.hp['EPSILON_END'])*max((self.hp['EPS_DECAY_EPISODE']-max(0,epsilon_count))/self.hp['EPS_DECAY_EPISODE'], 0)
            
            # if at least one backprop swept has been performed
            if loss_count > 0:
                epsilon_count += 1

                score_window.append(episode_score)
                # save best model
                if np.mean(score_window) > best_score:
                    # update best score
                    best_score = np.mean(score_window)
                    # save model
                    torch.save({"best_score": best_score, "episode": episode, "model_state_dict": self.policy_net.state_dict()}, self.hp['LOG_PATH']+"op_"+str(self.hp['ID_OPERATOR'])+"_checkpoint.pt")

            # training logs
            scores.append(episode_score)
            if loss_count == 0:
                losses.append(episode_loss)
            else:
                losses.append(episode_loss/loss_count)
            epsilons.append(epsilon)
            # update lr after every episode (only if at least one update step has been done)
            if (self.hp['LR_STEP_SIZE'] is not None or self.hp['LR_FACTOR'] is not None) and loss_count>0:
                lrs.append(self.scheduler.get_last_lr()[0])
                self.scheduler.step(episode_loss/loss_count)
            else:
                lrs.append(self.hp['LR'])       

        # save training logs
        data = {
            "Episode": list(range(1, len(scores) + 1)),
            "Score": scores,
            "Loss": losses,
            "Epsilon": epsilons,
            "Learning rate": lrs
        }
        return pd.DataFrame(data)
    

    def test(self, n_runs: int = 200):
        # load checkpoint
        checkpoint = torch.load(self.hp['MODEL_PATH'], weights_only=False)
        print('\rModel saved - Episode {} - Score (SMA): {:.2f}'.format(checkpoint['episode'], checkpoint['best_score']))
        
        # load model weights
        self.policy_net.load_state_dict(checkpoint['model_state_dict'])
        self.policy_net.eval()

        env = CobotEnv()
        scores = []
        combos = {}

        for _ in range(n_runs):
            state = env.reset(0)
            episode_score = 0.0
            done = False
            while not done:
                # get valid actions
                mask = env.get_valid_actions()
                # act greedily
                action = self.act(self.convert_state(state), mask, epsilon=0).to(self.device)
                self.policy_net.eval()
                # execute action
                state, reward, done = env.step(action.item() + 1) # +1 makes the action range between 1 and ACTION_SIZE
                # cumulate reward
                episode_score += reward
            scores.append(episode_score)

            combo = np.array([0,0,0,0,0,0])
            combo[np.where(np.isin(np.array([7,8,9,12,13,14]), env.operator_done*env.operator_task_id))[0]] = 1
            if str(combo) in combos.keys():
                combos[str(combo)] += 1
            else:
                combos[str(combo)] = 1

        with open('slow_delta_DQN.txt', 'a') as f:
            print(combos, file=f)
            
        #print("Mean Score:",np.mean(scores))
        #print("Std Score:",np.std(scores))
