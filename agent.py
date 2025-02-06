import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import numpy as np
from tqdm import tqdm
import pandas as pd
import random
import math

# my packages
from models.dqn import DQN
from env.env_v2 import CobotEnv
from utils.functions import soft_update, hard_update, plot
from memory.replay import ReplayBuffer


class Agent:
    def __init__(self, hyperparameters: dict):
        self.hp = hyperparameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device used: ",self.device)

        # policy network
        self.policy_net = DQN(self.hp['STATE_SIZE'], self.hp['HIDDEN_LAYERS'], self.hp['ACTION_SIZE']).to(self.device)
        # target network
        self.target_net = DQN(self.hp['STATE_SIZE'], self.hp['HIDDEN_LAYERS'], self.hp['ACTION_SIZE']).to(self.device)
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
####            self.scheduler = StepLR(self.optimizer, step_size=self.hp['LR_STEP_SIZE'], gamma=self.hp['LR_FACTOR'])
            # REDUCE LR ON PLATEAU
            self.scheduler = ReduceLROnPlateau(self.optimizer, factor=self.hp['LR_FACTOR'], patience=self.hp['LR_STEP_SIZE'])
            
        # experience Replay
        self.memory = ReplayBuffer(max_capacity=self.hp['BUFFER_SIZE'], batch_size=self.hp['BATCH_SIZE'])


    def state_to_flat_tensor(self, state: tuple) -> torch.Tensor:
        """
        Convert a state (tuple) into a flat tensor.

        Parameters
        ----------
            state: tuple
                Tuple containing the state of the environment to convert into a tensor.

        Returns
        -------
            torch.float32
                Flat tansor of the state.          
        """
        s0 = torch.tensor(state[0], dtype=torch.float32)
        s1 = torch.tensor([state[1]], dtype=torch.float32)
        s2 = torch.tensor(state[2], dtype=torch.float32)
        s3 = torch.tensor(state[3], dtype=torch.float32)
        s4 = torch.tensor([state[4]], dtype=torch.float32)
        s5 = torch.tensor(state[5], dtype=torch.float32)
        return torch.cat((s0,s1,s2,s3,s4,s5))

    
    def act(self, state: torch.Tensor, mask: torch.Tensor, epsilon: float = None) -> torch.Tensor:
        """
        Select a valid (non-masked) action according to an epsilon greedy policy.

        Parameters
        ----------
            state: torch.Tensor
                State of the environment.
            mask: torch.Tensor
                Mask of valid actions from the given state.
            epsilon: float
                For epsilon greedy policy, if None -> act deterministically.

        Returns
        -------
            torch.float32
                Flat tansor of the state             
        """  
        # select a random action
        if epsilon is not None and random.random() < epsilon:
            valid_actions = torch.where(mask == 1)[0]  # available actions indeces
            action = torch.tensor(random.choice(valid_actions.tolist()), dtype=torch.int64) # tensor(a)            
        # select the greedy action
        else:
            with torch.no_grad():
                # unsqueeze to add the batch dimension as input for the net
                # we get tensor([[q1], [q2], [q3], [q4]]), so squeeze() to get tensor([q1, q2, q3, q4])
                # argmax gives tensor(a)
                self.policy_net.eval()
                q_values = self.policy_net(state.unsqueeze(dim=0).to(self.device)).squeeze().cpu() # move to cpu to free space
                large = -1000.
                action = (q_values+large*(1-mask)+large*(1-mask)).argmax()
                self.policy_net.train()
        return action
    

    def learn(self, batch: tuple) -> float:
        # separate the single elements of the transitions
        states, actions, rewards, next_states, dones, next_masks = zip(*batch)
        # stack tensors to create batch-like tensors
        states = torch.stack(states).to(self.device)
        actions = torch.stack(actions).to(self.device)
        rewards = torch.stack(rewards).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.stack(dones).to(self.device)
        next_masks = torch.stack(next_masks).to(self.device)
        
        # compute target Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            large = -1000.
            # masking and maxing
            max_next_q_value = (next_q_values + large*(1-next_masks) + large*(1-next_masks)).max(dim=1)[0]
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

    
    def train(self, n_episodes: int):

        # instance of the environment
        env = CobotEnv()

        # list to keep track of return, loss, epsilon, lr and model saving collected from every episode
        scores, losses, epsilons, lrs, saves = [], [], [], [], []
        # to track best return for model selection
        best_episode_score = -np.inf
        update_count = 0
        # initialize epsilon
        epsilon = self.hp['EPSILON_START']
        
        for episode in tqdm(range(1, n_episodes+1), desc="# of episodes"):
        #for episode in range(1, n_episodes+1):
            #print("\rEpisode n°:", episode)
            state = env.reset()
            state = self.state_to_flat_tensor(state) # to convert the state tuple to a flat tensor

            episode_score, episode_loss, loss_count = 0.0, 0.0, 0
            done = False

            # until episode is not over
            while not done:
                # get valid actions
                mask = env.get_valid_actions()
                # select best action, according to epsilon greedy policy and valid actions
                action = self.act(state, mask, epsilon)
                # execute action
                next_state, reward, done = env.step(action.item() + 1) # +1 makes the action range between 1 and ACTION_SIZE
                # cumulate reward
                episode_score += reward
                # get next state mask
                next_mask = env.get_valid_actions()
                # convert next state, reward and done to tensors
                next_state = self.state_to_flat_tensor(next_state)
                reward = torch.tensor(reward, dtype=torch.float32)
                done = torch.tensor(done, dtype=torch.float32)
                # store transition into replay buffer
                self.memory.push((state, action, reward, next_state, done, next_mask))

                state = next_state
                
                # if enough samples have been collected into the replay buffer
                if self.memory.size() >= self.hp['BATCH_SIZE']:
                    batch = self.memory.sample()
                    loss = self.learn(batch)
                    
                    episode_loss += loss
                    loss_count += 1
                    update_count += 1

                    if self.hp['HARD_UPDATE_EVERY'] is None:
                        soft_update(self.policy_net, self.target_net, self.hp['TAU'])
                    else:
                        if update_count > self.hp['HARD_UPDATE_EVERY']:
                            update_count = 0
                            hard_update(self.policy_net, self.target_net)
                        
                    # Decay epsilon after every update
                    if self.hp['EPSILON_DECAY'] is None:
                        #linear decay
                        epsilon = self.hp['EPSILON_START'] - (self.hp['EPSILON_START'] - self.hp['EPSILON_END']) * (episode / n_episodes)
                    else:
                        # exponential decay
                        epsilon = self.hp['EPSILON_END'] + (self.hp['EPSILON_START'] - self.hp['EPSILON_END']) * math.exp(-1. * episode / self.hp['EPSILON_DECAY'])
            
            # display the performance every 100 episodes
#            if episode % 100 == 0:
#                if loss_count == 0:
#                    print('\rEpisode {} | Score: {:.2f} | Loss: {:.2f} | LR: {:.5f} | Eps: {:.2f}'\
#                          .format(episode, episode_score, episode_loss, self.scheduler.get_last_lr()[0], epsilon))
#                else:
#                    print('\rEpisode {} | Score: {:.2f} | Loss: {:.2f} | LR: {:.5f} | Eps: {:.2f}'\
#                          .format(episode, episode_score, episode_loss/loss_count, self.scheduler.get_last_lr()[0], epsilon))
            
            # save model if episode
            if episode_score != 0.0 and episode_score > best_episode_score:
                best_episode_score = episode_score
                saves.append("Saved")
                #print('\rSaving model - Episode {} - Score: {:.2f}'.format(episode, episode_score))
                torch.save(self.policy_net.state_dict(), self.hp['BEST_MODEL_PATH'])
            else:
                saves.append("")

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
            "Learning rate": lrs,
            "Model saving": saves
        }
        df = pd.DataFrame(data)
        df.to_csv(self.hp['LOG_PATH'], index=False)
        plot(df, self.hp['PLOT_PATH'])
        print(f"\rBest episode score: {best_episode_score}.") 
        print(f"Dati salvati in {self.hp['LOG_PATH']} e plottati in {self.hp['PLOT_PATH']}.") 
                

    def evaluate(self, model, env, n_eval_episodes):
        """
        Qui dentro eseguire n_eval_episodes episodi e restituire mean return e std return
        """
        #prendere path del modello salvato, load del modello, creazione env, esecuzione degli episodi e calcolo metriche