import random
import numpy as np
import torch
from collections import deque

class ReplayBuffer:
    #def __init__(self, max_capacity: int, batch_size: int, device: torch.device|str):
    def __init__(self, max_capacity: int, batch_size: int):
        """
        Replay Buffer memory to save the state, action, reward sequence from the current episode.
        
        Parameters
        ----------
            max_capacity: int
                Max buffer capacity.
            batch_size: int
                Size of the batch.
        """
        self.capacity = max_capacity
        self.batch_size = batch_size
        #self.device = device
        
        # deque for storing transitions
        self.buffer = deque(maxlen=self.capacity)


#    def push(self, state, action, reward, next_state, done) -> None:
#        """
#        Store a transition into the buffer.
#        
#        Parameters
#        ----------
#            state: np.array
#                Current state.
#            action: int
#                Action.
#            reward: float
#                Reward.
#            next_state: np.array
#                Next state.
#            done: bool
#                Terminal flag.
#        """
#        assert isinstance(state, torch.float32), "Invalid state!"
#        assert isinstance(action, int), "Invalid action!"
#        assert isinstance(reward, float), "Invalid reward!"
#        assert isinstance(next_state, torch.float32), "Invalid next state!"
#        assert isinstance(done, bool), "Invalid terminal flag!"

#        self.buffer.append((state, action, reward, next_state, done))


#    def sample(self) -> tuple:
#        """
#        Sample a batch of transitions from the buffer.

#        Returns
#        -------
#            tuple
#                Batch of transitions.
#        """
#        assert self.size() >= self.batch_size, "Replay Buffer not big enough for sampling!"
        
#        batch = random.sample(self.buffer, self.batch_size)
#        states, actions, rewards, next_states, dones = zip(*batch)
        
 #       return torch.stack(states), \
  #          torch.stack(actions).unsqueeze(-1), \
   #         torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(-1), \
    #        torch.stack(next_states), \
     #       torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(-1)
    
    
    def push(self, transition: tuple) -> None:
        """
        Store a transition into the buffer.
        
        Parameters
        ----------
            transition: tuple
                Transition.
        """
        self.buffer.append(transition)

    
    def sample(self) -> tuple:
        """
        Sample a batch of transitions from the buffer.

        Returns
        -------
            tuple
                Batch of transitions.
        """
        return random.sample(self.buffer, self.batch_size)

    
    def size(self) -> int:
        """
        Current number of elements in the buffer.
        
        Returns
        -------
            int
                Number of stored transitions.
        """
        return len(self.buffer)