import random
import numpy as np
import torch
from collections import deque

class ReplayBuffer:
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
        # deque for storing transitions
        self.buffer = deque(maxlen=self.capacity)
    
    
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