import random
import torch

class Agent():
    def __init__(self, model, num_actions, epsilon):
        self.model = model
        self.num_actions = num_actions
        self.epsilon = epsilon
    
    def select_action(self, state):
        if random.random() < self.epsilon: # random numer 0,3 for random action
            return random.randint(0, self.num_actions -1)

        with torch.no_grad(): # stops pytorch building gradient graph
            q_values = self.model(state) # (1,4)
        return  q_values.argmax(dim =1).item() # exploit: best action as python int

        