from collections import defaultdict
from typing import List, Tuple
import random

State = Tuple[Tuple[int, ...], ...]


class agent():
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1, token=0):
        # q_table key is (state, action) -> float; defaultdict returns 0.0 for unseen keys
        self.q_table = defaultdict(lambda: 1.0) # setting unseen states as 1 encourages experimentation
        self.alpha = alpha      # learning rate
        self.gamma = gamma      # discount rate
        self.epsilon = epsilon  # exploration rate
        self.token = token
        self.wins = 0
        self.loss = 0
        self.draw = 0

    # returns the board as a nested tuple so it can be used as a dict key
    def get_state(self, board: List[List[int]]) -> State:
        return tuple(tuple(row) for row in board)

    # list of empty cell coordinates
    def get_actions(self, board):
        empty = []
        for i in range(len(board)):
            for j in range(len(board[i])):
                if board[i][j] == -1:
                    empty.append((i, j))
        return empty

    # Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
    def update(self, state, action, reward, next_board, done=False):
        key = (state, action)

        # terminal: no future, so the target is just the reward
        if done:
            self.q_table[key] += self.alpha * (reward - self.q_table[key])
            return

        next_state = self.get_state(next_board)
        possible_actions = self.get_actions(next_board)
        if possible_actions:
            best_next = max(self.q_table[(next_state, a)] for a in possible_actions)
        else:
            best_next = 0.0

        self.q_table[key] += self.alpha * (reward + self.gamma * best_next - self.q_table[key])

    # epsilon-greedy move selection
    def choose_move(self, board):
        possible_actions = self.get_actions(board)

        if random.random() < self.epsilon:
            return random.choice(possible_actions)

        state = self.get_state(board)
        best_action = possible_actions[0]
        best_reward = self.q_table[(state, best_action)]
        for action in possible_actions:
            reward = self.q_table[(state, action)]
            if reward > best_reward:
                best_reward = reward
                best_action = action

        return best_action
