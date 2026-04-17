import random

class agent():
    '''
    A Q learning agent that is able to interact with a tic tac toe board, find error, update values. 
    self.q_table is a dict keyed by (state, action) -> Q-value
    for state to be hashable the board must be a tuple of tuples or flat tuple
    Hyperparameters 
    self.alpha is the learning rate
    self.gamma is the discount 
    self.epsilon is the exploration rate
    self.token is the agents token naught or cross
    '''

    def __init__(self):
        '''Create a new agent with empty q state table'''
        self.q_table = {}
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.1
        self.token = None
    
    def update(self, board, action, reward, next_board, done=False):
        '''Apply the Q-learning Bellman update for a single transition.

        Update rule:
            Q(s, a) <- Q(s, a) + alpha * (reward + gamma * future - Q(s, a))
        where `future` is max over a' of Q(s', a') when the next state is
        non-terminal, and 0 when it is terminal (no future reward possible
        because the episode has ended).

        Parameters
        ----------
        board : list[list[int]]
            The state `s` in which the action was taken stored as a tuple(tuple(int)).
        action : tuple[int, int]
            The `(row, col)` move that was played from `board`.
        reward : float
            Immediate reward received for the transition (e.g. +1 for a win,
            -1 for a loss, 0 otherwise).
        next_board : list[list[int]]
            The resulting state `s'` after the move (and the opponent's
            response, if the caller treats a full turn as one transition).
        done : bool
            True if `next_board` is terminal (win, loss, or draw). When set,
            `future` is forced to 0 because no further rewards are possible.
        '''
        # Freeze the current board into a hashable key so it can index
        # `self.q_table`. Same rationale as in `find_best_move`.
        state = tuple(tuple(row) for row in board)

        # Existing estimate for this (state, action). Missing entries default
        # to 0.0 - this is where new states enter the table implicitly.
        old = self.q_table.get((state, action), 0.0)

        # `future` is the agent's best-case continuation value from s'.
        # On a terminal transition there is no next action, so it is 0.
        if done:
            future = 0.0
        else:
            next_state = tuple(tuple(row) for row in next_board)
            next_moves = self.find_possible_moves(next_board)
            if not next_moves:
                # No legal moves left - treat as terminal even if `done`
                # was not passed in. Protects against caller oversight.
                future = 0.0
            else:
                # Greedy lookahead: assume the agent will play optimally
                # from s' onward. This is what makes Q-learning off-policy.
                future = max(
                    self.q_table.get((next_state, a), 0.0) for a in next_moves
                )

        # The Bellman update itself - shift the old estimate toward the
        # newly observed target `(reward + gamma * future)` by a fraction
        # `alpha`. Smaller alpha = slower but more stable learning.
        self.q_table[(state, action)] = old + self.alpha * (
            reward + self.gamma * future - old
        )

    def find_possible_moves(self, board):
        '''Return a list of `(row, col)` coordinates for every empty cell.

        An empty cell is one that still holds the sentinel value -1. The
        returned list represents every legal move the agent could make on
        the given board; an empty list means the board is full.
        '''
        empty = []
        # `enumerate` gives us the row/column indices alongside the values.
        # We need the coordinates (not the cell value itself) because the
        # caller uses them as the action in the Q-table key.
        for i, row in enumerate(board):
            for j, cell in enumerate(row):
                if cell == -1:
                    empty.append((i, j))
        return empty

    def find_best_move(self, board):
        '''Select a move for the current board using an epsilon-greedy policy.

        With probability `self.epsilon`, explore by picking a random legal
        move. Otherwise, exploit by picking the move with the highest stored
        Q-value for the current state. Unseen `(state, action)` pairs are
        treated as Q = 0.0 (optimistic initialization), which nudges the
        agent toward trying actions it has never seen before.

        Parameters
        ----------
        board : list[list[int]]
            The current 3x3 board. Not mutated by this method.

        Returns
        -------
        tuple[int, int] | None
            The chosen `(row, col)`, or None if the board has no empty cells
            (the caller should treat this as a terminal state).
        '''
        # Every cell the agent is legally allowed to play into this turn.
        possible = self.find_possible_moves(board)
        if not possible:
            return None  # Board is full - nothing to play.

        # Exploration branch: roll a random number and, if it falls under
        # epsilon, ignore the Q-table and pick any legal move. This stops
        # the agent from locking onto early (and often wrong) Q estimates.
        if random.random() < self.epsilon:
            return random.choice(possible)

        # Q-table keys are (state, action). The state must be hashable, so
        # freeze the nested list of rows into a tuple of tuples.
        state = tuple(tuple(row) for row in board)

        # Exploitation: walk every legal move, look up its Q-value (default
        # 0.0 for unseen pairs), and keep the best. A manual loop is used
        # instead of `max(..., key=...)` so ties resolve to the first move
        # encountered - deterministic and easy to reason about.
        best_move = possible[0]
        best_value = self.q_table.get((state, best_move), 0.0)
        for move in possible[1:]:
            value = self.q_table.get((state, move), 0.0)
            if value > best_value:
                best_value = value
                best_move = move

        return best_move

    
