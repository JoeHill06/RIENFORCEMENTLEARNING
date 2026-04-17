# Q-Learning Tic Tac Toe — Notes

A tabular Q-learning agent that learns to play tic tac toe by self-bootstrapping against a uniformly-random opponent. No neural network, no PyTorch — just a Python dictionary mapping `(state, action)` pairs to estimated values.

## What the project contains

| File | Role |
|------|------|
| `game.py` | The environment — a 3×3 board with move / win / draw logic |
| `agent.py` | The learner — Q-table, action selection, and the Bellman update |
| `train.py` | The loop — plays episodes vs a random opponent and calls `agent.update` |

## The environment — `game.py`

The board is a 3×3 grid of integers where:

- `-1` means empty
- `0` means **naught** (O)
- `1` means **cross** (X)

Three methods work the game:

- `move(i, j, value)` — places `value` at `(i, j)`, validating bounds and emptiness. Returns `True` on success, `False` on an illegal move.
- `check_win()` — returns the winning token (`0` or `1`) or `-1` if no player has three in a row yet. Checks every row, column, and both diagonals.
- `check_draw()` — `True` only when **both** no player has won **and** every cell is filled. Must be combined with `check_win()` to determine the terminal state.

The environment is stateless beyond the board — it does not track whose turn it is. That responsibility lives in the training loop.

## The agent — `agent.py`

### Q-table

```python
self.q_table: dict[tuple[tuple[tuple[int, ...], ...], tuple[int, int]], float]
```

In plain English: a dictionary keyed by `(state, action)`, where

- **state** is the board frozen into a tuple of tuples (so it's hashable — lists can't be dict keys)
- **action** is `(row, col)`
- the value is the current estimate of `Q(state, action)` — the discounted return the agent expects from playing `action` in `state` and then acting greedily forever after

Unseen pairs default to `0.0` — **optimistic initialization**, nudging the agent to try actions it has never taken.

### Action selection — `find_best_move`

ε-greedy policy:

- With probability `ε` (`self.epsilon`, default `0.1`), pick a random legal move. This keeps exploration alive.
- Otherwise, evaluate `Q(state, a)` for every legal `a` and pick the argmax. Ties are broken by first-seen, making behavior deterministic on repeated states.

### The learning method — `update` (the heart of the algorithm)

This is the one method that actually modifies the Q-table. The rule applied is the standard off-policy **Q-learning** update, a special case of temporal-difference learning:

$$
Q(s, a) \;\leftarrow\; Q(s, a) + \alpha \bigl[\, r + \gamma \cdot \max_{a'} Q(s', a') - Q(s, a) \,\bigr]
$$

where

- `s` is the state the move was taken from
- `a` is the action taken
- `r` is the immediate reward received
- `s'` is the resulting state
- `α` (`self.alpha`, default `0.1`) is the **learning rate**: how aggressively the new estimate overrides the old one
- `γ` (`self.gamma`, default `0.9`) is the **discount factor**: how much the agent cares about future reward relative to immediate reward

The bracketed expression is the **TD-error** (temporal difference error) — the difference between the *target* (`r + γ · max Q(s', a')`) and the current estimate. Every call to `update` drags the old estimate toward the target by a fraction `α`.

#### How the code implements it

```python
# 1. Current estimate. Missing keys default to 0.0 - unseen pairs enter
#    the Q-table implicitly the first time they are updated.
old = self.q_table.get((state, action), 0.0)

# 2. `future` = best-case continuation value from s'. This is what makes
#    Q-learning OFF-POLICY: it assumes the agent will play optimally from
#    s' onward, regardless of what policy actually chose the move.
if done:
    future = 0.0                       # terminal: no more reward possible
else:
    next_moves = self.find_possible_moves(next_board)
    if not next_moves:
        future = 0.0                   # defensive: also terminal
    else:
        future = max(
            self.q_table.get((next_state, a), 0.0) for a in next_moves
        )

# 3. Bellman update: pull the old value toward (r + gamma * future) by alpha.
self.q_table[(state, action)] = old + self.alpha * (
    reward + self.gamma * future - old
)
```

#### Why `max` over `Q(s', a')`, not the action the agent actually picked?

Because Q-learning is **off-policy**. The target assumes perfect future play, not actual future play. This is why the algorithm can learn an optimal policy *even while exploring randomly* — the exploration noise doesn't contaminate the learned values.

#### Why `future = 0` on terminal?

Once the game ends, there are no more rewards. Adding any non-zero value would inject fictional future reward into the terminal Q-value, distorting every upstream estimate via backup.

#### Why does this converge?

For a tabular Q-learner in a finite MDP, classical results (Watkins 1989) guarantee convergence to the optimal Q-function provided (a) every `(s, a)` is visited infinitely often, and (b) `α` satisfies standard Robbins–Monro conditions. ε-greedy exploration with fixed `ε > 0` handles (a); a fixed small `α` violates (b) strictly but is the usual practical choice and converges in expectation to a neighborhood of the optimum.

### Reward shape

Rewards only flow at terminal transitions:

| Outcome | Reward to learner |
|---------|-------------------|
| Win | +1 |
| Loss | −1 |
| Draw | 0 |
| Mid-game move | 0 |

All learning signal propagates *backward* through the Q-table via repeated updates — the value of a move three turns before the end gets pulled toward the terminal reward one `γ`-discounted step at a time.

## Training — `train.py`

The training loop plays one episode at a time against a uniformly-random opponent.

### The delayed-update trick

A move's true consequence is not visible on the turn it is played — the opponent still gets to respond. Applying `update` immediately with reward 0 would hide the fact that the move just played might have *handed the opponent a win*.

The fix is to **buffer** the agent's most recent `(state, action)` and apply its update one turn later, once we can see the opponent's response:

```
agent moves  -> maybe terminal -> if yes, update with ±1 / 0
             -> else buffer (state, action)

opponent moves -> maybe terminal -> if yes, update the BUFFERED move with -1 / 0
               -> else update the BUFFERED move with reward 0 and next_board
```

The effect: every learner move eventually receives the correct reward for the state two plies later (its own move, then opponent's reply).

### Alternating first-player

Odd-numbered episodes the learner goes first; even-numbered episodes the opponent does. Without this, the agent would only ever see states consistent with first-player openings.

### Reporting window

Every 100 episodes, the loop prints rolling win/loss/draw percentages and the current Q-table size. The counters are zeroed after each report so each line reflects recent performance only.

## Observed behaviour

Running 10,000 episodes from scratch (seed 0):

- Win-rate climbs from ~76% in the first hundred episodes to ~87% by episode 600.
- After episode ~600 the curve plateaus — the bouncing between 83% and 90% is noise from ε-exploration, not continued learning.
- Q-table saturates near ~1,000 unique `(state, action)` entries. New entries slow from ~70 per 100 episodes early to ~5 per 100 late — state coverage has filled in.

### Why isn't win-rate 100%?

Because `ε = 0.1`: roughly 1 in 10 of the agent's moves is a coin flip, and a fraction of those coin-flips walk into losing positions. The true greedy policy is much stronger than the observed training win-rate suggests — evaluating with `epsilon = 0` shows this cleanly.

## Limitations and next steps

- **Training opponent is dumb.** A random opponent cannot produce adversarial states, so the agent may have soft spots against any competent opponent. Self-play (two Q-learners against each other) is the standard next step.
- **No epsilon decay.** A common improvement is linearly annealing `ε` from ~1.0 down to ~0.01 over training so early episodes explore aggressively and late episodes exploit.
- **Table scale.** Tabular Q-learning works here because tic tac toe has at most a few thousand reachable states. For anything larger (connect four, checkers), function approximation — i.e. deep Q-learning — replaces the dict with a neural network. Beyond that other methods are used policy gradients, actor-critic, AlphaZero-style approaches. 
