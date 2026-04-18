# Q-Learning Tic Tac Toe (Self-Play) — Notes

Two tabular Q-learners trained against each other. No neural network — just two Python dictionaries, each mapping `(state, action)` pairs to estimated values. Project 1 trained one agent against a uniformly-random opponent; this project replaces the random opponent with a second learner and trains them simultaneously.

## What the project contains

| File | Role |
|------|------|
| `game.py` | The environment — a 3×3 board with move / win / draw logic (same as project 1) |
| `agent.py` | The learner — Q-table, ε-greedy action selection, and the Bellman update |
| `train.py` | The self-play loop — two agents alternate moves and each runs its own delayed update |

## Why self-play matters

A random opponent is a poor sparring partner. Its moves cover positions no serious player would ever reach, and it never punishes a bad agent move with a convincing follow-up. That biases the Q-table toward "looks fine against noise" rather than "actually good."

Self-play fixes this two ways:

1. **Adversarial coverage.** Each agent's exploitation is the other agent's training signal. Positions agent A reaches through half-trained play are exactly the positions agent B needs to learn from — and vice versa. The distribution of (state, action) pairs the Q-tables see drifts toward positions a competent player would actually reach.
2. **Zero-sum learning signal.** A win for one agent is a loss for the other. Every terminal reward updates two Q-values, with opposite signs. No effort is wasted generating adversarial examples — the opponent's own learning produces them.

The failure mode to watch for is **non-stationarity**: agent A is optimising against a moving target (agent B, which keeps changing). In practice for tic tac toe this is fine — the state space is small and both agents converge to approximately optimal play within 20k episodes — but it's the reason self-play in larger games often needs tricks like frozen opponent snapshots or league training.

## The agent — `agent.py`

Unchanged in spirit from project 1. Q-table is a `defaultdict(float)` keyed by `(state, action)`; action selection is ε-greedy; the update rule is the same Bellman equation:

$$
Q(s, a) \;\leftarrow\; Q(s, a) + \alpha \bigl[\, r + \gamma \cdot \max_{a'} Q(s', a') - Q(s, a) \,\bigr]
$$

The only semantic change is that **state no longer includes whose turn it is**. The token is carried on the agent instead (`self.token`), so agent 1's Q-table only ever contains states where it's X's turn and agent 2's only ever contains O-to-move states. The two tables never share keys even though they look structurally identical.

### Terminal handling

On a terminal transition the future term is forced to 0, since there is no next move to evaluate. If the agent's move ended the game it gets the terminal reward directly; if the opponent's move ended the game, the agent's *last* move — the one buffered from the previous turn — gets updated with the opposing terminal reward.

## Training — `train.py`

### Two buffers, one per agent

Project 1 had a single buffer because only one agent was learning. Here both agents learn, so each needs its own `(state, action)` buffer. A single shared buffer would conflate the two learners' last moves and produce corrupt updates.

The loop, condensed:

```
while not terminal:
    # player moves
    state  = get_state(board)
    move   = player.choose_move(board)
    board' = apply(board, move, player.token)

    if terminal:
        player.update(state, move, terminal_reward, board', done=True)
        if player2_buffer is not None:
            player2.update(*player2_buffer, -terminal_reward, board', done=True)
        break

    # non-terminal: pay forward player2's pending move now that the board has advanced
    if player2_buffer is not None:
        player2.update(*player2_buffer, 0, board', done=False)

    player_buffer = (state, move)

    # ... symmetric block for player2's turn ...
```

The two key lines are:

- **`player2.update(*player2_buffer, -terminal_reward, ...)`** — when one agent wins, the other agent's last move is retroactively marked as a losing move. This is the zero-sum part.
- **`player2.update(*player2_buffer, 0, board', done=False)`** — on non-terminal transitions, the opponent's buffered move is updated *after* the current agent's move advances the board. That advanced board is what goes in as `next_board` for the Bellman update, which is exactly what off-policy Q-learning needs.

### Alternating who opens

The outer loop swaps `player` / `player2` on alternating episodes:

```python
if episode % 2 == 0:
    player, player2 = agent1, agent2
else:
    player, player2 = agent2, agent1
```

Without this, one agent would always see "empty board, my turn" states and the other would always see "one opposing piece on the board, my turn" states. Swapping the opener guarantees both Q-tables get full coverage of both opening roles.

### Epsilon decay (shared schedule)

Both agents share the same ε schedule — linear decay from 1.0 down to 0.05 over 80% of training, then clamped. Linking the schedules keeps neither agent permanently weaker than the other, which would distort the self-play dynamic (a permanently random agent would just mimic project 1's random-opponent setup).

```python
decayed = max(
    EPSILON_END,
    EPSILON_START - (EPSILON_START - EPSILON_END) * episode / DECAY_EPISODES,
)
agent1.epsilon = decayed
agent2.epsilon = decayed
```

### Reward shape

Same as project 1 — all signal flows at terminal transitions, zero reward at every mid-game step. The difference is that one game now produces **two** terminal rewards (one per agent) with opposite signs.

| Outcome for player | Reward to player | Reward to opponent |
|--------------------|------------------|--------------------|
| Win                | +1               | −1                 |
| Loss               | −1               | +1                 |
| Draw               |  0               |  0                 |

## Observed behaviour

Running 20,000 episodes:

- **Early** (ε near 1.0): near-50/50 win/loss split, draw rate under 15%. Both agents playing essentially random.
- **Mid** (ε decaying): draw rate climbs as both agents learn to block three-in-a-row threats.
- **Late** (ε = 0.05): draw rate dominates, ~75% of games end in a draw. Wins and losses stay balanced (because the agents are roughly equal in strength) but shrink in absolute count.

A high draw rate at convergence is the signature of self-play working. Tic tac toe is a theoretical draw with perfect play — if both agents converge toward optimal play, wins become accidents caused by residual ε-exploration.

### Q-table size

After 20k episodes each agent has roughly 4,700 unique `(state, action)` entries. Project 1's single agent only saw ~1,100; the larger tables here reflect how much broader the state coverage becomes under adversarial exploration.

## Limitations and next steps

- **Exploitation asymmetry.** If one agent drifts ahead in learning quality, the other may be reduced to playing bad moves and getting punished, producing a collapse where the stronger agent trains mainly on "wins against a weak opponent." With equal ε schedules and no model capacity issues this tends not to happen in tic tac toe, but it is the root of most self-play instability in larger games.
- **Evaluation gap.** Win-rate during training mixes two decaying exploration rates. The honest metric is ε=0 play against an external reference (e.g. a minimax player) — expect 100% draws against minimax if self-play converged.
- **Symmetry.** Tic tac toe has 8-way board symmetry (4 rotations × mirror). The Q-table stores all 8 copies separately, wasting roughly 8× the memory. Collapsing equivalent states would be a natural extension.
- **Next environment.** With a larger board (connect four, 5-in-a-row) the state space breaks the dict. That's the setup for function approximation / deep Q-learning.
