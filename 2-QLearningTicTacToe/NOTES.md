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

The failure mode to watch for is **non-stationarity**: agent A is optimising against a moving target (agent B, which keeps changing). In practice for tic tac toe this is fine — the state space is small and both agents converge to approximately optimal play within 30k episodes — but it's the reason self-play in larger games often needs tricks like frozen opponent snapshots or league training.

## The agent — `agent.py`

Unchanged in spirit from project 1. Q-table is a `defaultdict(float)` keyed by `(state, action)`; action selection is ε-greedy; the update rule is the same Bellman equation:

$$
Q(s, a) \;\leftarrow\; Q(s, a) + \alpha \bigl[\, r + \gamma \cdot \max_{a'} Q(s', a') - Q(s, a) \,\bigr]
$$

The only semantic change is that **state no longer includes whose turn it is**. The token is carried on the agent instead (`self.token`), so agent 1's Q-table only ever contains states where it's X's turn and agent 2's only ever contains O-to-move states. The two tables never share keys even though they look structurally identical.

### Terminal handling

On a terminal transition the future term is forced to 0, since there is no next move to evaluate. If the agent's move ended the game it gets the terminal reward directly; if the opponent's move ended the game, the agent's *last* move — the one buffered from the previous turn — gets updated with the opposing terminal reward.

## Hyperparameters — `alpha`, `gamma`, and optimistic init

### `alpha = 1.0` — deterministic environment

Q-learning usually uses a small α because the target `r + γ · max Q(s', a')` is computed from a noisy sample — random rewards, random transitions — and you need to average many samples to recover the true expected target. Small α is how you convert noisy samples into a stable estimate.

Tic tac toe's rules are **deterministic**: the same `(state, action)` always produces the same reward and the same board update. No noise to average out. So α can be pushed all the way up — every update is "believe what you just saw." The only residual variance comes from the opponent's ε-greedy moves (the next state the agent sees depends on their choice), which is why α=0.5 is defensible too. α=1 is the aggressive end, trading a little extra opponent-variance noise for faster convergence.

- Deterministic: tic-tac-toe, chess, grid-worlds without slippery tiles, most board games.                 
- Stochastic: poker (random cards), backgammon (dice), robotics (sensor/motor noise), Atari games (frame   
randomization), stock trading, anything physics-based with noise.   

### `gamma = 1.0` — finite horizon, terminal-only rewards

γ = 1 means no discount — a reward 9 moves away counts just as much as a reward right now. In most RL      
  settings that's dangerous. In tic-tac-toe, it's safe. Here's why.                                          
                                                                                                             
  What γ actually prevents                                                                                   
                                                                                                             
  The γ < 1 rule exists to solve three problems:                                                             
                                                                                                             
  1. Infinite horizons blow up                                                                               
   
  Imagine a game that never ends (e.g., a robot walking forever). Each step earns small reward. Without      
  discount:                                                                                                  
  V = r₁ + r₂ + r₃ + ...   →   ∞                                                                             
  Undefined value, no optimization target. γ < 1 collapses the infinite sum to a finite number: r₁ + γr₂ + 
  γ²r₃ + ... → bounded geometric series.                                                                     
                                                                                                             
  TTT doesn't have this problem. Every game terminates within 9 moves. Total return is bounded by the single 
  ±1 terminal reward. Can't diverge.                                                                         
                                                            
  2. Encouraging speed                                                                                       
                                                            
  In many domains you want "win fast, not eventually." γ<1 creates this pressure naturally — a win at step 3 
  is worth γ³, at step 9 is worth γ⁹. Fewer steps = more value.
                                                                                                             
  Do you care about winning fast in TTT? Honestly, no. A win in 9 moves and a win in 3 moves are both wins.  
  The game has no clock. So losing the "prefer faster wins" pressure is fine here.
                                                                                                             
  3. Uncertainty about the distant future                                                                    
   
  In stochastic/long environments, value estimates far in the future are unreliable. γ<1 says "I trust what I
   see in 5 steps more than what I think will happen in 50." It's a built-in humility term.
                                                                                                             
  TTT is deterministic and 9 steps max. Nothing is "far" here — every game is a short, well-defined sequence.
   No distant-future distrust needed.

### Optimistic initialization — `defaultdict(lambda: 1.0)`

Unseen `(state, action)` pairs default to **1.0** instead of 0.0. Since 1.0 is the maximum possible terminal reward, every unvisited action *looks like a certain win* until the agent actually tries it and collects the real value. Consequence: the agent's greedy policy naturally prefers unseen moves over moves that have been tried and returned a mediocre result. Exploration is baked into the init — ε-greedy is still there to handle ties and late refinement, but the initial coverage pressure comes from the optimism itself, which is why `epsilon_start` can sit at 0.2 instead of the 1.0 a cold-start setup would need.

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

Both agents share the same ε schedule — linear decay from 0.2 down to 0.05 over 80% of training, then stuch. Linking the schedules keeps neither agent permanently weaker than the other, which would distort the self-play dynamic (a permanently random agent would just mimic project 1's random-opponent setup).

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

Running 30,000 episodes:

- **Early** (ε near 0.2): the optimistic init is still driving most exploration — agents greedily try whichever `(state, action)` pairs they haven't yet updated down from 1.0. Draw rate is low; win/loss is roughly balanced.
- **Mid** (ε decaying): draw rate climbs as both agents learn to block three-in-a-row threats.
- **Late** (ε = 0.05): draw rate dominates, ~75% of games end in a draw. Wins and losses stay balanced (because the agents are roughly equal in strength) but shrink in absolute count.

A high draw rate at convergence is the signature of self-play working. Tic tac toe is a theoretical draw with perfect play — if both agents converge toward optimal play, wins become accidents caused by residual ε-exploration.

### Q-table size
This board model without knoledge of rotations and symmetry can find a total of 5,748 different states but we dont store terminal states (states where the game has ended) so actually our model can only find 4,520 non terminal states.

After 30k episodes the combined Q-table holds **28,113** entries — about 14,056 per agent. Against ~2,260 states per agent that averages out to ~6 `(state, action)` entries per state, which is reasonable given that early-game states (few pieces, many empty cells) have up to 9 legal actions each. Coverage isn't literally exhaustive, but it's close to the reachable game tree from each agent's side.

Project 1's single agent only saw ~1,100 entries total; adversarial self-play drives the tables an order of magnitude wider because positions that a random opponent would almost never produce are exactly the positions a half-trained learner reaches on the way to competent play.

## Limitations and next steps

- **Exploitation asymmetry.** If one agent drifts ahead in learning quality, the other may be reduced to playing bad moves and getting punished, producing a collapse where the stronger agent trains mainly on "wins against a weak opponent." With equal ε schedules and no model capacity issues this tends not to happen in tic tac toe, but it is the root of most self-play instability in larger games.
- **Evaluation gap.** Win-rate during training mixes two decaying exploration rates. The honest metric is ε=0 play against an external reference (e.g. a minimax player) — expect 100% draws against minimax if self-play converged.
- **Symmetry.** Tic tac toe has 8-way board symmetry (4 rotations × mirror). The Q-table stores all 8 copies separately, wasting roughly 8× the memory. Collapsing equivalent states would be a natural extension.
- **Next environment.** With a larger board (connect four, 5-in-a-row) the state space breaks the dict. That's the setup for function approximation / deep Q-learning.
