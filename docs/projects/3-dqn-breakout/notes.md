# Deep Q-Network — Atari Breakout — Notes

A convolutional neural network trained to play Atari Breakout from raw pixels. Projects 1 and 2 stored `Q(s, a)` in a Python dict — that only works when the state space is small and enumerable. Breakout has 210×160×3-byte frames and a continuous notion of "state" (ball velocity, paddle position, brick layout), so the dict approach breaks. DQN replaces the table with a neural network that *approximates* `Q(s, a)` for any state, including states never seen during training.

## What the project contains

| File | Role |
|------|------|
| `CNN.py` | The network — a tiny builder that turns a list of layer specs into a `torch.nn.Module` |
| `agent.py` | The action selector — ε-greedy over the network's Q-value output |
| `game.py` | The environment wrapper — grayscale/resize, 4-frame stack, FIRE-on-life-loss |
| `train.py` | The training loop — replay buffer, target network, Huber loss, Adam |
| `evaluate.py` | Load a checkpoint, play N games at low ε, report mean/best/worst/std |

## Why DQN (and not a Q-table)

A Q-table is a lookup: you hand it `(state, action)` and it hands back a number. That works for tic-tac-toe because the entire state space fits in ~5k entries. Breakout doesn't:

- A single 84×84 grayscale frame has 256<sup>84·84</sup> ≈ 10<sup>16,980</sup> possible values. Even if only a tiny fraction ever occurs in real play, the reachable state space is astronomically larger than anything a dict can enumerate.
- Two frames that differ by one pixel of paddle position are effectively the same state to a human — but to a dict they'd be separate keys with independent Q-values. No generalisation across visually similar states.

A neural network fixes both problems at once. Instead of memorising `Q(s, a)` per state, it learns a *function* `Q(s, a; θ)` parameterised by network weights θ. Nearby states produce nearby Q-values automatically, because convolutional filters share structure across the input — learning to recognise "ball moving left" on one frame transfers to every other frame where the ball is moving left.

The cost: the table is gone, so you can't just overwrite an entry. Every update to one `(s, a)` shifts the network's predictions for *every* other state too. That's why DQN needs two extra tricks — the replay buffer and the target network — that weren't needed in the tabular projects.

## The network — `CNN.py`

The "Nature DQN" architecture, the same one DeepMind used in the 2015 Atari paper:

```
Input:  (B, 4, 84, 84)          # 4 stacked grayscale frames
Conv2d(4,  32, kernel=8, stride=4) -> ReLU   # (B, 32, 20, 20)
Conv2d(32, 64, kernel=4, stride=2) -> ReLU   # (B, 64,  9,  9)
Conv2d(64, 64, kernel=3, stride=1) -> ReLU   # (B, 64,  7,  7)
Flatten                                      # (B, 3136)
Linear(3136, 512) -> ReLU                    # (B, 512)
Linear(512, 4)                               # (B, 4) -> Q-values per action
```

The output layer has **one neuron per action**, not per (state, action) pair. A single forward pass produces Q-values for every action simultaneously, so action selection is a single `argmax` over the last dimension. This is the whole reason DQN scales: the cost of "evaluating all actions for a given state" no longer depends on how many states you've seen.

`CNN.py` builds the layers from a list of tuples (`[("Conv2d", in, out, k, s), ("ReLU",), ...]`). It's verbose compared to writing a hard-coded module, but it means the architecture is a single data structure that both `train.py` and `evaluate.py` can reference, keeping them in sync.

## The environment wrapper — `game.py`

Raw Atari frames are 210×160×3 bytes at 60 Hz. Training directly on that is wasteful — colour rarely matters, resolution is overkill, and a single frame doesn't show which way the ball is moving. `Game` (a `gym.Wrapper`) folds the standard preprocessing into the step/reset API:

1. **Grayscale.** `obs.mean(dim=2)` collapses RGB to a single channel. Brick colours carry no gameplay meaning.
2. **Resize to 84×84.** Bilinear interpolation via `torch.nn.functional.interpolate`. Cuts input size ~6× without losing ball/paddle visibility.
3. **Scale to [0, 1].** Divide by 255 so inputs are in the same range ReLU + Adam expect.
4. **Stack 4 frames.** A single frame hides velocity — "ball at row 50" could be moving up or down. Stacking the last 4 frames in the channel dimension means the network sees ball *trajectory*, not just position.
5. **FIRE on life loss.** Breakout only auto-launches the ball at game start. When the agent loses a life the environment sits idle until it presses FIRE again; the wrapper inserts that FIRE automatically so the agent doesn't have to learn "wait, press FIRE after losing a life" (which is a dense sparse-reward trap).

`reset()` primes the 4-frame deque with copies of the opening frame so the first `stacked_state()` call has a valid tensor shape before any transitions have happened.

## The agent — `agent.py`

Agent is a 15-line wrapper around the network. `select_action(state)` flips a biased coin: with probability ε, return a random action; otherwise forward the state, `argmax` over the Q-output, return that. Identical in spirit to ε-greedy in projects 1 and 2 — the only difference is that "look up Q-values" now means "run a forward pass" instead of "index a dict."

`with torch.no_grad():` wraps the forward pass so no gradient graph is built during action selection. Gradients only come from the training step, and building a graph we'll immediately throw away is pure overhead.

## The two DQN-specific tricks — replay buffer + target network

Tabular Q-learning updates one entry at a time, and that entry has no effect on any other entry. Neural Q-learning doesn't have that luxury — every weight update shifts the predictions everywhere. Naively applying the Bellman update in-place causes the network to diverge. Two standard fixes:

### Replay buffer — breaking temporal correlation

`ReplayBuffer` is a `deque(maxlen=100_000)` of `(state, action, reward, next_state, terminated)` tuples. Every step of the game appends one tuple; every training step samples a random batch of 32.

Two problems this solves:

1. **Correlated samples.** Consecutive frames are almost identical. Training on them in order would feed the network a batch of 32 near-duplicates, and the gradient would be a strongly biased estimate of the true expected gradient. Random sampling from the buffer shuffles old and new experiences together, so each batch looks more like an IID sample from the agent's lifetime experience.
2. **Data re-use.** Every transition gets sampled many times over its lifetime in the buffer. A random interaction at step 20,000 can still be providing gradient signal at step 80,000. Massive sample efficiency win compared to on-policy methods that use each transition once.

The buffer has a 10,000-step **warmup** period during which no training happens. This fills it with enough variety that the first batches aren't all from one game.

### Target network — freezing the bootstrap target

Off-policy Q-learning's update rule is:

$$
Q(s, a) \;\leftarrow\; Q(s, a) + \alpha \bigl[\, r + \gamma \cdot \max_{a'} Q(s', a') - Q(s, a) \,\bigr]
$$

The target `r + γ · max Q(s', a')` depends on the same `Q` being updated. In the tabular case that's fine — changing one entry doesn't touch any other. With a neural net, the forward pass for `Q(s', a')` uses the same weights that are about to be updated, so every gradient step shifts the target itself. The network chases its own tail and diverges.

The fix is a **target network**: a frozen copy of the online network, used only to compute `max Q(s', a')`. It's updated by copying weights from the online network every `target_sync=1000` steps. Between copies, the target is fixed — a stable regression target the online network can actually converge on for a while.

```python
target_model = copy.deepcopy(model)
for p in target_model.parameters():
    p.requires_grad = False
```

In `train_step`:

```python
with torch.no_grad():
    next_q = target_model(next_states).max(dim=1).values
    target = rewards + gamma * next_q * (1 - dones)
```

`(1 - dones)` zeroes out the future term on terminal transitions, same pattern as the tabular projects — no future reward to bootstrap from when the episode is over.

## The loss — Huber instead of MSE

```python
loss = torch.nn.functional.smooth_l1_loss(q_sa, target)
```

Huber loss (aka smooth L1) is quadratic for small errors and linear for large ones. MSE would be tempting, but Bellman targets can have large outliers — an unexpected reward or a rare state can produce a TD error of ±10 when the typical error is ±0.1. MSE's quadratic tails turn those outliers into gradient spikes that blow up the weights. Huber caps the gradient magnitude at the transition point (|error| = 1 by default), which keeps training stable without sacrificing the quadratic shape near zero where most errors live.

## Hyperparameters — `train.py`

| Name | Value | Why |
|------|-------|-----|
| `gamma` | 0.99 | Long horizon — Breakout games last thousands of steps. γ close to 1 is needed so terminal-ish bricks-worth-points reward propagates backward. Not 1.0 because that risks divergence in a long/noisy domain. |
| `lr` | 1e-4 | Adam's standard DQN default. Lower than supervised learning because the target is non-stationary — the thing we're regressing to keeps moving. |
| `batch_size` | 32 | Nature DQN default. Small enough to train fast, large enough to average out batch noise. |
| `warmup` | 10,000 | Number of steps before training begins. Buffer needs variety before sampling makes sense. |
| `target_sync` | 1,000 | Steps between target-network copies. Smaller = more responsive but less stable; larger = more stable but slower to propagate new information. |
| `train_every` | 4 | Training step runs once per 4 environment steps. Most frames are near-duplicates anyway; skipping cuts compute without hurting learning. |
| `epsilon` | 1.0 → 0.1 over 100k steps | Linear decay. Much more aggressive exploration than projects 1/2 because the action-conditioned value landscape is genuinely unknown at startup (no optimistic init to lean on). |

Epsilon decay uses **environment steps**, not episodes — a single Breakout episode is much longer than a tic-tac-toe episode, and step count is the stable clock for "how much experience the agent has seen."

## The training loop — condensed

```
for episode in range(epochs):
    state = game.reset()
    while not game_over:
        action = agent.select_action(state)                  # ε-greedy forward pass
        next_state, reward, terminated, truncated = game.make_move(action)

        buffer.add_sample(state, action, reward, next_state, terminated)
        state = next_state
        step_count += 1
        agent.epsilon = max(0.1, 1.0 - step_count / 100_000)

        if len(buffer.samples) >= warmup and step_count % train_every == 0:
            batch = buffer.get_samples(batch_size)
            train_step(batch)                                # Huber loss + Adam

        if step_count % target_sync == 0:
            target_model.load_state_dict(model.state_dict()) # freeze new target
```

Three separate clocks are running on the same `step_count`:

- **ε schedule** — every step, a tiny bit more greedy.
- **Training cadence** — every 4th step, one gradient update (after warmup).
- **Target sync** — every 1000th step, copy the online weights into the target.

Keeping these independent is what lets DQN stably learn from the same stream of transitions without any of them interfering with the others.

## Observed behaviour

Trained for 3,000 episodes on CPU. Evaluation is 30 games at ε=0.05 (a sliver of exploration to break deterministic stuck-loops):

```
Average over 30 games: 15.6
Best game:             24
Worst game:            9
Std dev:               4.2
```

Not superhuman — the DeepMind reference implementation hits ~400 on Breakout after ~50M frames. This agent saw roughly 2% of that and ran on a laptop. What it's clearly learned:

- **Keep the paddle under the ball.** Survival time is reliably long. Worst game is 9 points, not 0.
- **Favour LEFT/RIGHT over NOOP/FIRE.** The action distribution at ε=0 is mostly lateral movement.
- **No tunnelling strategy yet.** The legendary "dig a channel up one side" emergent behaviour takes far more training than 3,000 episodes. Current play just bats the ball back until it gets past.

The 4.2-point standard deviation is mostly variance in *where* the ball deflects — small differences in paddle contact point cascade into wildly different brick-clearing sequences.

## Limitations and next steps

- **Double DQN.** Standard DQN's `max_a' Q_target(s', a')` systematically overestimates Q-values — the max of noisy estimates is biased upward. Double DQN decouples action selection (online net) from action evaluation (target net): `Q_target(s', argmax_a Q_online(s', a))`. Roughly a free stability win.
- **Prioritised replay.** Uniform sampling from the buffer is wasteful — transitions where the TD error is large carry more learning signal. Prioritised experience replay samples those more often.
- **Longer training + GPU.** 3k episodes on CPU is a proof of concept, not a fair test. Going to 50k+ on a GPU would let the tunnelling strategy emerge.
- **Dueling architecture.** Split the output head into a state-value stream `V(s)` and an advantage stream `A(s, a)`, recombine as `Q = V + (A - mean A)`. Helps when many actions have similar values (most frames in Breakout — only 1-2 actions per frame actually matter).
- **Frame skip inside the wrapper.** Gymnasium's Breakout-v5 already frame-skips internally; if swapped for NoFrameskip-v4 a custom skip wrapper becomes an extra lever.
