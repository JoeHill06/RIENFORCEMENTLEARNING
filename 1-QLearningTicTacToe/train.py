from agent import agent
from game import game
from pathlib import Path
import json
import random

# create agent and set tokens
Agent1 = agent()
Agent1.token = 1         # cross
OPPONENT_TOKEN = 0       # naught


epochs = 20000

# ---- epsilon decay schedule ----
# Start fully random, decay linearly down to a small residual by the time
# we've used DECAY_FRACTION of training. The tail (the remaining episodes)
# runs at EPSILON_END so the agent spends the end of training refining the
# Q-values it actually cares about instead of exploring.
EPSILON_START   = 1.0
EPSILON_END     = 0.05
DECAY_FRACTION  = 0.8
DECAY_EPISODES  = int(epochs * DECAY_FRACTION)

# Rolling counts so we can see the agent improving over time.
wins = losses = draws = 0

for episode in range(epochs):
    # Linear epsilon decay. `max` clamps to EPSILON_END once we're past
    # the decay window, so the tail of training is almost pure exploitation.
   

    # Fresh game every episode - otherwise the board from the previous game would be used
    Game = game()

    # Alternate who opens the episode. Without this the Q-table only ever
    # sees "empty board, my turn" positions, so the agent has no idea what
    # to do when an opponent has already moved (which is exactly what
    # happens in the browser demo whenever the user clicks 'You first').
    # On odd episodes, let the random opponent play once before the main
    # loop starts; the loop below always runs agent-move-then-opponent.
    if episode % 2 == 1:
        opp_empty = [(i, j) for i in range(3) for j in range(3)
                     if Game.board[i][j] == -1]
        opp_i, opp_j = random.choice(opp_empty)
        Game.move(opp_i, opp_j, OPPONENT_TOKEN)

    # Buffer for the DELAYED update. The agent's move can't be scored
    # until after the opponent responds (a move that sets up the opponent
    # to win should be punished, not rewarded with 0). So we remember the
    # last (state, action) and apply the update one turn later.
    pending_state = None
    pending_action = None

    gameNotTerminal = True
    while gameNotTerminal:

        # ---- agent makes a move ----
        # Snapshot the board BEFORE the move. `state` must be frozen now,
        # or it will point at the same list we are about to mutate.
        state = [row[:] for row in Game.board]

        agent_move = Agent1.find_best_move(Game.board)
        Game.move(agent_move[0], agent_move[1], Agent1.token)

        # Did the agent's move end the game?
        if Game.check_win() == Agent1.token:
            Agent1.update(state, agent_move, +1, Game.board, done=True)
            wins += 1
            gameNotTerminal = False
            continue
        if Game.check_draw():
            Agent1.update(state, agent_move, 0, Game.board, done=True)
            draws += 1
            gameNotTerminal = False
            continue

        # Non-terminal: we do not know the reward yet. Stash the move so
        # we can update it once the opponent has responded.
        pending_state = state
        pending_action = agent_move

        # ---- random move is made in response ----
        board = Game.board
        empty = []
        for i in range(len(board)):
            for j in range(len(board[i])):
                if board[i][j] == -1:
                    empty.append((i, j))
        response_move = random.choice(empty)
        Game.move(response_move[0], response_move[1], OPPONENT_TOKEN)

        # ---- calculate reward & update agent ----
        # If the opponent just won or drew, blame/credit lands on the
        # agent's LAST move (the one we buffered above).
        if Game.check_win() == OPPONENT_TOKEN:
            Agent1.update(pending_state, pending_action, -1, Game.board, done=True)
            losses += 1
            gameNotTerminal = False
            continue
        if Game.check_draw():
            Agent1.update(pending_state, pending_action, 0, Game.board, done=True)
            draws += 1
            gameNotTerminal = False
            continue

        # Both players survived the round. Apply the pending update with
        # reward 0; `Game.board` is the state the agent will see next turn.
        Agent1.update(pending_state, pending_action, 0, Game.board, done=False)

    # Periodic progress report so you can watch learning happen.
    if (episode + 1) % 100 == 0:
        total = wins + losses + draws
        print(
            f"episode {episode + 1:>5} | "
            f"win {wins / total:.1%}  loss {losses / total:.1%}  draw {draws / total:.1%} "
            f"| qtable size {len(Agent1.q_table)}"
        )
        wins = losses = draws = 0


# ---- export the trained Q-table for the browser demo ----
# Q-table keys are (state, action) = (tuple-of-tuples, (row, col)). JSON
# has no tuple keys, so each key is encoded as a string:
#   9-char state (row-by-row) where '.' = empty, 'X' = cross (1, agent),
#   'O' = naught (0, opponent), joined to "row,col" with a '|'.
#   Example: ".XO......|1,2"
# The JS front-end re-parses this exact encoding to look moves up.
def encode_cell(v):
    if v == 1:
        return "X"
    if v == 0:
        return "O"
    return "."


serialized = {}
for (state, action), value in Agent1.q_table.items():
    flat = "".join(encode_cell(c) for row in state for c in row)
    key = f"{flat}|{action[0]},{action[1]}"
    serialized[key] = value

# Write straight into the docs site so the page can fetch it at runtime.
out_dir = Path(__file__).parent.parent / "docs" / "projects" / "1-qlearning-tic-tac-toe"
out_dir.mkdir(parents=True, exist_ok=True)
with (out_dir / "q_table.json").open("w") as f:
    json.dump(serialized, f)

print(f"saved {len(serialized)} q-values to {out_dir / 'q_table.json'}")

# Mirror NOTES.md into the docs site so the project page can fetch the
# latest notes without an extra build step. Keeping a copy (instead of a
# symlink) so GitHub Pages serves it reliably.
notes_src = Path(__file__).parent / "NOTES.md"
notes_dst = out_dir / "notes.md"
if notes_src.exists():
    notes_dst.write_text(notes_src.read_text())
    print(f"copied notes to {notes_dst}")
