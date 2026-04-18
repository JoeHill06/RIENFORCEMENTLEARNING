from agent import agent
from game import game
from pathlib import Path
import json
import random

# Create agents: alpha, gamma, epsilon, token
agent1 = agent(1, 1, 0.2, 1) # since out game is deterministic we can know our next state so we can make alpha large
agent2 = agent(1, 1, 0.2, 0) # we can make gamma 1 since it always ends in 9 moves or less so the future is bounded a win 8 moves away is as good as 1 move away

epochs = 30000
epsilon_start = agent1.epsilon
epsilon_end = 0.05
decay_fraction = 0.8
Decay_episodes = int(epochs * decay_fraction)


for episode in range(epochs):
    decayed = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * episode / Decay_episodes)
    agent1.epsilon = decayed
    agent2.epsilon = decayed

    Game = game()

    if episode % 2 == 0:
        player, player2 = agent1, agent2
    else:
        player, player2 = agent2, agent1

    # an agent can't update its Q-value until the opponent responds, so stash prior state+action
    player_state = None
    player_action = None
    player2_state = None
    player2_action = None

    while True:
        # --- player's turn ---
        state = player.get_state(Game.board)
        player_move = player.choose_move(Game.board)
        Game.move(player_move[0], player_move[1], player.token)

        if Game.check_win() == player.token:
            player.update(state, player_move, +1, Game.board, done=True)
            player.wins += 1
            if player2_state is not None:
                player2.update(player2_state, player2_action, -1, Game.board, done=True)
                player2.loss += 1
            break

        if Game.check_draw():
            player.update(state, player_move, 0, Game.board, done=True)
            player.draw += 1
            if player2_state is not None:
                player2.update(player2_state, player2_action, 0, Game.board, done=True)
                player2.draw += 1
            break

        # non-terminal: board has advanced, so update player2's prior move now
        if player2_state is not None:
            player2.update(player2_state, player2_action, 0, Game.board, done=False)

        # stash player's move to update after player2 responds
        player_state = state
        player_action = player_move

        # --- player2's turn ---
        state2 = player2.get_state(Game.board)
        player2_move = player2.choose_move(Game.board)
        Game.move(player2_move[0], player2_move[1], player2.token)

        if Game.check_win() == player2.token:
            player2.update(state2, player2_move, +1, Game.board, done=True)
            player2.wins += 1
            player.update(player_state, player_action, -1, Game.board, done=True)
            player.loss += 1
            break

        if Game.check_draw():
            player2.update(state2, player2_move, 0, Game.board, done=True)
            player2.draw += 1
            player.update(player_state, player_action, 0, Game.board, done=True)
            player.draw += 1
            break

        # non-terminal: update player's prior move now that board has advanced
        player.update(player_state, player_action, 0, Game.board, done=False)

        player2_state = state2
        player2_action = player2_move


    last_w1 = last_d1 = last_l1 = 0                                                                                                                                                                                  
    last_w2 = last_d2 = last_l2 = 0    
    if (episode + 1) % 100 == 0:                                                                                                                                                                                 
        w1 = agent1.wins - last_w1                                                                                                                                                                               
        d1 = agent1.draw - last_d1                        
        l1 = agent1.loss - last_l1                                                                                                                                                                               
        w2 = agent2.wins - last_w2                        
        d2 = agent2.draw - last_d2                                                                                                                                                                               
        l2 = agent2.loss - last_l2
                                                                                                                                                                                                                
        print(                                                                                                                                                                                                   
            f"ep {episode+1:>5} | "
            f"ε={agent1.epsilon:.3f} | "                                                                                                                                                                         
            f"A1 W/D/L {w1:>3}/{d1:>3}/{l1:>3} | "        
            f"A2 W/D/L {w2:>3}/{d2:>3}/{l2:>3} | "                                                                                                                                                               
            f"Q-table {len(agent1.q_table) + len(agent2.q_table):>5}"
        )                                                                                                                                                                                                        
                                                        
        last_w1, last_d1, last_l1 = agent1.wins, agent1.draw, agent1.loss
        last_w2, last_d2, last_l2 = agent2.wins, agent2.draw, agent2.loss


# ---- export the trained Q-table for the browser demo ----
# The frontend expects the agent to play X. agent1 trained as token=1 (X),
# so only its Q-table is serialized. Key format matches project 1:
#   9-char state (row-major) with '.' empty, 'X' cross, 'O' naught,
#   joined to "row,col" with a '|'. Example: ".XO......|1,2"
def encode_cell(v):
    if v == 1:
        return "X"
    if v == 0:
        return "O"
    return "."


def serialize(q_table):
    out = {}
    for (state, action), value in q_table.items():
        flat = "".join(encode_cell(c) for row in state for c in row)
        key = f"{flat}|{action[0]},{action[1]}"
        out[key] = value
    return out


out_dir = Path(__file__).parent.parent / "docs" / "projects" / "2-qlearning-tic-tac-toe"
out_dir.mkdir(parents=True, exist_ok=True)

for name, ag in [("agent1", agent1), ("agent2", agent2)]:
    data = serialize(ag.q_table)
    path = out_dir / f"q_table_{name}.json"
    with path.open("w") as f:
        json.dump(data, f)
    print(f"saved {len(data)} q-values to {path}")

# Mirror NOTES.md into the docs site so the project page can fetch it.
notes_src = Path(__file__).parent / "NOTES.md"
notes_dst = out_dir / "notes.md"
if notes_src.exists():
    notes_dst.write_text(notes_src.read_text())
    print(f"copied notes to {notes_dst}")

