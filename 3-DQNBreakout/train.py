from CNN import Model
from agent import Agent
from game import Game
import gymnasium as gym
import ale_py
from gymnasium.wrappers import RecordVideo  
import random
from collections import deque
import copy
import torch

class ReplayBuffer:
    def __init__(self, capacity=100_000):                                                                                                                                                                                         
        self.samples = deque(maxlen=capacity)
    
    def add_sample(self, temp_state, action, reward, state, terminated):
        self.samples.append((temp_state, action, reward, state, terminated))
    
    def get_samples(self, num):
        return random.sample(self.samples, num)


architecture = [                                                                                                                                                                                                           
    ("Conv2d", 4, 32, 8, 4),    # (1,4,84,84)  -> (1,32,20,20)
    ("ReLU",),                                                                                                                                                                                                             
    ("Conv2d", 32, 64, 4, 2),   #              -> (1,64,9,9)
    ("ReLU",),                                                                                                                                                                                                             
    ("Conv2d", 64, 64, 3, 1),   #              -> (1,64,7,7)
    ("ReLU",),                                                                                                                                                                                                             
    ("Flatten",),               #              -> (1, 3136)
    ("Linear", 3136, 512),      #              -> (1, 512)                                                                                                                                                                 
    ("ReLU",),                                                                                                                                                                                                             
    ("Linear", 512, 4),         #              -> (1, 4)  <- Q-values!
  ]     

# set up environment
Training = True
if Training:
    render_mode = "rgb_array"
else:
    render_mode = "human"

gym.register_envs(ale_py)
env = gym.make("ALE/Breakout-v5", render_mode = render_mode, continuous=False)
if Training:
    env = RecordVideo(                                                                                           
      env,    
      video_folder="videos",
      episode_trigger=lambda ep: ep % 100 == 0,  # save every 100th episode                                                  
      name_prefix="breakout",
    )  


# set up network
model = Model(architecture)
target_model = copy.deepcopy(model)
for p in target_model.parameters():
    p.requires_grad = False

# set up game, agent, buffer, optimzer
agent = Agent(model, num_actions=4, epsilon=0.3)
buffer = ReplayBuffer()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
game = Game(env)
step_count = 0

def train_step(batch):
    states, actions, rewards, next_states, dones = zip(*batch)
    states = torch.cat(states)
    next_states = torch.cat(next_states)
    actions = torch.tensor(actions, dtype=torch.long)
    rewards = torch.tensor(rewards, dtype=torch.float16)
    dones = torch.tensor(dones, dtype=torch.float32)

    q_sa = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_q = target_model(next_states).max(dim=1).values
        target = rewards + gamma * next_q * (1-dones)
    
    loss = torch.nn.functional.smooth_l1_loss(q_sa, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


# hyperparameters
epochs = 3000
gamma = 0.99
step_count = 0
batch_size = 32
warmup = 10000
target_sync = 1000
train_every = 4

for episode in range(epochs):

    state, _ = game.reset()
    game_over = False
    ep_reward = 0.0

    while not game_over:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, info = game.make_move(action)
        game_over = terminated or truncated
        ep_reward += reward

        buffer.add_sample(state, action, reward, next_state, terminated)
        state = next_state
        step_count += 1
        agent.epsilon = max(0.1, 1.0 - step_count / 100_000)

        if len(buffer.samples) >= warmup and step_count % train_every == 0:
            batch = buffer.get_samples(batch_size)
            train_step(batch)
        
        if step_count % target_sync == 0:
            target_model.load_state_dict(model.state_dict())
    
    if Training and episode % 100 == 0:                                                                       
        torch.save(model.state_dict(), f"weights/breakout-episode-{episode}.pt")      
    print(f"episode {episode}  reward {ep_reward:.0f}  eps {agent.epsilon:.3f}  buffer {len(buffer.samples)}") 

