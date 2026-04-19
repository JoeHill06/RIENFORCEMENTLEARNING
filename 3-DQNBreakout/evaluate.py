from CNN import Model                                                                                                                                                                  
from agent import Agent                                                                                                                                                                  
from game import Game
import gymnasium as gym                                                                                                                                                                  
import ale_py                                                                                                                                                                            
from gymnasium.wrappers import RecordVideo
import torch                                                                                                                                                                             
                                                                                                                                                                                        
architecture = [
    ("Conv2d", 4, 32, 8, 4), ("ReLU",),                                                                                                                                                  
    ("Conv2d", 32, 64, 4, 2), ("ReLU",),                                                                                                                                                 
    ("Conv2d", 64, 64, 3, 1), ("ReLU",),                                                                                                                                                 
    ("Flatten",),                                                                                                                                                                        
    ("Linear", 3136, 512), ("ReLU",),                                                                                                                                                    
    ("Linear", 512, 4),                                                                                                                                                                
]                                                                                                                                                                                        
                                                                                                                                                                                        
WEIGHTS = "weights/breakout-episode-2900.pt"   # <- pick the checkpoint                                                                                                                  
WATCH = True                                    # True = live human window                                                                                                             
EPISODES = 30                                   # how many test games                                                                                                                    
EPSILON = 0.05                                  # small noise prevents "stuck" loops                                                                                                     
                                                                                                                                                                                        
gym.register_envs(ale_py)                                                                                                                                                                
render_mode = "human" if WATCH else "rgb_array"                                                                                                                                          
env = gym.make("ALE/Breakout-v5", render_mode=render_mode, continuous=False)                                                                                                             
if not WATCH:                                                                                                                                                                            
    env = RecordVideo(env, video_folder="eval_videos",                                                                                                                                   
                    episode_trigger=lambda ep: True,  # record every episode                                                                                                           
                    name_prefix="eval")                                                                                                                                                
                                                                                                                                                                                        
# load the model                                                                                                                                                                         
model = Model(architecture)                                                                                                                                                            
model.load_state_dict(torch.load(WEIGHTS))                                                                                                                                               
model.eval()                                    # inference mode                                                                                                                       
agent = Agent(model, num_actions=4, epsilon=EPSILON)                                                                                                                                     

game = Game(env)                                                                                                                                                                         
rewards = []                                                                                                                                                                           
                                                                                                                                                                                        
for ep in range(EPISODES):                                                                                                                                                             
    state, _ = game.reset()
    done = False
    ep_reward = 0.0
    while not done:                                                                                                                                                                      
        with torch.no_grad():
            action = agent.select_action(state)                                                                                                                                          
        state, reward, terminated, truncated, _ = game.make_move(action)                                                                                                               
        done = terminated or truncated                                                                                                                                                   
        ep_reward += reward
    rewards.append(ep_reward)                                                                                                                                                            
    print(f"episode {ep}: {ep_reward:.0f}")                                                                                                                                            
                                                                                                                                                                                        
env.close()
                                                                                                                                                                                        
import statistics                                                                                                                                                                      
print(f"\nAverage over {EPISODES} games: {statistics.mean(rewards):.1f}")
print(f"Best game: {max(rewards):.0f}")                                                                                                                                                  
print(f"Worst game: {min(rewards):.0f}")                                                                                                                                                 
print(f"Std dev:    {statistics.stdev(rewards):.1f}")   