# import arcade learning envionment ale_py
import torch
from collections import deque
import gymnasium as gym


class Game(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.frames = deque(maxlen=4)
        self.lives = 5
    
    def make_move(self, action):                                                                                                                                                                                                  
        obs, reward, terminated, truncated, info = self.env.step(action)
                                                                                                                                                                                                                                
        # lost a life? relaunch the ball                                                                                                                                                                                          
        current_lives = info.get("lives", self.lives)
        if current_lives < self.lives and not terminated:                                                                                                                                                                         
            obs, r, t, tr, info = self.env.step(1)  # FIRE
            reward += r                                                                                                                                                                                                           
            terminated = terminated or t
            truncated  = truncated  or tr                                                                                                                                                                                         
        self.lives = current_lives                                                                                                                                                                                                

        self.frames.append(self.screen_tensor(obs))                                                                                                                                                                               
        return self.stacked_state(), reward, terminated, truncated, info
    
    def stacked_state(self): # stacks 4 frames along channel dim (1,4,84,84)
        return torch.stack(list(self.frames)).unsqueeze(0)

    def screen_tensor(self, obs): # returns screen as pytorch (84,84)
        obs = torch.from_numpy(obs).float() / 255.0   # (210,160,3)
        obs = obs.mean(dim=2)                          # (210,160) — grayscale                                                                                                                                                             
        obs = obs.unsqueeze(0).unsqueeze(0)           # (1,1,210,160) for interpolate                                                                                                                                             
        obs = torch.nn.functional.interpolate(obs, size=(84,84))        # (1,1,84,84)                                                                                                                                                               
        return obs.squeeze(0).squeeze(0)              # (84,84)  
    
    def reset(self):                                                                                                                                                                                                                  
        obs, info = self.env.reset()
        obs, _, _, _, _ = self.env.step(1)   # FIRE to launch the ball
        frame = self.screen_tensor(obs)
        for i in range(4):
            self.frames.append(frame)
        return self.stacked_state(), info   

        

