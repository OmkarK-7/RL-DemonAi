import os
import cv2
import numpy as np
from gym import Env
from gym.spaces import Discrete, Box
from vizdoom import DoomGame

class VizDoomBase(Env): 
    """Base OpenAI Gym environment wrapper for ViZDoom."""
    def __init__(self, render: bool = False, config_path: str = '', action_size: int = 3): 
        super().__init__()
        self.game = DoomGame()
        self.game.load_config(config_path)
        
        # Render frame logic
        self.game.set_window_visible(render)
        self.game.init()
        
        # Action space and observation space (Grayscale image)
        self.observation_space = Box(low=0, high=255, shape=(100, 160, 1), dtype=np.uint8) 
        self.action_space = Discrete(action_size)
        self.action_size = action_size
        
    def render(self): 
        pass
    
    def reset(self): 
        self.game.new_episode()
        state = self.game.get_state().screen_buffer
        return self.grayscale(state)
    
    def grayscale(self, observation: np.ndarray) -> np.ndarray:
        """Grayscale the game frame and resize it."""
        gray = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (160, 100), interpolation=cv2.INTER_CUBIC)
        state = np.reshape(resize, (100, 160, 1))
        return state
    
    def close(self): 
        self.game.close()

class VizDoomBasic(VizDoomBase):
    def __init__(self, render: bool = False):
        super().__init__(render, 'scenarios/basic.cfg', action_size=3)
        
    def step(self, action: int):
        actions = np.identity(self.action_size, dtype=np.uint8)
        reward = self.game.make_action(actions[action], 4) 
        
        if self.game.get_state(): 
            state = self.grayscale(self.game.get_state().screen_buffer)
            ammo = self.game.get_state().game_variables[0]
            info = {"ammo": ammo}
        else: 
            state = np.zeros(self.observation_space.shape)
            info = {"ammo": 0}
        
        done = self.game.is_episode_finished()
        return state, reward, done, info 

class VizDoomDefendCenter(VizDoomBase):
    def __init__(self, render: bool = False):
        super().__init__(render, 'scenarios/defend_the_center.cfg', action_size=3)
        
    def step(self, action: int):
        actions = np.identity(self.action_size, dtype=np.uint8)
        reward = self.game.make_action(actions[action], 4) 
        
        if self.game.get_state(): 
            state = self.grayscale(self.game.get_state().screen_buffer)
            ammo = self.game.get_state().game_variables[0]
            info = {"ammo": ammo}
        else: 
            state = np.zeros(self.observation_space.shape)
            info = {"ammo": 0}
        
        done = self.game.is_episode_finished()
        return state, reward, done, info 

class VizDoomCorridor(VizDoomBase):
    def __init__(self, render: bool = False):
        super().__init__(render, 'scenarios/deadly_corridor_s1.cfg', action_size=7)
        self.damage_taken = 0
        self.hitcount = 0
        self.ammo = 52 
        
    def step(self, action: int):
        actions = np.identity(self.action_size, dtype=np.uint8)
        movement_reward = self.game.make_action(actions[action], 4) 
        
        reward = 0 
        if self.game.get_state(): 
            state = self.grayscale(self.game.get_state().screen_buffer)
            
            # Reward shaping
            game_variables = self.game.get_state().game_variables
            health, damage_taken, hitcount, ammo = game_variables
            
            # Calculate reward deltas
            damage_taken_delta = -damage_taken + self.damage_taken
            self.damage_taken = damage_taken
            hitcount_delta = hitcount - self.hitcount
            self.hitcount = hitcount
            ammo_delta = ammo - self.ammo
            self.ammo = ammo
            
            reward = movement_reward + damage_taken_delta * 10 + hitcount_delta * 200 + ammo_delta * 5 
            info = {"ammo": ammo, "health": health}
        else: 
            state = np.zeros(self.observation_space.shape)
            info = {"ammo": 0, "health": 0}
        
        done = self.game.is_episode_finished()
        return state, reward, done, info 
