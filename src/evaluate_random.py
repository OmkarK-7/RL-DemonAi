import argparse
import time
import random

from src.envs.vizdoom_env import VizDoomBasic, VizDoomDefendCenter, VizDoomCorridor

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate an Untrained (Random) Agent on ViZDoom")
    parser.add_argument('--scenario', type=str, default='basic', choices=['basic', 'defend', 'corridor'],
                        help="The ViZDoom scenario to evaluate on.")
    parser.add_argument('--episodes', type=int, default=3,
                        help="Number of episodes to evaluate.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Select environment with rendering ENABLED
    if args.scenario == 'basic':
        env = VizDoomBasic(render=True)
    elif args.scenario == 'defend':
        env = VizDoomDefendCenter(render=True)
    elif args.scenario == 'corridor':
        env = VizDoomCorridor(render=True)
    else:
        raise ValueError("Invalid scenario selected")
    
    # Evaluate policy using random actions
    print(f"Evaluating untrained agent over {args.episodes} episodes...")
    
    for episode in range(args.episodes): 
        obs = env.reset()
        done = False
        total_reward = 0
        while not done: 
            action = env.action_space.sample() # Random action!
            obs, reward, done, info = env.step(action)
            total_reward += reward
            time.sleep(0.04) # Slow it down so the user can see
        print(f'Total Reward for untrained episode {episode + 1} is {total_reward}')
        time.sleep(1) # Pause between episodes
        
    env.close()

if __name__ == '__main__':
    main()
