import argparse
import os
from stable_baselines3 import PPO

from src.envs.vizdoom_env import VizDoomBasic, VizDoomDefendCenter, VizDoomCorridor
from src.utils.callbacks import TrainAndLoggingCallback

def parse_args():
    parser = argparse.ArgumentParser(description="Train a PPO Agent on ViZDoom")
    parser.add_argument('--scenario', type=str, default='basic', choices=['basic', 'defend', 'corridor'],
                        help="The ViZDoom scenario to train on.")
    parser.add_argument('--timesteps', type=int, default=100000,
                        help="Total number of timesteps to train the agent.")
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help="Learning rate for PPO.")
    parser.add_argument('--n_steps', type=int, default=2048,
                        help="Number of steps to run for each environment per update.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Select environment
    if args.scenario == 'basic':
        env = VizDoomBasic()
        n_steps = args.n_steps
    elif args.scenario == 'defend':
        env = VizDoomDefendCenter()
        n_steps = 4096
    elif args.scenario == 'corridor':
        env = VizDoomCorridor()
        n_steps = 8192
    else:
        raise ValueError("Invalid scenario selected")
    
    # Configure logging and pathways
    CHECKPOINT_DIR = f'./train/train_{args.scenario}'
    LOG_DIR = f'./logs/log_{args.scenario}'
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)
    
    # Initialize agent
    model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, 
                learning_rate=args.learning_rate, n_steps=n_steps)
    
    # In corridor scenario we need some custom hyperparameters from notebook
    if args.scenario == 'corridor':
        model.clip_range = 0.1
        model.gamma = 0.95
        model.gae_lambda = 0.9
        model.learning_rate = 0.00001
        
    print(f"Starting training for {args.scenario} scenario for {args.timesteps} timesteps...")
    model.learn(total_timesteps=args.timesteps, callback=callback)
    print("Training finished!")
    env.close()

if __name__ == '__main__':
    main()
