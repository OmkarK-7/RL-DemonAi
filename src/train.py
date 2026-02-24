import argparse
import os
import wandb
from wandb.integration.sb3 import WandbCallback
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
        # Corridor uses a smaller specialized learning rate
        args.learning_rate = 0.00001
    else:
        raise ValueError("Invalid scenario selected")
    
    # Configure logging and pathways
    CHECKPOINT_DIR = f'./train/train_{args.scenario}'
    LOG_DIR = f'./logs/log_{args.scenario}'
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)
    
    # Initialize WandB run
    run = wandb.init(
        project="RL-DemonAi",
        config={
            "scenario": args.scenario,
            "timesteps": args.timesteps,
            "learning_rate": args.learning_rate,
            "n_steps": n_steps,
        },
        sync_tensorboard=True,  # Auto-upload sb3's tensorboard metrics
        monitor_gym=True,       # Auto-upload the videos of agents playing the game
        save_code=True,         # Auto-save the main python script
    )
    
    # WandbCallback automatically logs all metrics
    wandb_callback = WandbCallback(
        gradient_save_freq=1000,
        model_save_path=f"models/{run.id}",
        verbose=2,
    )
    
    # Setup PPO hyperparameters
    ppo_kwargs = {
        'policy': 'CnnPolicy',
        'env': env,
        'tensorboard_log': LOG_DIR,
        'verbose': 1,
        'learning_rate': args.learning_rate,
        'n_steps': n_steps,
    }
    
    # In corridor scenario we need some custom hyperparameters from notebook
    if args.scenario == 'corridor':
        ppo_kwargs['clip_range'] = 0.1
        ppo_kwargs['gamma'] = 0.95
        ppo_kwargs['gae_lambda'] = 0.9
        
    # Initialize agent
    model = PPO(**ppo_kwargs)
        
    print(f"Starting training for {args.scenario} scenario for {args.timesteps} timesteps...")
    # List of callbacks format
    model.learn(total_timesteps=args.timesteps, callback=[callback, wandb_callback])
    print("Training finished!")
    env.close()
    run.finish()

if __name__ == '__main__':
    main()
