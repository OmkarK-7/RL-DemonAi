import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import time

from src.envs.vizdoom_env import VizDoomBasic, VizDoomDefendCenter, VizDoomCorridor

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a Trained PPO Agent on ViZDoom")
    parser.add_argument('--scenario', type=str, default='basic', choices=['basic', 'defend', 'corridor'],
                        help="The ViZDoom scenario to evaluate on.")
    parser.add_argument('--model_path', type=str, required=True,
                        help="Path to the saved best_model.zip to evaluate.")
    parser.add_argument('--episodes', type=int, default=10,
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
    
    # Load model
    print(f"Loading model from {args.model_path}")
    model = PPO.load(args.model_path)
    
    # Evaluate policy
    print(f"Evaluating model over {args.episodes} episodes...")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=args.episodes)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")
    
    # Watch the agent play
    for episode in range(args.episodes): 
        obs = env.reset()
        done = False
        total_reward = 0
        while not done: 
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward
        print(f'Total Reward for visual episode {episode + 1} is {total_reward}')
        time.sleep(1) # Pause between episodes
        
    env.close()

if __name__ == '__main__':
    main()
