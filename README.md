# RL-DemonAi: Autonomous Agents in ViZDoom

A robust Reinforcement Learning project combining classical Exploration-Exploitation strategies with Modern Deep RL representations to train autonomous agents in complex, 3D first-person shooter environments. 

## 🧠 Project Overview
This repository showcases the implementation of diverse Reinforcement Learning algorithms, focusing on the highly complex **ViZDoom** engine. 
It demonstrates proficiency in crafting reward functions, environment wrapping, and optimizing policies using `Stable-Baselines3`.

### Key Features
* **Multi-Armed Bandits from Scratch**: Implementations of foundational exploration strategies including `Epsilon-Greedy`, `UCB`, `KL-UCB`, and `Thompson Sampling` to build a strong theoretical RL foundation.
* **Deep Reinforcement Learning (PPO)**: Using Proximal Policy Optimization (PPO) with Convolutional Neural Networks (CNN) to process raw pixel data from the ViZDoom engine.
* **Diverse Scenarios**: 
    - `Basic`: Finding and shooting a static monster.
    - `Defend the Center`: Surviving in a circular arena against approaching enemies.
    - `Deadly Corridor`: Navigating a hallway packed with enemies while managing health, damage, and ammo through complex Custom Reward Shaping.
* **Production Architecture**: A modular, script-based application structure designed for MLOps and automated training.

## ⚙️ Installation & Usage

### 1. Local Setup
Ensure you have Python 3.9+ installed.
```bash
git clone https://github.com/your-username/RL-DemonAi.git
cd RL-DemonAi

# Install the required dependencies including ViZDoom and PyTorch
pip install -r requirements.txt
```

### 2. Training an Agent
You can train an agent locally using the modular training script. Select the scenario you want to train on:
```bash
# Train on the basic scenario
python -m src.train --scenario basic --timesteps 100000

# Train on the complex corridor scenario
python -m src.train --scenario corridor --timesteps 500000
```

### 3. Evaluating an Agent
Watch the agent play the game in real-time by passing your trained model file:
```bash
# Evaluate a trained model visually
python -m src.evaluate --scenario basic --model_path ./train/train_basic/best_model_10000.zip

# Or evaluate a completely untrained agent taking random actions 
python -m src.evaluate_random --scenario basic
```

## 🏗️ Architecture
```
RL-DemonAi/
├── src/
│   ├── envs/
│   │   └── vizdoom_env.py      # OpenAI Gym wrappers for ViZDoom scenarios
│   ├── agents/
│   │   └── bandits.py          # Classical Multi-Armed Bandit implementations
│   ├── utils/
│   │   └── callbacks.py        # Checkpointing and Logging utilities
│   ├── train.py                # Main PPO Deep RL Training Script
│   ├── evaluate.py             # Visual Evaluation Script
│   └── evaluate_random.py      # Random Action Evaluation
├── scenarios/                  # ViZDoom Configuration and WAD files
└── requirements.txt            # Locked package dependencies
```

## 📈 Next Steps & Capabilities
The environment framework allows extending training across the 30+ different native ViZDoom scenarios. Custom curriculum learning and alternative deep architectures (DQN, A2C) can be quickly integrated into the `train.py` loop. 
