# Reinforcement Learning Agent for Golf Simulation

## Project Overview

This repository contains the implementation of a Reinforcement Learning (RL) agent trained to play a simplified 2D golf game. The project is divided into three main components: a Golf Simulator, a Gymnasium Environment, and RL training code.

## Table of Contents
- [Introduction](#introduction)
- [Golf Simulator](#golf-simulator)
- [Gymnasium Environment](#gymnasium-environment)
- [Reinforcement Learning](#reinforcement-learning)
- [Installation](#installation)
- [Usage](#usage)

## Introduction

The goal of this project is to train an RL agent to play golf by interacting with a custom-built golf simulator. The simulator models golf as a Markov Decision Process (MDP), where the state space includes the ball's current location, lie, and course layout. The action space involves selecting a direction to aim and a club to hit. Rewards are structured to penalize shots and out-of-bounds or hazard landings, with the episode ending when the ball reaches the green.

## Golf Simulator

The Golf Simulator forms the foundation of this project, providing a realistic 2D golf environment. This section describes the various files that make up the simulator:

- **`course.py`**: Generates random courses with varying shapes and hazard locations.
- **`ball.py`**: Manages the ball's position and handles ball movement animations.
- **`aiming.py`**: Implements the aiming mechanism, including a Gaussian overlay and direction arrow.
- **`game.py`**: Contains the main game logic, including score keeping and event handling (e.g., mouse clicks).
- **`main.py`**: Runs the main PyGame loop.
- **`ui.py`**: Manages UI elements such as club selection, scoreboard, and buttons.
- **`utils.py`**: Provides miscellaneous utilities like Bézier curve generation.
- **`constants.py`**: Defines constants such as colors and graphical settings.
- **`profile.json`**: Stores a lookup table for the distances and horizontal/vertical standard deviations for each club and lie combination for a specific golfer.

These files work together to create a dynamic and interactive golf simulation, allowing for infinite course variations and realistic gameplay mechanics.

## Gymnasium Environment

The Gymnasium Environment wraps the Golf Simulator into a format compatible with RL algorithms, offering a simple API for interaction:
- **State Representation**: Includes the ball's current location, lie, and course layout.
- **Action Space**: Consists of aiming direction and club selection.
- **Reward Structure**: Provides feedback based on the shot outcome and course rules.

### Key Functions
- `reset()`: Resets the environment to the starting state.
- `step(action)`: Takes an action and returns the new state, reward, and episode status.
- `render()`: Visualizes the current state of the environment.

## Reinforcement Learning

This component focuses on training the RL agent using Proximal Policy Optimization (PPO). It includes:
- **Training**: Code to train the agent through interactions with the Gymnasium Environment.
- **Evaluation**: Tools to evaluate the performance of the trained agent.
- **Visualization**: Utilities to visualize training progress and results.

### Key Scripts
- `train.py`: Script to train the RL agent using PPO.
- `evaluate.py`: Script to evaluate the trained agent.
- `visualize.py`: Script to visualize training metrics and performance.

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/LucasBosman/234_final.git
cd golf-rl-agent
pip install -r requirements.txt
```

## Usage

### Running the Simulator
```bash
python golf_simulator.py
```

### Training the RL Agent
```bash
python train.py
```

### Evaluating the RL Agent
```bash
python evaluate.py
```

### Visualizing Training Results
```bash
python visualize.py
```

Feel free to explore the code, experiment with different parameters, and improve the RL agent's performance. Happy golfing!