# Snake AI with Deep Q-Learning

An AI that learns to play Snake using reinforcement learning. The agent starts with no knowledge and gets better through trial and error.

## What it does

- AI observes the game state (dangers, food location, direction)
- Makes decisions using a neural network (go straight, turn left/right)
- Gets rewards for eating food (+20) and penalties for crashing (-20/-30)
- Learns from experience to play better over time

## Neural Network Input (14 features)

```
[danger_straight, danger_right, danger_left,          # Collision detection (3)
 dir_left, dir_right, dir_up, dir_down,               # Current direction (4) 
 food_left, food_right, food_up, food_down,           # Food direction (4)
 distance_straight, distance_right, distance_left]    # Obstacle distances (3)
```

**Output**: 3 actions [straight, turn_right, turn_left]

## Files

- `agent.py` - AI brain that makes decisions and learns
- `model.py` - Neural network and training logic
- `game.py` - Snake game built with Pygame
- `train.py` - Main training loop

## How to run

```bash
pip install pygame torch numpy matplotlib
python train.py
```

Watch the AI learn in real-time! It starts playing randomly but gradually develops strategies like avoiding walls and finding efficient paths to food.

## Results

With training, the AI learns to:
- Avoid collisions
- Navigate efficiently to food
- Score 30+ points consistently after 100+ iterations
- Plan ahead to avoid traps
