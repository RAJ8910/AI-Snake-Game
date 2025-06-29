import random
import numpy as np
from collections import deque
from game import Direction, Point,BLOCK_SIZE
from model import Linear_QNet, QTrainer
import torch
import os 


MAX_MEMORY = 1_000_000  
BATCH_SIZE = 2048      
LR = 0.001             


class Agent:
    def __init__(self):
        self.n_games = 0 
        self.epsilon = 0  
        self.gamma = 0.91 
        self.memory = deque(maxlen=MAX_MEMORY) 

        self.model = Linear_QNet(14, 512, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

        model_name = 'model_final.pth' 
        model_path = os.path.join(os.getcwd(), model_name) 

        if os.path.exists(model_path):
            print(f"Loading pre-trained model from {model_path}")
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval() 
        else:
            print("No pre-trained model found. Starting training from scratch.")


    def _get_normalized_distance(self, distance):
        if distance == 0:
            return 1.0  
        if distance < BLOCK_SIZE:
            return 1.0
        return 0.0 

    def get_state(self, game):

        head = game.snake[0] 
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        straight_point = None
        right_point = None
        left_point = None

        if dir_r: 
            straight_point = Point(head.x + BLOCK_SIZE, head.y)      
            right_point = Point(head.x, head.y + BLOCK_SIZE)        
            left_point = Point(head.x, head.y - BLOCK_SIZE)         
        elif dir_l:  
            straight_point = Point(head.x - BLOCK_SIZE, head.y)    
            right_point = Point(head.x, head.y - BLOCK_SIZE)        
            left_point = Point(head.x, head.y + BLOCK_SIZE)         
        elif dir_u:  
            straight_point = Point(head.x, head.y - BLOCK_SIZE)    
            right_point = Point(head.x + BLOCK_SIZE, head.y)         
            left_point = Point(head.x - BLOCK_SIZE, head.y)         
        else:  
            straight_point = Point(head.x, head.y + BLOCK_SIZE)     
            right_point = Point(head.x - BLOCK_SIZE, head.y)      
            left_point = Point(head.x + BLOCK_SIZE, head.y)       

        state = [
            game.is_collision(straight_point), 
            game.is_collision(right_point),   
            game.is_collision(left_point),    

            dir_l, 
            dir_r, 
            dir_u, 
            dir_d,

            game.food.x < game.head.x,  
            game.food.x > game.head.x, 
            game.food.y < game.head.y,  
            game.food.y > game.head.y   
        ]

        max_dist = 400  

        def scan_direction(base_point_for_vector):
            
            dx_step = 0 
            dy_step = 0

            if base_point_for_vector.x > head.x: dx_step = BLOCK_SIZE
            elif base_point_for_vector.x < head.x: dx_step = -BLOCK_SIZE
            if base_point_for_vector.y > head.y: dy_step = BLOCK_SIZE
            elif base_point_for_vector.y < head.y: dy_step = -BLOCK_SIZE

            for i in range(1, (max_dist // BLOCK_SIZE) + 1):
                check_x = head.x + i * dx_step
                check_y = head.y + i * dy_step
                check_point = Point(check_x, check_y)

                if not (0 <= check_x < game.w and 0 <= check_y < game.h):
                    return 1 / i 

                if check_point in game.snake[1:]:
                    return 1 / i 
            
            return 0.0 

        dist_straight = scan_direction(straight_point)
        dist_right = scan_direction(right_point)
        dist_left = scan_direction(left_point)
        state.extend([dist_straight, dist_right, dist_left])
        return np.array(state, dtype=np.float32)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) 
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.model.eval() 
        self.trainer.train_step(state, action, reward, next_state, done)
        self.model.train() 

    def get_action(self, state):
        self.epsilon = 80 - self.n_games 
        final_move = [0, 0, 0] 
        if random.randint(0, 100) < self.epsilon:
            move = random.randint(0, 2) #
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            self.model.eval() 
            if state0.ndim == 1:
                state0 = torch.unsqueeze(state0, 0)
            
            prediction = self.model(state0) 
            move = torch.argmax(prediction).item() 
            final_move[move] = 1 
            self.model.train() 
            
        return final_move
