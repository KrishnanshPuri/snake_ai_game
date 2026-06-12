import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEM = 100_000
B_SIZE = 1000
LR = 0.001 # Lowered to prevent catastrophic forgetting

class Agent:
    def __init__(self):
        self.n_games = 0
        self.eps = 1.0      
        self.eps_min = 0.01 
        self.eps_dec = 0.995 
        self.gamma = 0.9     
        self.mem = deque(maxlen=MAX_MEM)
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        pt_l = Point(head.x - 20, head.y)
        pt_r = Point(head.x + 20, head.y)
        pt_u = Point(head.x, head.y - 20)
        pt_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(pt_r)) or 
            (dir_l and game.is_collision(pt_l)) or 
            (dir_u and game.is_collision(pt_u)) or 
            (dir_d and game.is_collision(pt_d)),

            # Danger right
            (dir_u and game.is_collision(pt_r)) or 
            (dir_d and game.is_collision(pt_l)) or 
            (dir_l and game.is_collision(pt_u)) or 
            (dir_r and game.is_collision(pt_d)),

            # Danger left
            (dir_d and game.is_collision(pt_r)) or 
            (dir_u and game.is_collision(pt_l)) or 
            (dir_r and game.is_collision(pt_u)) or 
            (dir_l and game.is_collision(pt_d)),
            
            # Move dir
            dir_l, dir_r, dir_u, dir_d,
            
            # Food loc
            game.food.x < game.head.x,  # left
            game.food.x > game.head.x,  # right
            game.food.y < game.head.y,  # up
            game.food.y > game.head.y   # down
        ]
        return np.array(state, dtype=int)
    
    def remember(self, state, action, reward, next_state, done):
        self.mem.append((state, action, reward, next_state, done))

    def train_long_mem(self):
        if len(self.mem) > B_SIZE:
            sample = random.sample(self.mem, B_SIZE)
        else:
            sample = self.mem
            
        states, actions, rewards, next_states, dones = zip(*sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_mem(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # Epsilon decay math
        self.eps = max(self.eps_min, self.eps * self.eps_dec)
        move = [0, 0, 0]
        
        if random.random() < self.eps:
            idx = random.randint(0, 2)
            move[idx] = 1
        else:
            s0 = torch.tensor(state, dtype=torch.float)
            pred = self.model(s0)
            idx = torch.argmax(pred).item()
            move[idx] = 1
            
        return move

def train():
    scores = []
    mean_scores = []
    tot_score = 0
    rec = 0
    agent = Agent()
    game = SnakeGameAI()
    
    while True:
        s_old = agent.get_state(game)
        move = agent.get_action(s_old)
        reward, done, score = game.play_step(move)
        s_new = agent.get_state(game)

        agent.train_short_mem(s_old, move, reward, s_new, done)
        agent.remember(s_old, move, reward, s_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_mem()

            if score > rec:
                rec = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record', rec)
            
            scores.append(score)
            tot_score += score
            mean_score = tot_score / agent.n_games
            mean_scores.append(mean_score)
            plot(scores, mean_scores)

if __name__ == '__main__':
    train()