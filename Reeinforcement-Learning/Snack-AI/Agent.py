import torch
import random
import numpy as np
from collections import deque
from Game import SnakeGameAI, Direction, Point, BLOCK_SIZE, SPEED
from Model import Linear_QNet, QTrainer
from Helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0                        # randomness
        self.gamma = 0.9                        # discount rate (smaller than 1)
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft() after memory overlap

        # Input:  11 states
        # Middle: 256 neural
        # Output: 3 [straight, right, left]

        self.model = Linear_QNet(11, 256, 3)

        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):

        head = game.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # Create the list with 11 values
        state = [
            # Danger Straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger Right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move Direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food Location
            game.food.x < game.head.x,      # food left
            game.food.x > game.head.x,      # food right
            game.food.y < game.head.y,      # food up
            game.food.y > game.head.y       # food down
        ]

        # Convert the list on a Numpy array
        # Dtype=int --> Convert bool into [0 or 1] scale
        return np.array(state, dtype=int)


    def remember(self, state, action, reward, nest_state, done):
        # Save into the deque as tuple
        self.memory.append((state, action, reward, nest_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)    # return list of tuples
        else:
            mini_sample = self.memory

        # Extract the content
        # zip(*) to zip all content together (all states, all action ... like a for loop)
        states, actions, rewards, nest_states, dones = zip(*mini_sample)

        # Train the model
        self.trainer.train_step(states, actions, rewards, nest_states, dones)

    def train_short_memory(self, state, action, reward, nest_state, done):
        self.trainer.train_step(state, action, reward, nest_state, done)

    def get_action(self, state):
        # Random moves: tradeoff Exploration / Exploitation

        # Exploration: Find more about Environment
        #              More immediate Rewards for maximizing Future Rewards

        # Exploitation: Profitable Information about the Environment to maximize immediate Rewards

        # Example:
        # Exploitation: Going to same favourite restaurant
        # Exploration:  Trying out new restaurants

        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]

        # The more games we do, epsilon increases and less random moves we had!
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1

        # Predict the next state with the model
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            # Max number (probability)
            # [5.41, 2.19, 0.65] --> straight
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_scores = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()

    while True:
        # get current state
        state = agent.get_state(game)

        # get move based on state
        final_move = agent.get_action(state)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)

        # get new state
        new_state = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state, final_move, reward, new_state, done)

        # remember
        agent.remember(state, final_move, reward, new_state, done)

        if done:
            # reset the game
            game.reset()
            agent.n_games += 1

            # train long memory
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record', record)

            plot_scores.append(score)
            total_scores += score
            mean_score = total_scores / agent.n_games
            plot_mean_scores.append(mean_score)

            # Plot Scores and Mean scores
            plot(plot_scores, plot_mean_scores)




if __name__ == '__main__':
    train()