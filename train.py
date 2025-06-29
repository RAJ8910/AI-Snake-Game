from agent import Agent
from game import SnakeGame
import matplotlib.pyplot as plt


def plot(scores, mean_scores):
    plt.figure(1)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores, label="Score")
    plt.plot(mean_scores, label="Mean Score")
    plt.ylim(ymin=0)

    if scores: 
        plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    if mean_scores:
        plt.text(len(mean_scores)-1, mean_scores[-1], str(round(mean_scores[-1], 1)))
    plt.legend()
    plt.draw()
    plt.pause(0.001)

def train():
    scores = []
    mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGame()

    try:
        while True:
            state_old = agent.get_state(game)
            final_move = agent.get_action(state_old)
            reward, done, score = game.play_step(final_move)
            state_new = agent.get_state(game)
            # print("Train:",state_old)
            agent.train_short_memory(state_old, final_move, reward, state_new, done)
            agent.remember(state_old, final_move, reward, state_new, done)

            if done:
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()

                if score > record:
                    record = score
                    agent.model.save()

                print(f'Game {agent.n_games} - Score: {score} - Record: {record}')
                scores.append(score)
                total_score += score
                mean_score = total_score / agent.n_games
                mean_scores.append(mean_score)
            
    except KeyboardInterrupt:
        print("Training interrupted. Saving latest model...")
    finally:
        agent.model.save("model_final.pth")
        print("Model saved to model_final.pth.")
        plot(scores, mean_scores)
    
if __name__ == '__main__':
    train()
