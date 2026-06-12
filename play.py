import torch
from game import SnakeGameAI
from agent import Agent

def play():
    
    game = SnakeGameAI()
    agent = Agent()
    
    
    agent.model.load_state_dict(torch.load('model/model_Final.pth'))
    agent.model.eval() 
    
    
    agent.eps = 0 
    
    print("Loading Production Brain. Commencing Showcase...")

    while True:
       
        state = agent.get_state(game)
        
       
        state0 = torch.tensor(state, dtype=torch.float)
        prediction = agent.model(state0)
        move = torch.argmax(prediction).item()
        
       
        final_move = [0, 0, 0]
        final_move[move] = 1

       
        reward, done, score = game.play_step(final_move)
        
      
        game.clock.tick(15) 

        if done:
            print(f'Game Over! Final Score: {score}')
            game.reset()

if __name__ == '__main__':
    play()