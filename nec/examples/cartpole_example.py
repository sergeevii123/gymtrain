import gym

from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from env_wrappers.cartpole_wrapper import CartpoleWrapper
from models.models import CartPoleDQN
from nec_agent import NECAgent

def main():
  env = PongWrapper(gym.make('CartPole-v0'))
  embedding_model = CartPoleDQN(5)
  agent = NECAgent(env, embedding_model)
  agent.train()

if __name__ == '__main__' and __package__ is None:
  main()        
