import gym

from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from env_wrappers.pong_wrapper import PongWrapper
from models.models import AtariDQN
from nec_agent import NECAgent

def main():
  env = PongWrapper(gym.make('Pong-v0'))
  embedding_model = AtariDQN(5)
  agent = NECAgent(env, embedding_model)
  agent.train()

if __name__ == '__main__' and __package__ is None:
  main()        
