import gym
import time
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("Taxi-v3")
env.reset()
env.render()
