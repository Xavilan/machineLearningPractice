# Acrobot-v1  CartPole-v1 MountainCar-v0  MountainCarContinuous-v0 Pendulum-v0

import gym
env = gym.make('CartPole-v1')
env.reset()
for _ in range(500):
    env.render()
    env.step(env.action_space.sample())  # take a random action
env.close()
