import gymnasium as gym
import numpy as np
import random

def f(x):

    return 2 * x - 1

class Env:

    def __init__(self, seedx, seedy, xbounds, ybounds):

        self.seedx = seedx
        self.seedy = seedy
        self.xpos = seedx
        self.ypos = seedy
        self.xbounds = xbounds
        self.ybounds = ybounds
        self.steps = 0

        self.xpaths = []
        self.ypaths = []

        self.TRUNCATE = 100

    def reset(self):
        self.xpos = self.seedx
        self.ypos = self.seedx
        self.steps = 0

        self.xpaths = []
        self.ypaths = []

        return np.array([self.xpos, self.ypos]), {}

    def step(self, action):

        if self.steps >= self.TRUNCATE:
            return np.array([self.xpos, self.ypos]), 0, True, True, {}

        self.xpos += f(action[0])
        self.ypos += f(action[1])

        if self.xpos < self.xbounds[0]:

            self.xpos = self.xbounds[0]

        elif self.xpos > self.xbounds[1]:

            self.xpos = self.xbounds[1]

        if self.ypos < self.ybounds[0]:

            self.ypos = self.ybounds[0]

        elif self.ypos > self.ybounds[1]:

            self.ypos = self.ybounds[1]

        self.steps +=1
        self.xpaths.append(self.xpos)
        self.ypaths.append(self.ypos)
        return np.array([self.xpos, self.ypos]), 0, False, False, {}

class EnvWrapper:

    def __init__(self, name, info):

        self.name = name
        self.info = info

        if (name == "2dbox"):

            xbounds = info["xbounds"]
            ybounds = info["ybounds"]

            self.env = Env(random.uniform(xbounds[0], xbounds[1]), random.uniform(ybounds[0], ybounds[1]), info["xbounds"], info["ybounds"])

            self.obs_space_dims = 2
            self.action_space_dims = 2

        else:

            self.env = gym.make(name)
            self.obs_space_dims = self.env.observation_space.shape[0]
            self.action_space_dims = self.env.action_space.shape[0]