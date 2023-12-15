import numpy as np

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

        self.TRUNCATE = 25

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

        self.xpos += action[0]
        self.ypos += action[1]

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