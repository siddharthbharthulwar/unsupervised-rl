import gymnasium as gym

env = gym.make('HumanoidStandup-v4', render_mode='human')
env.reset()

for _ in range(1000):
    env.render()
    action = env.action_space.sample()  # Replace this with your action
    env.step(action)
env.close()