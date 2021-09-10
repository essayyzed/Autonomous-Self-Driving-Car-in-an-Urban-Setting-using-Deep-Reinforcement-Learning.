import gym
import time
import numpy as np

env = gym.make("MountainCar-v0")


LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 2500

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (
    env.observation_space.high - env.observation_space.low
) / DISCRETE_OS_SIZE


q_table = np.random.uniform(
    low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n])
)


def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int64))


discrete_state = get_discrete_state(env.reset())


done = False

while  not done:
    action = np.argmax(q_table[discrete_state])    # go right 
    new_state, reward, done, _ = env.step(action)
    new_discrete_state = get_discrete_state(new_state)
    env.render()
    if not done:
        max_future_q = np.max(q_table[new_discrete_state])

    
env.close()

