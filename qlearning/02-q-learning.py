import gym
import numpy as np
import matplotlib.pyplot as plt
import pyglet 

env = gym.make("MountainCar-v0")


LEARNING_RATE = 0.1
DISCOUNT = 0.95  # actually a weight defines how important are the future actions (or reward) over the current actions (reward)
EPISODES = 25000

SHOW_EVERY = 500



DISCRETE_OS_SIZE = [40] * len(env.observation_space.high)
discrete_os_win_size = (
    env.observation_space.high - env.observation_space.low
) / DISCRETE_OS_SIZE


epsilon = 1
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES//2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)


q_table = np.random.uniform(
    low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n])
)

episode_rewards = []    # reward during each episode
aggr_episode_rewards = {
    "ep" : [],
    "avg": [],
    "min": [],
    "max": []
}


# **Function to give us the Discrete value for given continous state**


def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int32))


for episode in range(EPISODES):
    ep_reward = 0
    if episode % 10 == 0:
        np.save(f"qtables/{episode}-qtable.npy", q_table)
    
    if episode % SHOW_EVERY == 0:
        render = True
        print(episode)

    else:
        render = False

    discrete_state = get_discrete_state(
        env.reset()
    )  # with this discrete state we can start to take actions and then slowly generates the new q-values1
    done = False
    while not done:

        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, done, _ = env.step(action)
        ep_reward += reward
        new_discrete_state = get_discrete_state(new_state)
        if render:
            env.render()
            
        # If simulation did not end yet after last step - update Q table
        if not done:
            # Maximum possible Q value in next step (for new state)
            max_future_q = np.max(q_table[new_discrete_state])
            
            # Current Q value (for current state and performed action)
            current_q = q_table[discrete_state + (action,)]

            # And here's our equation for a new Q value for current state and action
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (
                reward + DISCOUNT * max_future_q
            )
            
            
            # Update Q table with new Q value
            q_table[discrete_state + (action,)] = new_q

        # Simulation ended (for any reason) - if goal position is achieved - update Q value with reward directly
        elif new_state[0] >= env.goal_position:
            print(f"We made it on episode {episode}")
            q_table[discrete_state + (action,)] = 0
        discrete_state = new_discrete_state

    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value
        
    episode_rewards.append(ep_reward)
    
    if not episode % SHOW_EVERY:
        average_reward = sum(
            episode_rewards[-SHOW_EVERY:])/len(episode_rewards[-SHOW_EVERY:])
        aggr_episode_rewards["ep"].append(episode)
        aggr_episode_rewards["avg"].append(average_reward)
        aggr_episode_rewards["min"].append(min(episode_rewards[-SHOW_EVERY:]))
        aggr_episode_rewards["max"].append(max(episode_rewards[-SHOW_EVERY:]))
        
        print(
            f'Episode: {episode:>5d}, average reward: {average_reward:>4.1f}, current epsilon: {epsilon:>1.2f}')
        

env.close()
plt.plot(aggr_episode_rewards["ep"], aggr_episode_rewards["avg"], label="avg")
plt.plot(aggr_episode_rewards["ep"], aggr_episode_rewards["min"], label="min")
plt.plot(aggr_episode_rewards["ep"], aggr_episode_rewards["max"], label="max")
plt.legend(loc=4)
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.show()
