from init import *
import gym 
import numpy as np

# load q_table
q_table = np.load("/Q_table/q_table_0.npy")
env = gym.make("MountainCar-v0",render_mode = "human")

q_table_size = [20,20]
q_table_segment_size = (env.observation_space.high - env.observation_space.low) / q_table_size


def convert_state(state):
    new_state = (state - env.observation_space.low) // q_table_segment_size
    return tuple(new_state.astype(np.int))


env.reset()
for i in range(TEST_EPISODES):
    done = False
    current_state = convert_state(env.reset()[0])
    while not done:
        action = np.argmax(q_table[current_state])
        next_s = env.step(action)
        done = next_s[2]
        next_real_state = next_s[0]
        if done:
            # finish
            print("finish : ",i)
        next_state = convert_state(next_real_state)
        reward = next_s[1]
        # print(next_real_state)
        # print(reward)
        # current_q_value = q_table[current_state+(action,)]
        # new_q_value = (1 - LEARNING_RATE) * current_q_value + LEARNING_RATE * (reward + DISCOUNT * np.max(q_table[next_state]))
        # q_table[current_state + (action,)] = new_q_value
        current_state = next_state
        

