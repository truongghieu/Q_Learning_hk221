import gym
import numpy as np
from init import *
from matplotlib import pyplot as plt

env = gym.make("MountainCar-v0")
env.reset()
# observation 
print(env.observation_space.low)
print(env.observation_space.high)
# action 
# print(env.action_space.n)
# state
# print(env.state)
#  gym render window
# env.render()



# convert state to discrete


q_table_size = [20,20]
q_table_segment_size = (env.observation_space.high - env.observation_space.low) / q_table_size
q_table = np.random.uniform(low=-2, high=0, size=(q_table_size + [env.action_space.n]))


def convert_state(state):
    new_state = (state - env.observation_space.low) // q_table_segment_size
    return tuple(new_state.astype(np.int))

# array to store reward for each learning rate 
reward_list = []
action_list = []

#  for another LEARNING_RATE
for i in range(5):
    tempActionList = []
    tempRewardList = []
    q_table = np.random.uniform(low=-2, high=0, size=(q_table_size + [env.action_space.n]))
    DISCOUNT += 0.1
    for eps in range(EPISODES):
        tempAction = 0
        tempReward = 0
        lastVelocity = 0
        done = False
        current_state = convert_state(env.reset()[0])
        while not done:
            action = np.argmax(q_table[current_state])
            next_s = env.step(action)
            done = next_s[2]
            next_real_state = next_s[0]
            lastVelocity = next_real_state[1]
            if done:
                # finish
                print("DISCOUNT : "+ str(round(DISCOUNT,2))  +" Episodes : ",eps)
                tempActionList.append(tempAction)
                tempRewardList.append(tempReward)
                
            next_state = convert_state(next_real_state)
            # calculate reward
            if(abs(lastVelocity - next_real_state[1]) > 0 ):
                reward = next_s[1] + MORE_REWARD
            else:
                reward = next_s[1]
            # print(next_real_state)
            # print(reward)
            current_q_value = q_table[current_state+(action,)]
            new_q_value = (1 - LEARNING_RATE) * current_q_value + LEARNING_RATE * (reward + DISCOUNT * np.max(q_table[next_state]))
            q_table[current_state + (action,)] = new_q_value
            current_state = next_state
            tempAction += 1
            tempReward += reward
    np.save("Q_table/q_table_"+ str(i)+".npy",q_table)
    reward_list.append(tempRewardList)
    action_list.append(tempActionList)



# save Q table 
# draw graph with name of LEARNING_RATE
for i in range(5):
    plt.plot(reward_list[i],label = "D : "+ str(round(0.1*(i+5),2)))
plt.xlabel("Episode")
plt.ylabel("Reward")
# maxReward = max(rewardEachEps)
plt.title("MountainCar-v0")
plt.legend()
# show max reward in graph
# plt.annotate("Max "+str(maxReward),xy=(rewardEachEps.index(maxReward),maxReward),xytext=(rewardEachEps.index(maxReward),maxReward+10),arrowprops=dict(facecolor='black', shrink=0.05))
# save graph full size 
plt.savefig("fig_discount/Reward.png",bbox_inches='tight')

# plt.savefig("rewardEachEps.png")
# del rewardEachEps
plt.clf()
for i in range(5):
    plt.plot(action_list[i],label = "D : "+ str(round(0.1*(i+5),2)))
# minAction = min(numberActionEachEps)
# show min action in graph
# plt.annotate("Min "+str(minAction),xy=(numberActionEachEps.index(minAction),minAction),xytext=(numberActionEachEps.index(minAction),minAction+10),arrowprops=dict(facecolor='black', shrink=0.05))
plt.xlabel("Episode")
plt.ylabel("Number Action")
plt.legend()
plt.savefig("fig_discount/Action.png",bbox_inches='tight')

# save number action each learning rate 
minAct = []
for i in range(5):
    minAct.append(min(action_list[i]))
f = open("fig_discount/minAction.txt","w")
f.write(str(minAct))
f.close()
plt.clf()

for i in range(5):
    plt.plot(action_list[i])
    plt.xlabel("Episode")
    plt.ylabel("Number Action")
    plt.title("MountainCar-v0 DIS : "+ str(round(0.1*(i+5),2)))
    plt.savefig("fig_discount/Action_"+str(i)+".png",bbox_inches='tight')
    plt.clf()
    plt.plot(reward_list[i])
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("MountainCar-v0 DIS : "+ str(round(0.1*(i+5),2)))
    plt.savefig("fig_discount/Reward_"+str(i)+".png",bbox_inches='tight')
    plt.clf()