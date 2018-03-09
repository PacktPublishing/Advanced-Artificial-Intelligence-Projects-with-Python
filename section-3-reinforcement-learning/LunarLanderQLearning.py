
# This code is inspired by Matthew Chan's solution to the Cart-Pole problem:
# https://gist.github.com/tuzzer/90701191b50c2e7bafca167858fcb234

import gym

env = gym.make('LunarLander-v2')

# Nop, fire left engine, main engine, right engine
ACTIONS = env.action_space.n

# Landing pad is always at coordinates (0,0). Coordinates are the first two numbers in state vector.
# Reward for moving from the top of the screen to landing pad and zero speed is about 100..140 points.
# If lander moves away from landing pad it loses reward back. Episode finishes if the lander crashes or
# comes to rest, receiving additional -100 or +100 points. Each leg ground contact is +10. Firing main
# engine is -0.3 points each frame. Solved is 200 points.

import numpy as np
import random

def discretize_state(state):
    dstate = list(state[:5])
    dstate[0] = int(0.5*(state[0]+0.7)*10/2.0) # pos x
    dstate[1] = int(0.5*(state[1]+0.5)*10/2.0) # pos y
    dstate[2] = int(0.5*(state[2]+1.5)*10/3.0) # vel x
    dstate[3] = int(0.5*(state[3]+2)*10/3.0) # vel y
    dstate[4] = int(0.5*(state[4]+3.14159)*10/(2*3.14159)) # angle
    if dstate[0] >= 5: dstate[0] = 4
    if dstate[1] >= 5: dstate[1] = 4
    if dstate[2] >= 5: dstate[2] = 4
    if dstate[3] >= 5: dstate[3] = 4
    if dstate[4] >= 5: dstate[4] = 4
    if dstate[0] < 0: dstate[0] = 0
    if dstate[1] < 0: dstate[1] = 0
    if dstate[2] < 0: dstate[2] = 0
    if dstate[3] < 0: dstate[3] = 0
    if dstate[4] < 0: dstate[4] = 0
    return tuple(dstate)

def run(num_episodes, alpha, gamma, explore_mult):
    max_rewards = []
    last_reward = []
    qtable = np.subtract(np.zeros((5, 5, 5, 5, 5, ACTIONS)), 100) # start all rewards at -100
    explore_rate = 1.0
    for episode in range(num_episodes):
        s = env.reset()
        state = discretize_state(s)
        
        for step in range(10000):

            # select action
            if random.random() < explore_rate:
                action = random.choice(range(ACTIONS))
            else:
                action = np.argmax(qtable[state])

            (new_s, reward, done, _) = env.step(action)
            new_state = discretize_state(new_s)

            # update Q
            best_future_q = np.amax(qtable[new_state]) # returns best possible reward from next state
            prior_val = qtable[state + (action,)]
            qtable[state + (action,)] = (1.0-alpha)*prior_val + alpha*(reward + gamma * best_future_q)
            state = new_state
            
            if done or step == 9999:
                last_reward.append(reward)
                break
        
        if explore_rate > 0.01:
            explore_rate *= explore_mult    
        max_rewards.append(np.amax(qtable))
        
    return (max_rewards, last_reward[-50:], qtable) # return rewards from last 50 episodes


num_episodes = 100
for alpha in [0.05, 0.10, 0.15]:
    for gamma in [0.85, 0.90, 0.95]:
        (max_rewards, last_reward, _) = run(num_episodes=num_episodes, alpha=alpha, gamma=gamma, explore_mult=0.995)
        print("alpha = %.2f, gamma = %.2f, mean last 50 outcomes = %.2f, q max: %.2f, q mean: %.2f" % (alpha, gamma, np.mean(last_reward), np.max(max_rewards), np.mean(max_rewards)))

(max_rewards, last_reward, qtable) = run(num_episodes=200, alpha=0.1, gamma=0.95, explore_mult=0.995)
print("mean last 50 outcomes = %.2f, q max: %.2f, q mean: %.2f" % (np.mean(last_reward), np.max(max_rewards), np.mean(max_rewards)))
np.save('qtable.npy', qtable)

# Use best qtable to play the game (no learning anymore)
import gym
import numpy as np
env = gym.make('LunarLander-v2')
qtable = np.load('qtable.npy')
for i in range(100):
    s = env.reset()
    state = discretize_state(s)
    for step in range(10000):
        env.render()

        # select action
        action = np.argmax(qtable[state])

        (new_s, reward, done, _) = env.step(action)
        new_state = discretize_state(new_s)

        if done or step == 9999:
            break

        state = new_state

