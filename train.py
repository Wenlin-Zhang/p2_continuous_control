from unityagents import UnityEnvironment
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import time
from ddpg_agent import Agent


env = UnityEnvironment(file_name='./Reacher_Linux/Reacher.x86_64')

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

# Create an agent
agent = Agent(state_size=state_size, action_size=action_size, random_seed=1)

def ddpg(n_episodes=2000, max_t=1000):
    scores_deque = deque(maxlen=print_every)
    mean_scores = []
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        agent.reset()
        scores = np.zeros(num_agents)
        start_time = time.time()
        for t in range(max_t):
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            # put every actors' experiences to the replay buffer
            for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                agent.step(state, action, reward, next_state, done, t)
            states = next_states
            scores += rewards
            if np.any(dones):
                break
        mean_score = np.mean(scores)
        max_score = np.max(scores)
        min_score = np.min(scores)
        mean_scores.append(mean_score)
        scores_deque.append(mean_score)
        moving_avg_score = np.mean(scores_deque)
        
        duration = time.time() - start_time
        print('\rEpisode {}({:.2f} sec)\tMin: {:.2f}\tMax: {:.2f}\tAvg: {:.2f}\t Moving Avg: {:.2f}'.format(i_episode, duration, min_score, max_score, mean_score, moving_avg_score))
              
        if i_episode >= 100 and moving_avg_score >= solved_score:
            print('\nEnvironment SOLVED in {} episodes!\tMoving Average Score: {:.1f} over last 100 episodes'.format(\
                                    i_episode-100, moving_avg_score))            
        torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
        torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
    return mean_scores

scores = ddpg()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig('./train.png')




