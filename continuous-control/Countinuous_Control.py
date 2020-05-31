from unityagents import UnityEnvironment
import numpy as np
import torch
from collections import deque
import sys
from ddpg_agent import Agent
import matplotlib.pyplot as plt
import numpy as np

#PLOT FUNCTION
def plot(scores):
    plt.plot(np.arange(1,len(scores)+1), scores)
    plt.title('score plot')
    plt.xlabel('# of episodes')
    plt.ylabel('scores')
    plt.show()


# ENVIRONMENT INFORMATION
env = UnityEnvironment(file_name='Reacher_Linux/Reacher.x86_64')
# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

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


# AGENT INITIALISED
agent = Agent(state_size,action_size,0)


def run(agent, env, num_iterations=2000, save_every = 100, print_every = 1, average_every = 100, update_every = 20):
    average_scores = deque(maxlen=100)
    scores = []
    for i_iteration in range(1, num_iterations+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations
        score = 0
        agent.reset()
        for t in range(700):
            action = agent.act(state)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations
            reward = env_info.rewards
            done = env_info.local_done
            #agent should learn from these experiences
        
            if(t%update_every==0):
                agent.step(state, action, reward[0], next_state, done[0], update=True)
            else:
                agent.step(state, action, reward[0], next_state, done[0],update=False)
                
            state = next_state
            score += reward[0]

            
            if done[0]:
                break
        scores.append(score)
        average_scores.append(score)
                
        # Save the state dicts for target network of the actor and critic both
        if i_iteration%save_every==0 : 
            torch.save(agent.actor_target.state_dict(),'actor.pth')
            torch.save(agent.critic_target.state_dict(),'critic.pth')
        
        
        if i_iteration%print_every==0 :
            i_score = scores[-1]
            print('\rEpisode : {} | Score : {:.2f}'.format(i_iteration,i_score),end='')
            sys.stdout.flush()
            
            if i_iteration%average_every==0 :
                average_score = np.mean(average_scores)
                print(' | Average Score : {:.2f}'.format(average_score))
                
        if(np.mean(average_scores)>=30.0):
            print('Environment Solved! | Average Score: {:.2f}'.format(np.mean(average_scores)))
            break
    env.close()      
    return scores

print()
print('***** TRAINING INITIALISED *****')
scores = run(agent, env)
plot(scores)
print()