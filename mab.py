from main import *
import numpy as np

class ThompsonAgent(object):

    def __init__(self, env, c, max_iterations):
        self.env = env
        self.c = c
        self.iterations = max_iterations
    def act(self):
      pass

class SimulationAgent(object):

  def __init__(self, env, initial_q, initial_visits):
    self.env = env                                                    # pass State() instance
    self.initial_q = initial_q                                        # Initial Q_values [ How optimistic do you want to be? ]
    self.initial_visits = initial_visits                              # Initial visits 
    ### memory... 
    self.q_values = np.ones(self.env.k_arms) * self.initial_q
    self.arm_counts = np.ones(self.env.k_arms) * self.initial_visits
    self.arm_rewards = np.zeros(self.env.k_arms)

  def act(self):
    """
    implement optimized optimistic algorithm 
    """
    ### we choose an arm. 
    arm = np.argmax(self.q_values)
    print(self.q_values)
    ### we get rewards from the environment  
    reward = self.env.take_action(arm,self.q_values)
    print(f'The rewards at timestamp {self.env.current_time} is {reward}')
    #we sum 1 to arm counts
    self.arm_counts[arm] = self.arm_counts[arm] + 1

    #assign rewards for all arms
    for arm in range(self.env.k_arms):
      #update data 
      self.arm_rewards[arm] = self.arm_rewards[arm] + reward[arm]
      self.q_values[arm] = self.q_values[arm] + (1/self.arm_counts[arm]) * (reward[arm] - self.q_values[arm])

class AI(object):

  def __init__(self, env, tau=1):
    self.env = env
    self.tau = tau
    """
    For high tau ( temperature ), all actions have nearly same probability.
    The lower the temperature,  the more expected rewards affect the probability, the probability of the action with the highest expected reward tends to 1.
    """
    self.action_probas = np.zeros(self.env.k_arms)
    self.q_values = np.zeros(self.env.k_arms)
    self.arm_counts = np.ones(self.env.k_arms)
    self.arm_rewards = np.zeros(self.env.k_arms)
    
    self.rewards = [0.0]
    self.cum_rewards = [0.0]  

  def act(self):
    self.action_probas = np.exp(self.q_values/self.tau) / np.sum(np.exp(self.q_values/self.tau))
    arm = np.random.choice(self.env.k_arms, p=self.action_probas)
    reward = self.env.take_action(arm,self.q_values)
    print(f'The rewards at timestamp {self.env.current_time} is {reward}')
    #sum one to the arm that was choosen 
    self.arm_counts[arm] = self.arm_counts[arm] + 1
    #assign rewards for all arms
    for arm in range(self.env.k_arms):
      self.arm_rewards[arm] += reward[arm]
      self.q_values[arm] = self.q_values[arm] + (1/self.arm_counts[arm]) * (reward[arm] - self.q_values[arm])

    self.rewards.append(sum(reward))
    self.cum_rewards.append(sum(self.rewards) / len(self.rewards))

    return {"arm_counts": self.arm_counts, "rewards": self.rewards, "cum_rewards": self.cum_rewards}


class EpsilonGreedyAgent(object):

  def __init__(self, env, epsilon, decay_rate, decay_interval,max_iterations):
    self.env = env
    self.iterations = max_iterations
    self.epsilon = epsilon
    self.decay_rate = decay_rate
    self.decay_interval = decay_interval

    self.q_values = np.zeros(self.env.k_arms)
    self.arm_counts = np.ones(self.env.k_arms)
    self.arm_rewards = np.zeros(self.env.k_arms)

    self.rewards = [0.0]
    self.cum_rewards = [0.0]

  def act(self):
    count = 0
    old_estimate = 0.0
    for i in range(self.iterations-1):
        arm = np.random.choice(self.env.k_arms) if np.random.random() < self.epsilon else np.argmax(self.q_values)
        reward = self.env.take_action(arm,self.q_values)
        print(f'The rewards at timestamp {self.env.current_time} is {reward}')
        print(f'The remaining budget at timestamp {self.env.current_time} is {self.env.remaining}')
        self.arm_counts[arm] += 1
        for arm in range(self.env.k_arms):
            self.arm_rewards[arm] += reward[arm]
            self.q_values[arm] = self.q_values[arm] + (1/self.arm_counts[arm]) * (reward[arm] - self.q_values[arm])
        self.rewards.append(sum(reward))
        count += 1
        current_estimate = old_estimate + (1/count)*(sum(reward)-old_estimate)
        self.cum_rewards.append(current_estimate)
        old_estimate = current_estimate

        if i % self.decay_interval == 0:
            self.epsilon = self.epsilon * self.decay_rate

        #DYNAMIC TESTING 
        self.env.dynamic()

    return {"arm_counts": self.arm_counts, "rewards": self.rewards, "cum_rewards": self.cum_rewards}

class SoftmaxExplorationAgent(object):

  def __init__(self, env, max_iterations=500, tau=0.5):
    self.env = env
    self.iterations = max_iterations
    self.tau = tau

    self.action_probas = np.zeros(self.env.k_arms)
    self.q_values = np.zeros(self.env.k_arms)
    self.arm_counts = np.ones(self.env.k_arms)
    self.arm_rewards = np.zeros(self.env.k_arms)
    
    self.rewards = [0.0]
    self.cum_rewards = [0.0]  

  def act(self):
    count = 0
    old_estimate = 0.0
    for i in range(self.iterations-1):
        self.action_probas = np.exp(self.q_values/self.tau) / np.sum(np.exp(self.q_values/self.tau))
        print(self.action_probas)
        arm = np.random.choice(self.env.k_arms, p=self.action_probas)
        reward = self.env.take_action(arm,self.q_values)
        print(f'The rewards at timestamp {self.env.current_time} is {reward}')
        #sum one to the arm that was choosen 
        self.arm_counts[arm] = self.arm_counts[arm] + 1
        #assign rewards for all arms
        for arm in range(self.env.k_arms):
            self.arm_rewards[arm] += reward[arm]
            self.q_values[arm] = self.q_values[arm] + (1/self.arm_counts[arm]) * (reward[arm] - self.q_values[arm])

        self.rewards.append(sum(reward))
        self.cum_rewards.append(sum(self.rewards) / len(self.rewards))
        #DYNAMIC TESTING 
        self.env.dynamic()

    return {"arm_counts": self.arm_counts, "rewards": self.rewards, "cum_rewards": self.cum_rewards}

class OptimisticAgent(object):

  def __init__(self, env, initial_q, initial_visits, max_iterations):
    self.env = env
    self.iterations = max_iterations
    self.initial_q = initial_q
    self.initial_visits = initial_visits

    self.q_values = np.ones(self.env.k_arms) * self.initial_q
    self.arm_counts = np.ones(self.env.k_arms) * self.initial_visits
    self.arm_rewards = np.zeros(self.env.k_arms)
    
    self.rewards = [0.0]
    self.cum_rewards = [0.0]  

  def act(self):
    count = 0
    old_estimate = 0.0
    for i in range(self.iterations-1):
        arm = np.argmax(self.q_values)
        reward = self.env.take_action(arm,self.q_values)
        print(f'The rewards at timestamp {self.env.current_time} is {reward}')
        #sum one to the arm that was choosen 
        self.arm_counts[arm] = self.arm_counts[arm] + 1
        #assign rewards for all arms
        for arm in range(self.env.k_arms):
            self.arm_rewards[arm] = self.arm_rewards[arm] + reward[arm]
            self.q_values[arm] = self.q_values[arm] + (1/self.arm_counts[arm]) * (reward[arm] - self.q_values[arm])
        #print(self.q_values)
        self.rewards.append(sum(reward))
        count += 1
        current_estimate = old_estimate + (1/count)*(sum(reward)-old_estimate)
        self.cum_rewards.append(current_estimate)
        old_estimate = current_estimate
        #DYNAMIC TESTING 
        self.env.dynamic()

    return {"arm_counts": self.arm_counts, "rewards": self.rewards, "cum_rewards": self.cum_rewards}

class UCBAgent(object):

  def __init__(self, env, c, max_iterations):
    self.env = env
    self.c = c
    self.iterations = max_iterations

    self.q_values = np.zeros(self.env.k_arms, dtype=np.float32)
    self.arm_counts = np.ones(self.env.k_arms, dtype=np.int)
    self.arm_rewards = np.zeros(self.env.k_arms, dtype=np.float32)

    self.rewards = [0.0]
    self.cum_rewards = [0.0]

  def act(self):
    count = 0
    old_estimate = 0.0
    for i in range(0, self.iterations-1):
        if i < len(self.q_values):
            arm = i
        else:
            U = self.c * np.sqrt(np.log(i) / self.arm_counts)
            arm = np.argmax(self.q_values + U)

        reward = self.env.take_action(arm,self.q_values)
        print(f'The rewards at timestamp {self.env.current_time} is {reward}')
        #print(f'Current q values are: {self.q_values}')
        self.arm_counts[arm] += 1
        for arm in range(self.env.k_arms):
            self.arm_rewards[arm] += reward[arm]
            self.q_values[arm] = self.q_values[arm] + (1/self.arm_counts[arm]) * (reward[arm] - self.q_values[arm])
        
        self.rewards.append(sum(reward))
        count += 1
        current_estimate = old_estimate + (1/count)*(sum(reward)-old_estimate)
        self.cum_rewards.append(current_estimate)
        old_estimate = current_estimate
        #DYNAMIC TESTING 
        self.env.dynamic()

    return {"arm_counts" : self.arm_counts, "rewards": self.rewards, "cum_rewards": self.cum_rewards}


