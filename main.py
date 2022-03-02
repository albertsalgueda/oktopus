from logging import raiseExceptions
import math
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from itertools import combinations, permutations
import copy

from mab import *

class Campaign():

    def __init__(self,id,budget,spent,impressions,conversions,roi):
        #falta determinar como podemos saber el tiempo que lleva la campaña 
        self.id = id    
        self.budget = budget
        self.spent = spent
        self.impressions = impressions
        self.conversions = conversions
        self.roi = roi

    def update(self,impressions,conversions,roi):
        self.spent += self.budget
        self.impressions += int(impressions)
        self.conversions += int(conversions)
        self.roi = float(roi)

    def change_budget(self,increment):
        #increment debe ser un valor numerico para editar el ( daily budget )
        self.budget = self.budget + increment
    

class State(Campaign):

    def __init__(self,budget,total_time,campaigns):
        self.budget = budget
        self.time = total_time
        self.campaigns = campaigns
        self.current_time = 0
        self.current_budget = self.budget/self.time
        #dictionary that contains all states, where the key is a timestamp
        self.history = {}
        self.budget_allocation = {}
        self.remaining = budget

        self.step = 0.01
        self.max_steps = 5

        self.k_arms = len(campaigns)

        self.initial_allocation()

    def next_timestamp(self):
        self.current_time += 1
        self.remaining-=self.current_budget
        if self.remaining < 0:
            raise Exception('No budget left')
        #reward = self.get_reward()
        #old_state = self.get_state(self.history[self.current_time-1][0])
        #new_state = self.get_state(self.budget_allocation)

    def get_reward(self):
        if self.current_time == 0:
            return list(np.zeros(len(self.budget_allocation)))
        rewards = [] 
        for campaign in self.campaigns:
            rewards.append(campaign.roi*self.current_budget*self.budget_allocation[campaign.id])
        #norm = [float(i)/sum(rewards) for i in rewards]
        #print(f'The rewards at timestamp {self.current_time} is {rewards}')
        return rewards

    def take_action(self,arm):
        random_action = []
        if self.current_time < 1:
            print('AI still has no data so no action taken')
        else:
            print(f'AI is increasing budget of campaign {arm}')
            actions = self.available_actions(self.budget_allocation)
            max = 0
            for budget in actions:
                if budget[arm] > max:
                    random_action=budget
            #update budget
            i = 0
            for campaign in self.campaigns:
                self.budget_allocation[campaign.id] = random_action[i]
                i += 1
        b = copy.deepcopy(self.budget_allocation)
        rewards = self.get_reward()
        self.history[self.current_time] = [b,rewards]
        print(f'Current state: {self.budget_allocation} at timestamp {self.current_time}')
        self.allocate_budget()
        self.next_timestamp()
        return rewards

    @staticmethod        
    def get_state(budget_allocation):
        return tuple(budget_allocation.values())

    def initial_allocation(self):
        for campaign in self.campaigns:
            self.budget_allocation[campaign.id] = 1/len(self.campaigns)
        b = copy.deepcopy(self.budget_allocation)
        self.history[self.current_time] = [b,self.get_reward()]
        self.allocate_budget()

    def allocate_budget(self):
        #turns a distribution into a value
        #campaign budget = current budget * campaign%allocation
        for campaign in self.campaigns:
            campaign.budget = round(self.current_budget*self.budget_allocation[campaign.id],2)

    @staticmethod
    def available_actions(budget_distribution):
        #returns a set of available actions given a particular state
        #action is a tupple of n campaigns lenght
        #EX: if campaigns = 2 then action (x,y) will represent the change on each campaign respectevely
        #EX: for n campaigns action (n1,n2...n)
        #el step es el nivel de cambio permitido step=0.01 por defecto
        #max_step el numero de steps permitidos, si max_step es 5 maximo se puede incrementar un 5%
        # actions = set()
        max_step = 0.04
        step = 0.01

        def validate_budget(budget_allocation):
        #total budget allocation cannot be different 1
            total = 0
            for campaign in budget_allocation:
                if campaign > 1: return False
                elif campaign < 0: return False
                else:
                    total += campaign
            if total > 0.95 and total < 1:
                return True
            else:
                return False

        # max percentage of change from an iteration
        amount_campaigns = len(budget_distribution)

        # we have to get the current budget allocation as a list such that //
        # budget_allocation[0] is budget campaign 1, budget_allocation[1] es budget campaign 2, etc
        # something like this:

        budget_allocation = list(budget_distribution.values())

        # creation of a list that contains all possible values of change that can be applied to each campaign
        possible_budget_reallocation_units = [0]

        # we define it, and now we fulfill it with this loop

        for i in range(int(max_step / step)):
            possible_budget_reallocation_units.append((i + 1) * step)
            possible_budget_reallocation_units.append(-(i + 1) * step)

        # creation of a list that contains all possible paradigms of change regarding all campaigns //
        # note that every possible paradigm of change is a list such that
        # list[0] is change in campaign 1, list[1] is change in campaign 2, etc

        no_change_option = []
        for i in range(amount_campaigns):
            no_change_option.append(0)

        possible_budget_reallocation = [tuple(no_change_option)]

        temp = combinations(possible_budget_reallocation_units, amount_campaigns)
        for combination in list(temp):
            if sum(list(combination)) == 0:
                possible_permutations = list(permutations(list(combination)))
                possible_budget_reallocation.extend(possible_permutations)

        # creation of a list (all available actions than could be reached) that combines our actual distribution
        # with all the possible paradigms of change we obtained with possible_budget_reallocation

        possible_scenarios = []

        for possible_case in possible_budget_reallocation:
            new_distribution = []
            for i in range(len(possible_case)):
                new_distribution.append(budget_allocation[i] * possible_case[i])
            if validate_budget(new_distribution):
                possible_scenarios.append(new_distribution)

        for possible_case in possible_budget_reallocation:
            case = []
            for i in range(len(possible_case)):
                num = budget_allocation[i] + possible_case[i]
                case.append(round(num, 4))
            possible_scenarios.append(case)

        actual_cases_checked = [case for case in possible_scenarios if validate_budget(case)]

        return actual_cases_checked
        
    @staticmethod
    def validate_budgets(budget_allocation):
        #total budget allocation cannot be different 1
            total = 0
            for campaign in budget_allocation:
                total += campaign
            if total != 1: return False
            return True

class RandomAgent(object):

  def __init__(self, env, max_iterations):
    self.env = env
    self.iterations = max_iterations

    self.q_values = np.zeros(self.env.k_arms)
    self.arm_counts = np.ones(self.env.k_arms)
    self.arm_rewards = np.zeros(self.env.k_arms)

    self.rewards = [0.0]
    self.cum_rewards = [0.0]

  def act(self):
    count = 0
    old_estimate = 0.0
    for i in range(self.iterations):
        arm = np.random.choice(self.env.k_arms)
        reward = self.env.take_action(arm)
        print(f'The rewards at timestamp {self.env.current_time} is {reward}')
        self.arm_counts[arm] += 1
        for arm in range(self.env.k_arms):
            self.arm_rewards[arm] += reward[arm]
            self.q_values[arm] = self.q_values[arm] + (1/self.arm_counts[arm]) * (reward[arm] - self.q_values[arm])
        self.rewards.append(sum(reward))
        count += 1
        current_estimate = old_estimate + (1/count)*(sum(reward)-old_estimate)
        self.cum_rewards.append(current_estimate)
        old_estimate = current_estimate

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
    for i in range(self.iterations):
        arm = np.random.choice(self.env.k_arms) if np.random.random() < self.epsilon else np.argmax(self.q_values)
        reward = self.env.take_action(arm)
        print(f'The rewards at timestamp {self.env.current_time} is {reward}')
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
    for i in range(self.iterations):
        self.action_probas = np.exp(self.q_values/self.tau) / np.sum(np.exp(self.q_values/self.tau))
        arm = np.random.choice(self.env.k_arms, p=self.action_probas)
        reward = self.env.take_action(arm)
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
    for i in range(self.iterations):
        arm = np.argmax(self.q_values)
        reward = self.env.take_action(arm)
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
    for i in range(0, self.iterations):
        if i < len(self.q_values):
            arm = i
        else:
            U = self.c * np.sqrt(np.log(i) / self.arm_counts)
            arm = np.argmax(self.q_values + U)

        reward = self.env.take_action(arm)
        print(f'The rewards at timestamp {self.env.current_time} is {reward}')
        self.arm_counts[arm] += 1
        for arm in range(self.env.k_arms):
            self.arm_rewards[arm] += reward[arm]
            self.q_values[arm] = self.q_values[arm] + (1/self.arm_counts[arm]) * (reward[arm] - self.q_values[arm])
        
        self.rewards.append(sum(reward))
        count += 1
        current_estimate = old_estimate + (1/count)*(sum(reward)-old_estimate)
        self.cum_rewards.append(current_estimate)
        old_estimate = current_estimate

    return {"arm_counts" : self.arm_counts, "rewards": self.rewards, "cum_rewards": self.cum_rewards}
