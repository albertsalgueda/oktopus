from cmath import isnan
from logging import raiseExceptions
import math
import random
import numpy as np
import pandas as pd
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

    def take_action(self,arm,q_values):
        random_action = []
        if self.current_time < 1:
            print('AI still has no data so no action taken')
        else:
            print(f'AI is increasing budget of campaign {arm}')
            self.act2(arm,q_values)
        b = copy.deepcopy(self.budget_allocation)
        rewards = self.get_reward()
        self.history[self.current_time] = [b,rewards]
        print(f'Current state: {self.budget_allocation} at timestamp {self.current_time}')
        self.allocate_budget()
        self.next_timestamp()
        return rewards

    def act(self,arm,q_values):
        #given a q values, take an action 
        #current action policy => increase choosen arm, decrease least arm
        #TODO --> improve action policy given q_values 
        temp_budget = copy.deepcopy(self.budget_allocation)
        temp_budget[arm] *= 1.005
        q_values = list(q_values)
        decrease = q_values.index(min(q_values))            
        #check that chosen arm is not the minim 
        if decrease != arm:
            temp_budget[decrease] *= 0.995
        else: 
            while decrease == arm:
                decrease = random.randint(0,len(self.campaigns)-1)
            temp_budget[decrease] *= 0.995
        #validate that the budget is corrent before updating it
        if self.validate_budget(temp_budget):
            self.budget_allocation = temp_budget
        else:
            #print(sum(temp_budget.values()))
            #TODO --> find a way out of here man
            while not self.validate_budget(temp_budget):
                decrease = random.randint(0,len(self.campaigns)-1)
                if decrease != arm:
                    temp_budget[decrease] *= 0.995
    
    def act2(self,arm,q_values):
        population = list(range(len(self.campaigns)))
        step = 0.005
        temp_budget = copy.deepcopy(self.budget_allocation)
        temp_budget[arm] += step
        q_values = q_values.tolist()
        #if we have no data, randomly decrease an campaign
        if all(v == 0 for v in q_values):
            dec = random.randint(0,len(self.campaigns)-1)
            if dec != arm:
                temp_budget[dec] -= step
            else:
                while dec == arm:
                    dec = random.randint(0,len(self.campaigns)-1)
                temp_budget[dec] -= step
        #if we have data, take a stochastic approach 
        else:
            norm = [float(i)/sum(q_values) for i in q_values]
            decrease_prob = [1-p for p in norm]
            #print(f'norm is {norm}')
            #print(f'population is {population}')
            dec = int(random.choices(population, weights=decrease_prob, k=1)[0])
            #we could test another action policy where the chosen arm would be also subject to a decrease, 
            # that would result in no action, I'll ignore that option
            if dec != arm:
                #SOLUTION OF BUG 1
                if temp_budget[dec] < 0.005:
                    temp_budget[arm] -= temp_budget[dec]
                    print(f'##### Campaign {dec} was stopped completely ###')
                else:
                    temp_budget[dec] -= step
            else:
                while dec == arm:
                    dec = int(random.choices(population, weights=decrease_prob, k=1)[0])
                #SOLUTION OF BUG 1
                if temp_budget[dec] < 0.005:
                    temp_budget[arm] -= temp_budget[dec]
                    temp_budget[dec] = 0
                    print(f'##### Campaign {dec} was stopped completely ###')
                else:
                    temp_budget[dec] -= step
            print(f'Ai has decreased campaign {dec} given probs {decrease_prob}')
        #validate that the budget is corrent before updating it
        if self.validate_budget(temp_budget):
            #round the budget to avoid RuntimeWarning: invalid value encountered in double_scalars
            for campaign in temp_budget:
                temp_budget[campaign] = round(temp_budget[campaign],4)
            #update budget
            self.budget_allocation = temp_budget
        else:
            raise Exception('Budget is not valid, act2 policy has failed')
            
    @staticmethod        
    def get_state(budget_allocation):
        return tuple(budget_allocation.values())

    def initial_allocation(self):
        for campaign in self.campaigns:
            self.budget_allocation[campaign.id] = round(1/len(self.campaigns),4)
        b = copy.deepcopy(self.budget_allocation)
        self.history[self.current_time] = [b,self.get_reward()]
        self.allocate_budget()

    def allocate_budget(self):
        #turns a distribution into a value
        #campaign budget = current budget * campaign%allocation
        for campaign in self.campaigns:
            campaign.budget = round(self.current_budget*self.budget_allocation[campaign.id],2)
        
    @staticmethod
    def validate_budget(budget_allocation):
        total = 0
        for campaign in budget_allocation.values():
            if campaign > 1: return False
            elif campaign < 0: return False
            else:
                total += campaign
        if total > 0.98 and total < 1:
            return True
        else:
            return False