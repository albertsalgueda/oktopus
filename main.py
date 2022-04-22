from cmath import isnan
from dataclasses import dataclass
import random
from xml.sax import parseString
import numpy as np
import matplotlib.pyplot as plt
import copy


#import time

from mab import *

@dataclass
class Campaign():
    def __init__(self,id,budget,spent,impressions,conversions,roi):
        #falta determinar como podemos saber el tiempo que lleva la campaña 
        self.id = id    
        self.budget = budget #daily budget
        self.spent = [spent]
        self.impressions = [impressions]
        self.conversions = [conversions]
        self.roi = roi

class State(Campaign):
    def __init__(self,budget,total_time,campaigns,initial_allocation=0):
        self.budget = budget
        self.time = total_time
        self.campaigns = campaigns
        self.current_time = 0
        self.current_budget = self.budget/self.time
        #dictionary that contains all states, where the key is a timestamp
        self.history = {}
        self.budget_allocation = {}
        self.remaining = budget

        self.step = 0.005

        self.k_arms = len(campaigns)

        self.stopped = []
        if initial_allocation == 0: 
            self.initial_allocation()
        else:
            for campaign in campaigns:
                self.budget_allocation[campaign.id] = initial_allocation[campaign.id]

    def update(self,distribution):
        i = 0
        for campaign in self.budget_allocation:
            self.budget_allocation[campaign] = distribution[i]
            i += 1
        assert self.validate_budget(self.budget_allocation), 'Budget distribution is incorrect'
        self.allocate_budget()
        self.next_step()

    def next_step(self):
        self.current_time += 1
        self.remaining-=self.current_budget
        if self.remaining <= 0:
            raise Exception('No budget left')
        #increase the capacity of an agent to take significant budget decisions
        self.step *= 1.001

    def overspent_case(self):
        # mandar notificacion al usuario
        # si no responde pararlas  
        for campaign in self.campaigns:
            campaign.budget = 0

    def get_reward(self):
        if self.current_time == 0:
            return list(np.zeros(len(self.budget_allocation)))
        rewards = [] 
        for campaign in self.campaigns:
            rewards.append(campaign.roi*self.current_budget*self.budget_allocation[campaign.id])
        norm = [float(i)/sum(rewards) for i in rewards]
        print(f'The rewards at timestamp {self.current_time} is {rewards}')
        return norm

    def take_action(self,arm,dec):
        if self.current_time < 1:
            pass
        else: 
            print(f'AI is increasing budget of campaign {arm}')
            self.act3(arm,dec)
        b = copy.deepcopy(self.budget_allocation)
        rewards = self.get_reward()
        self.history[self.current_time] = [b,rewards]
        print(f'Current state: {self.budget_allocation} at timestamp {self.current_time}')
        self.allocate_budget()
        self.next_step()
        return rewards

    def act3(self,arm,dec):
        temp_budget = copy.deepcopy(self.budget_allocation)
        temp_budget[arm] += self.step
        if dec not in self.stopped:
            temp_budget[dec] -= self.step
            if temp_budget[dec] < 0: 
                temp_budget[dec] = 0
                self.stopped.append(dec)
        else:
            a = True
            while a:
                dec = random.randint(0,len(self.campaigns)-1)
                if dec != arm:
                    if dec not in self.stopped:
                        temp_budget[dec] -= self.step
                        print(f'Ai has decreased campaign {dec}')
                    a = False
           
        for campaign in temp_budget:
            temp_budget[campaign] = round(temp_budget[campaign],8)
        #validate that the budget is corrent before updating it
        if self.validate_budget(temp_budget):
            self.budget_allocation = temp_budget
        else:
            print(f"Chosen arm: {arm}, Stopped Campaigns: {self.stopped}")
            raise Exception(f'Budget is not valid, act3 policy has failed, Failed budget is {temp_budget}')

    @staticmethod        
    def get_state(budget_allocation):
        return tuple(budget_allocation.values())

    def initial_allocation(self):
        for campaign in self.campaigns:
            self.budget_allocation[campaign.id] = round(1/len(self.campaigns),8)
        b = copy.deepcopy(self.budget_allocation)
        self.history[self.current_time] = [b,self.get_reward()]
        self.allocate_budget()

    def allocate_budget(self):
        #turns a distribution into a value
        #campaign budget = current budget * campaign%allocation
        for campaign in self.campaigns:
            campaign.budget = round(self.current_budget*self.budget_allocation[campaign.id],8)
        
    @staticmethod
    def validate_budget(budget_allocation):
        total = 0
        for campaign in budget_allocation.values():
            if campaign > 1: return False
            elif campaign < 0: return False
            else:
                total += campaign
        total = round(total,4)
        if total > 0.95 and total <= 1.025:
            return True
        else: return False
    
    def dynamic(self):
        for campaign in self.campaigns:
            a = random.randint(0,2)
            if a == 0:
                campaign.roi *= 1.05
            elif a == 1:
                if campaign.roi > 1:
                    campaign.roi *= 0.95
                elif campaign.roi > 0:
                    campaign.roi *= 0.99
                else:
                    campaign.roi *= 1.01
            else: 
                return self
    
                
