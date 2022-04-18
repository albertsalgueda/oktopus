import random
import numpy as np
import matplotlib.pyplot as plt
import copy
#import time
from mab import *

class Campaign():
    def __init__(self, id, assigned_budget, spent_budget, conversion_value):
        self.id = id
        self.assigned_budget = assigned_budget
        self.spent_budget = [spent_budget]
        self.conversion_values = [conversion_value]
        # assigned_budget is the daily budget
        # --> we will have to split it later on according the duration of the time steps
        # spent_budget = budget spent within the different time steps
        # --> recall the assigned budget in a time step is not always equal to the spent budget in that time
        # conversion_values = conversion value within the different time steps

    def update(self, last_spent_budget, last_conversion_value):
        self.spent_budget.append(last_spent_budget)
        self.conversion_values.append(last_conversion_value)

    def change_budget(self,increment):
        self.assigned_budget = self.assigned_budget + increment
    

class State(Campaign):
    def __init__(self, total_budget, total_time, campaigns, initial_allocation=0):
        self.total_budget = total_budget
        self.total_time = total_time
        self.campaigns = campaigns
        self.current_time = 0
        self.current_budget = self.total_budget/self.total_time
        self.history = {}
        self.budget_allocation = {}
        self.remaining_budget = total_budget
        # the metric unit of total_time is a time step
        # history is a dictionary that contains all the states
        # --> with key = number of time step
        # --> with value = the budget allocation in that time step
        # budget_allocation = a dictionary that contains budget allocations
        # --> with key = id of the campaign
        # --> with value = the normalized budget for that campaign respect the total budget for that time step

        self.step_size = 0.005
        self.k_arms = len(campaigns)
        self.stopped_campaigns = []
        # step_size = minimum of change in allocation the AI can play with for every campaign

        if initial_allocation == 0:  
            self.initial_allocation()
        else:
            for campaign in campaigns:
                self.budget_allocation[campaign.id] = initial_allocation[campaign.id]
    
    @staticmethod        
    def get_state(budget_allocation):
        return tuple(budget_allocation.values())


    def initial_allocation(self):
        for campaign in self.campaigns:
            self.budget_allocation[campaign.id] = round(1/len(self.campaigns),8)
        b = copy.deepcopy(self.budget_allocation)
        self.history[self.current_time] = [b,self.get_reward()]
        self.allocate_budget()


    def next_timestamp(self):
        if self.remaining_budget <= 0:
            raise Exception('No budget left')
        self.remaining_budget -= self.current_budget
        self.current_time += 1
        self.step_size *= 1.001
        # still to prove that changing the step size performs better
        #--> study in which way is better to change the step size


    def get_reward(self):
        if self.current_time == 0:
            return list(np.zeros(len(self.budget_allocation)))
        rewards = [] 
        for campaign in self.campaigns:
            print(campaign.conversion_values)
            rewards.append(campaign.conversion_values[-1])
        norm = [float(i)/sum(rewards) for i in rewards]
        #print(f'The rewards at time step {self.current_time} is {rewards}')
        return norm


    def take_action(self, campaign_index, q_values):
        if self.current_time < 1:
            print('AI still has no data so no action taken(( ')
        else:
            print(f'AI is increasing budget of campaign {campaign_index} ))')
            self.act2(campaign_index, q_values)
        b = copy.deepcopy(self.budget_allocation)
        rewards = self.get_reward()
        self.history[self.current_time] = [b,rewards]
        print(f'Current state: {self.budget_allocation} at timestamp {self.current_time}')
        self.allocate_budget()
        self.next_timestamp()
        return rewards
        # we have a new budget distribution -->  we want to store it in self.history
        # wait for its rewards, allocate budget and move time forward.


    def act2(self, campaign_index, q_values):
        population = list(range(len(self.campaigns)))
        temp_budget = copy.deepcopy(self.budget_allocation)
        temp_budget[campaign_index] += self.step_size
        q_values = q_values.tolist()
        
        """
        stochastic process to choose and decrease another campaign 
        """
        # SOLUTION TO BUG ID 7 -- read Test OKT-B
        #print(f"---{len(population)-len(self.stopped)}")
        #time.sleep(0.01)
        if len(population) - len(self.stopped_campaigns) == 1:
            temp_budget[campaign_index] = 1
        else:
            """
            if we have no data, randomly decrease a campaign
            """
            if all(v == 0 for v in q_values):
                dec = random.randint(0,len(self.campaigns)-1)
                if dec != campaign_index:
                    temp_budget[dec] -= self.step_size
                else:
                    while dec == campaign_index:
                        dec = random.randint(0,len(self.campaigns)-1)
                    temp_budget[dec] -= self.step_size
                """
            if we have data, take a stochastic approach 
                """
            else:
                norm = [float(i)/sum(q_values) for i in q_values]
                decrease_prob = [1-p for p in norm]
                #print(f'norm is {norm}')
                #print(f'population is {population}')
                dec = int(random.choices(population, weights=decrease_prob, k=1)[0])
                #we could test another action policy where the chosen arm would be also subject to a decrease, 
                # that would result in no action, I'll ignore that option 
                if dec != campaign_index and dec not in self.stopped_campaigns:
                    #TODO SOLUTION OF BUG 1
                    if temp_budget[dec] < self.step_size:
                        temp_budget[campaign_index] -= temp_budget[dec]
                        #temp_budget[arm] -= step
                        temp_budget[dec] = 0
                        print(f'##### Campaign {dec} was stopped completely ###')
                        #TODO delete campaign from the state ( make it ignore it )
                        self.stopped_campaigns.append(dec)
                    else:
                        temp_budget[dec] -= self.step_size
                else:
                    while True:
                        dec = int(random.choices(population, weights=decrease_prob, k=1)[0])
                        if dec == campaign_index:
                            continue
                        if dec in self.stopped_campaigns:
                            continue
                        else:
                            break
                    #SOLUTION OF BUG 1
                    if temp_budget[dec] < self.step_size:
                        temp_budget[campaign_index] -= temp_budget[dec]
                        temp_budget[campaign_index] -= self.step_size
                        temp_budget[dec] = 0
                        print(f'##### Campaign {dec} was stopped completely ###')
                        self.stopped_campaigns.append(dec)
                    else:
                        temp_budget[dec] -= self.step_size
                print(f'Ai has decreased campaign {dec} given probs {decrease_prob}')

        #round the budget to avoid RuntimeWarning: invalid value encountered in double_scalars
        for campaign in temp_budget:
            temp_budget[campaign] = round(temp_budget[campaign],8) # we give it 8 decimals
        """
        validate that the budget is corrent before updating it
        """
        if self.validate_budget(temp_budget):
            #update budget
            self.budget_allocation = temp_budget
        else:
            print(f"Chosen arm: {campaign_index}, Stopped Campaigns: {self.stopped_campaigns}")
            raise Exception(f'Budget is not valid, act2 policy has failed, Failed budget is {temp_budget}')
    
    def allocate_budget(self):
        """
        here we update campaign daily budget 

        turns a distribution into a value
        updates campaign budget 
        campaign budget = current budget * campaign%allocation
        """
        for campaign in self.campaigns:
            # TODO --> think about a way to universalize it.
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
            #it is better that we spend less than more. 
            return True
        else:
            return False
        # the difference between asigned and spent budget has limits


    """
    Changes campaigns ROAS randomly for testing purposes. 


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
    """