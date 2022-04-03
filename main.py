"""
import dependencies. 
"""
import random
import numpy as np
import matplotlib.pyplot as plt
import copy
#import time
from mab import * # import all our algorithms from mab.py

class Campaign():
    def __init__(self,id,budget,spent,conversion_value,roi):
        self.id = id    
        self.budget = budget # daily budget
        self.spent = spent #spent across time steps
        self.conversion_value = [].append(conversion_value) #conversion value across time steps  

    def update(self,new_value):
        """
        updates self.converion_value with current time step data 
        """
        self.conversion_value.append(new_value)

    def change_budget(self,increment):
        #increment debe ser un valor numerico para editar el ( daily budget )
        self.budget = self.budget + increment
    

class State(Campaign):
    def __init__(self,budget,total_time,campaigns,initial_allocation=0):
        """
        Let's start declaring all the variables of class State
        """
        self.budget = budget # we define budget constraint 
        self.time = total_time # we define time constraint
        self.campaigns = campaigns #w we define campaigns
        self.current_time = 0 # we initialize current time step at 0
        self.current_budget = self.budget/self.time #we initialize current budget proportionally 
        self.history = {} # dictionary that contains all states, where the key is a timestamp
        self.budget_allocation = {} # a dictionary that contains budget allocation {0:.25,1:.25,2:.25,3:.25}
        self.remaining = budget # budget 

        self.step = 0.005 # level of freedom for the AI to move 

        self.k_arms = len(campaigns) # numero de arms = number of campaigns 

        self.stopped = [] # number of stopped campaigns 
        """
        if there is no pre-defined budget allocation, we call function to make standardized distribution. 
        """
        if initial_allocation == 0:  
            self.initial_allocation()
        else:
            """
            if our user provides us with a budget allocation, we store it. 
            """
            for campaign in campaigns:
                self.budget_allocation[campaign.id] = initial_allocation[campaign.id]
    
    @staticmethod        
    def get_state(budget_allocation):
        """
        returns a tuple (0.33,0.33,0.33) at current time step. 
        """
        return tuple(budget_allocation.values())

    def initial_allocation(self):
        """
        We distribute budget proportionally. 
        """
        for campaign in self.campaigns:
            self.budget_allocation[campaign.id] = round(1/len(self.campaigns),8)
        b = copy.deepcopy(self.budget_allocation)
        self.history[self.current_time] = [b,self.get_reward()]
        self.allocate_budget()

    def next_timestamp(self):
        """
        move time forward xD  
        """
        self.remaining-=self.spent[self.current_time] # update reimaining budget
        self.current_time += 1 # move time 
        # if there is no budget left... 
        if self.remaining <= 0:
            raise Exception('No budget left')
        #increase the capacity of an agent to take significant budget decisions
        self.step *= 1.001 # we give 0.1% more freedom in every step, the kid gets older ;)

    def get_reward(self):
        """
        returns a normalized distribution for rewards, based on purchase conversion value. 
        """
        if self.current_time == 0:
            # no rewards for the first time step :( 
            return list(np.zeros(len(self.budget_allocation)))
        rewards = [] 
        for campaign in self.campaigns:
            rewards.append(campaign.conversion_value[self.current_time-1])
        norm = [float(i)/sum(rewards) for i in rewards] #normalize rewards 
        #print(f'The rewards at timestamp {self.current_time} is {rewards}')
        return norm

    def take_action(self,arm,q_values):
        """
        takes an action given chosen arm, q_values 
        """
        #if it is timestep 0, just wait. 
        if self.current_time < 1:
            print('AI still has no data so no action taken(( ')
        else:
            print(f'AI is increasing budget of campaign {arm} ))')
            self.act2(arm,q_values) # calls act2 action policy 2 ( 1 was shit )
        """
        we have a new baby, a new budget distribution. 
        now we want to store the new baby in self.history 
        wait for its rewards, 
        allocate budget and move time forward. 
        """
        b = copy.deepcopy(self.budget_allocation)
        rewards = self.get_reward()
        self.history[self.current_time] = [b,rewards]
        print(f'Current state: {self.budget_allocation} at timestamp {self.current_time}')
        self.allocate_budget()
        self.next_timestamp()
        return rewards

    def act2(self,arm,q_values):
        """
        Given a chosen campaign, change budget distribution. 
        My policy is to increase with step % campaigns[arm]
        and via some stochastic process based on q_values
        determine the decreasing arm. 
        Note: if you want to increase a campaign +1%, 
        then you also need to apple -1% to another one
        we'll call it, decreasing arm. 
        """

        """
        We start declaring our variables 
        """
        population = list(range(len(self.campaigns))) # list of campaigns to index
        step = self.step # degrees of freedom 
        temp_budget = copy.deepcopy(self.budget_allocation) # create a temporary variable  ( if not we overwrite ) 
        temp_budget[arm] += step #we increase campaign[arm]
        q_values = q_values.tolist() #we transform q_values to a list 
        
        """
        stochastic process to choose and decrease another campaign 
        """
        # SOLUTION TO BUG ID 7 -- read Test OKT-B
        #print(f"---{len(population)-len(self.stopped)}")
        #time.sleep(0.01)
        if len(population)-len(self.stopped)==1:
            temp_budget[arm] = 1
        else:
            """
            if we have no data, randomly decrease a campaign
            """
            if all(v == 0 for v in q_values):
                dec = random.randint(0,len(self.campaigns)-1)
                if dec != arm:
                    temp_budget[dec] -= step
                else:
                    while dec == arm:
                        dec = random.randint(0,len(self.campaigns)-1)
                    temp_budget[dec] -= step
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
                if dec != arm and dec not in self.stopped:
                    #TODO SOLUTION OF BUG 1
                    if temp_budget[dec] < step:
                        temp_budget[arm] -= temp_budget[dec]
                        #temp_budget[arm] -= step
                        temp_budget[dec] = 0
                        print(f'##### Campaign {dec} was stopped completely ###')
                        #TODO delete campaign from the state ( make it ignore it )
                        self.stopped.append(dec)
                    else:
                        temp_budget[dec] -= step
                else:
                    while True:
                        dec = int(random.choices(population, weights=decrease_prob, k=1)[0])
                        if dec == arm:
                            continue
                        if dec in self.stopped:
                            continue
                        else:
                            break
                    #SOLUTION OF BUG 1
                    if temp_budget[dec] < step:
                        temp_budget[arm] -= temp_budget[dec]
                        temp_budget[arm] -= step
                        temp_budget[dec] = 0
                        print(f'##### Campaign {dec} was stopped completely ###')
                        self.stopped.append(dec)
                    else:
                        temp_budget[dec] -= step
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
            print(f"Chosen arm: {arm}, Stopped Campaigns: {self.stopped}")
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
        """
        here we validate budgets we tolerate 
        a minimum of 95%
        a maximum of 102.5%
        -- +2.5% ===> maximum overspent allowed.  
        """
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
    """
    #Changes campaigns ROAS randomly for testing purposes. 
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