"""
import dependencies. 
"""
import random
import numpy as np
import matplotlib.pyplot as plt
import copy
import time

from mab import * # imports AI() 

class Campaign():
    def __init__(self,id,budget,spent,conversion_value):
        self.id = id                                                # unique campaign identifier
        self.daily_budget = budget                                  # active daily budget 
        self.spent = [spent]                                        # money spent across time 
        self.conversion_value = [conversion_value]                  # conversion value across time steps  

    def update(self, new_spent, new_value):
        self.spent.append(new_spent)
        self.conversion_value.append(new_value)

    def change_budget(self,value):
        self.daily_budget = self.daily_budget + value


class State(Campaign):
    def __init__(self,
        total_budget,
        total_time,
        campaigns,
        initial_allocation=0):
        """
        State class contains all the attributes and methods 
        related to the specific problem of budget optimizationin for digital adverising. 
        """
        self.total_budget = total_budget                            # budget constraint ( total budget )
        self.remaining = total_budget                               # float to keep track of remaining budget -> avoid overspent 
        self.current_budget = self.total_budget/self.total_time     # current budget = total budget / total time 
        self.budget_allocation = {}                                 # key: Campaign ID | Value: % of total budget 
   
        self.total_time = total_time                                # time constraint
        self.time_step = 12                                         # length of a time step in hours  
        self.current_time = 0                                       # current time step
        self.history = {}                                           # dictionary that contains all states, where the key is a timestamp

        self.campaigns = campaigns                                  # list that contains Campaign() instances
        self.step = 0.005                                           # step: % change allowed to AI to move. Set to 0.5%
        self.max_step = 0.010                                       # max step allowed
        self.k_arms = len(campaigns)                                # numero de arms = number of campaigns 
        self.stopped = []                                           # number of stopped campaigns | Solve bug #8
        """
        if there is no pre-defined budget allocation, we call function to make standardized distribution. 
        """
        if initial_allocation == 0:  
            self.initial_allocation()
        else:
            assert self.validate_budget(initial_allocation), 'Incorrect Budget Allocation'
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
        updates remaining budget and current_time
        increments step | level of freedom of AI()
        """
        self.remaining-=self.current_budget 
        self.current_time += 1 
        if self.remaining <= 0:
            raise Exception('No budget left')
        """
        TODO: Encontrar otras maneras de aumentar el step
        """
        self.step *= 1.001 
        if self.step>=self.max_step:
            self.step = self.max_step   

    def get_reward(self):
        """
        returns a normalized distribution for rewards, 
        based on purchase conversion value. 
        """
        if self.current_time == 0:
            return list(np.zeros(len(self.budget_allocation)))
        rewards = [] 
        for campaign in self.campaigns:
            print(campaign.conversion_value)
            rewards.append(campaign.conversion_value[-1])
        norm = [float(i)/sum(rewards) for i in rewards] 
        return norm

    def take_action(self,arm,q_values):
        """
        takes an action given chosen arm/campaign, q_values 
        """
        if self.current_time < 1:
            print('AI still has no data so no action taken(( ')
        else:
            print(f'AI is increasing budget of campaign {arm} ))')
            self.act2(arm,q_values) 
        """
        action functionalities v.2:
                new budget distribution. 
                store it in self.history 
                wait for its rewards, 
                allocate budget --> change campaign.daily_budget 
                and move time forward. 
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
        The policy is to increase with step % campaigns[arm],
        and decrease another arm based on
        stochastic process based on q_values. 
        """
        population = list(range(len(self.campaigns)))                   # list of campaigns to index
        step = self.step                                                # step = degrees of freedom 
        temp_budget = copy.deepcopy(self.budget_allocation)             # create a temporary variable  ( if not we overwrite ) 
        temp_budget[arm] += step                                        # we increase campaign[arm]
        q_values = q_values.tolist()                                    # we transform q_values to a list 
        
        """
        stochastic process to choose and decrease another campaign 
        """
        # SOLUTION TO BUG  7 
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
                dec = int(random.choices(population, weights=decrease_prob, k=1)[0])
                if dec != arm and dec not in self.stopped:
                    #TSOLUTION OF BUG 1
                    if temp_budget[dec] < step:
                        temp_budget[arm] -= temp_budget[dec]
                        temp_budget[dec] = 0
                        print(f'##### Campaign {dec} was stopped completely ###')
                        #BUG 3 delete campaign from the state ( make it ignore it )
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

        for campaign in temp_budget:
            temp_budget[campaign] = round(temp_budget[campaign],8)
        assert self.validate_budget(temp_budget)
        self.budget_allocation = temp_budget

    def allocate_budget(self):
        """
        here we update campaign daily budget 

        turns a distribution into a value
        updates campaign budget 
        campaign budget = current budget * campaign%allocation
        """
        x = 24/self.timestep
        for campaign in self.campaigns:
            campaign.budget = round(
                self.current_budget * self.budget_allocation[str(campaign.id)]*x, 8
            )
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