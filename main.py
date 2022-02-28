from logging import raiseExceptions
import math
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from itertools import combinations, permutations

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

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

    def __init__(self,budget,total_time,campaigns,algorithm):
        self.budget = budget
        self.time = total_time
        self.campaigns = campaigns
        self.current_time = 0
        self.current_budget = self.budget/self.time
        #dictionary that contains all states, where the key is a timestamp
        self.history = {}
        self.budget_allocation = {}
        self.remaining = budget
        self.oktopus = algorithm

        self.step = 0.01
        self.max_steps = 5

    def next_timestamp(self,action):
        self.current_time += 1
        self.remaining-=self.current_budget
        reward = self.get_reward()
        old_state = self.get_state(self.history[self.current_time-1][0])
        new_state = self.get_state(self.budget_allocation)
        self.oktopus.update(old_state, action, new_state, reward)
        return(self.remaining)

    def get_reward(self):
        #computes the reward for taking action a in state s
        #serà la suma de los rois actuales - los rois antiguos --> si han aumentado, positiva sino punishment
        if self.current_time == 0:
            return 0
        current_roi = sum([campaign.roi for campaign in self.campaigns])
        past_roi = self.history[self.current_time-1][1]
        reward = current_roi-past_roi
        print(f'The reward for the current timestamp is {reward}')
        return reward

    def take_action(self):
        #given an action (i,j...), changes budget allocation
        #stores old state
        self.history[self.current_time] = (self.budget_allocation,self.get_reward())
        #creates new state
        print("AI is taking an action...")
        action = self.oktopus.choose_action(self.get_state(self.budget_allocation))
        print(f'AI took action {action}')
        i = 0
        for value in self.budget_allocation:
            self.budget_allocation[value] *= action[i]
            i += 1
        self.allocate_budget()
        return action

    @staticmethod        
    def get_state(budget_allocation):
        return tuple(budget_allocation.values())

    def initial_allocation(self):
        #returns a dict with a proportional allocation
        for campaign in self.campaigns:
            self.budget_allocation[campaign.id] = 1/len(self.campaigns)
        self.allocate_budget()
        return self.budget_allocation

    def allocation(self):
        #returns a dictionary with the budget allocation
        if self.current_time == 0:
            self.initial_allocation()
            self.history[self.current_time] = (self.budget_allocation,self.get_reward())
            self.allocate_budget()
            return self.budget_allocation
        else:
            return self.budget_allocation

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
        step = 0.01    
        max_step = 0.05
        # max percentage of change from an iteration
        amount_campaigns = len(budget_distribution)

        # we have to get the current budget allocation as a list such that //
        # budget_allocation[0] is budget campaign 1, budget_allocation[1] es budget campaign 2, etc
        # something like this:

        budget_allocation = list(budget_distribution)

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

        possible_actions = set()

        for action in possible_budget_reallocation:
            new = []
            for item in action:
                item += 1
                new.append(item)
            possible_actions.add(tuple(new))

        return possible_actions
        
    @staticmethod
    def validate_budgets(budget_allocation):
        #total budget allocation cannot be different 1
            total = 0
            for campaign in budget_allocation:
                total += campaign
            if total != 1: return False
            return True
    


class AI(State):

    def __init__(self, alpha=0.5, epsilon=0.1):
        """
        Initialize AI with an empty Q-learning dictionary,
        an alpha (learning) rate, and an epsilon rate.

        The Q-learning dictionary maps `(state, action)`
        pairs to a Q-value (a number).
         - `state` is a tuple that contains a budget distribution (0.25,0.25,0.5)
         - `action` is a tuple `(i, j...)` for an action (1, 1.04, 0.96)
        """
        self.q = dict()
        self.alpha = alpha
        self.epsilon = epsilon

    def update(self, old_state, action, new_state, reward):
        """
        Update Q-learning model, given an old state, an action taken
        in that state, a new resulting state, and the reward received
        from taking that action.
        """
        old = self.get_q_value(old_state, action)
        best_future = self.best_future_reward(new_state)
        self.update_q_value(old_state, action, old, reward, best_future)

    def get_q_value(self, state, action):
        """
        Return the Q-value for the state `state` and the action `action`.
        If no Q-value exists yet in `self.q`, return 0.
        """
        if not bool(self.q):
            return 0
        else:
            key = (tuple(state),action)
            if self.q.get(key) == None:
                return 0
            else:
                return self.q[key]
            
    def update_q_value(self, state, action, old_q, reward, future_rewards):
        """
        Update the Q-value for the state `state` and the action `action`
        given the previous Q-value `old_q`, a current reward `reward`,
        and an estiamte of future rewards `future_rewards`.

        Use the formula:

        Q(s, a) <- old value estimate
                   + alpha * (new value estimate - old value estimate)

        where `old value estimate` is the previous Q-value,
        `alpha` is the learning rate, and `new value estimate`
        is the sum of the current reward and estimated future rewards.
        """
        new_value_estimate = reward + future_rewards
        self.q[tuple(state),tuple(action)] = float(old_q + self.alpha*(new_value_estimate-old_q))
        
    def best_future_reward(self, state):
        """
        Given a state `state`, consider all possible `(state, action)`
        pairs available in that state and return the maximum of all
        of their Q-values.

        Use 0 as the Q-value if a `(state, action)` pair has no
        Q-value in `self.q`. If there are no available actions in
        `state`, return 0.
        """
        available_actions = State.available_actions(state)
        actions = dict()
        #if there are no available actions in the state return 0 
        if len(available_actions) == 0:
            return 0
        for action in available_actions:
            actions[action] = self.get_q_value(state,action)
        max_action = max(actions, key=actions.get)
        return int(actions[max_action])

    def choose_action(self, state, epsilon=True):
        """
        Given a state `state`, return an action `(i, j)` to take.

        If `epsilon` is `False`, then return the best action
        available in the state (the one with the highest Q-value,
        using 0 for pairs that have no Q-values).

        If `epsilon` is `True`, then with probability
        `self.epsilon` choose a random available action,
        otherwise choose the best action available.

        If multiple actions have the same Q-value, any of those
        options is an acceptable return value.
        """
        #TODO if timestamp is 0, take no action. 
        available_actions = list(State.available_actions(state))
        actions = dict()
        if epsilon == False:
            for action in available_actions:
                actions[action] = self.get_q_value(state,action)
            max_action= max(actions, key=actions.get)
            return max_action
        elif epsilon == True:
            random_action = [0,1] # 0 will take a random action, 1 will take best availabe action
            next_action = random.choices(random_action,weights=(self.epsilon,1-self.epsilon),k=1)
            if next_action == [0]:
                return random.choice(available_actions)
            elif next_action == [1]:
                for action in available_actions:
                    actions[action] = self.get_q_value(state,action)
                max_action = max(actions, key=actions.get)
                return max_action

""""
class DeepAI(State):
    
    def __init__(self,state,state_size,action_space, model_name = 'Oktopus 2.0'):
        self.state = state
        self.action_space = action_space
        self.state_size = state_size

        self.model_name = model_name
        self.memory = deque(maxlen=200)
        self.gamma = 0.95  #learning rate 
        self.epsilon = 1.0 #randomness of actions
        self.epsilon_final = 0.01
        self.epsilon_decay = 0.995
    
        self.model = self.model_builder()
    
    def model_builder(self):
        #define the sequential deep neural network

        model = tf.keras.models.Sequential()
    
        model.add(tf.keras.layers.Dense(units=32, activation='relu', input_dim=self.state_size))
    
        model.add(tf.keras.layers.Dense(units=64, activation='relu'))
    
        #model.add(tf.keras.layers.Dense(units=128, activation='relu'))
    
        model.add(tf.keras.layers.Dense(units=self.action_space, activation='linear'))
    
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=0.001))
        #return the compiled model
        return model
    
    def take_action(self,state):
        if random.random() <= self.epsilon:
            #take a random action. 
            # if epsilon=1, always take a random action
            # as epsilon decrease to 1%, only 1% of random actions            
            return random.randrange(self.action_space)
        #if we don't take a random action
        #predict best actions
        actions = self.model.predict(state)
        #will return up to 3 bes possible actions, we'll return the best 
        return np.argmax(actions[0])

    def batch_train(self, batch_size):
        #barch_size --> number of the block of information we want to train 
        #initialize an empty train batch
        batch = []
        #experience replay
        for i in range(len(self.memory) - batch_size + 1, len(self.memory)):
            #take the last positions of the memory ( most recent )
            batch.append(self.memory[i])
        #itero entre todas las possibles actiones y estados en la memoria 
        for state, action, reward, next_state, done in batch:
            reward = reward
            #if the agent didn't finish --> we still have Budget 
            if not done:
                #we apply the Bellman equation
                #gamma is a discount factor applied to the best possible resultant state
                reward = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
    
        target = self.model.predict(state)
        #for the action that the agent took, I assign the reward calculated previously
        target[0][action] = reward
        #fit new information into the model
        self.model.fit(state, target, epochs=1, verbose=0)
        #decrement epsilon 
        if self.epsilon > self.epsilon_final:
            self.epsilon *= self.epsilon_decay


"""