import math
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

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

    def __init__(self,budget,total_time,campaigns):
        self.budget = budget
        self.time = total_time
        self.campaigns = campaigns
        self.current_time = 0
        self.current_budget = self.budget/self.time
        self.budget_allocation = {}
        self.remaining = budget

        self.step = 0.01
        self.max_steps = 5

    def next_timestamp(self):
        #think about it more...
        self.current_time +=1
        self.remaining-=self.current_budget
        return(self.remaining)

    def initial_allocation(self):
        #returns a dict with a proportional allocation
        for campaign in self.campaigns:
            self.budget_allocation[campaign.id] = 1/len(self.campaigns)
        return self.budget_allocation

    def get_state(self):
        #returns a dictionary with the budget allocation
        if self.current_time == 0:
            self.initial_allocation()
            self.allocate_budget()
        else:
            for campaign in self.campaigns:
                self.budget_allocation[campaign.id] = campaign.budget / self.current_budget
        return self.budget_allocation

    @classmethod
    def available_actions(budget_allocation):
        #returns a set of available actions given a particular state
        #action is a tupple of n campaigns lenght
        #EX: if campaigns = 2 then action (x,y) will represent the change on each campaign respectevely
        #EX: for n campaigns action (n1,n2...n)
        actions = set()
        for campaign in enumerate(budget_allocation):
            pass
    
    @classmethod
    def validate_budget(budget_allocation):
        #total budget allocation cannot surpass 1
        #returns True if it's valid
        #returns False if it's not valid
        total = 0
        for campaign in budget_allocation:
            total += budget_allocation[campaign] 
        if total > 1: return False
        return True
        
    
    def allocate_budget(self):
        #campaign budget = current budget * campaign%allocation
        for campaign in self.campaigns:
            campaign.budget = round(self.current_budget*self.budget_allocation[campaign.id],2)


class AI(State):
    
    def __init__(self,state,state_size,action_space, model_name = 'Oktopus 1.0'):
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