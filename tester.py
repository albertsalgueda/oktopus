from scipy.interpolate import make_interp_spline, BSpline
import os 
from main import *
from mab import *
import time

#we should do the testing here
#i'll start manually but we should automate it
#···· IMPORTANT!!! ---> campaign_id must be equal to arm id for the mab algorithm to work. 
def Average(lst):
    return sum(lst) / len(lst)

# function to return key for any value
def get_key(val,my_dict):
    for key, value in my_dict.items():
         if val == value:
             return key
 
    return "key doesn't exist"

def test(time,total_budget,initial_q,initial_visits):

    campaign1 = Campaign(0,0,0,0,2)
    campaign2 = Campaign(1,0,0,0,0.2)
    campaign3 = Campaign(2,0,0,0,3)

    campaigns = [campaign1,campaign2,campaign3]
    test_env = State(total_budget,time,campaigns)


    #epsilon_agent = EpsilonGreedyAgent(test_env, 0.9, 0.9, 5,time)
    #epsilon_agent_result = epsilon_agent.act()
    
    #softmax_agent = SoftmaxExplorationAgent(test_env, tau=0.5, max_iterations=time)
    #softmax_agent_result = softmax_agent.act()

    optimistic_agent = OptimisticAgent(test_env,initial_q,initial_visits,time)
    optimistic_agent_result = optimistic_agent.act()

    #ucb_agent = UCBAgent(test_env, 1,time)
    #ucb_agent_result = ucb_agent.act()

    total_rewards = sum(optimistic_agent_result["rewards"])
    print(f"Total Reward : {total_rewards}")
    
    
    #-Visualize results-
    cum_rewards = optimistic_agent_result["cum_rewards"]
    arm_counts = optimistic_agent_result["arm_counts"]
    rewards = np.array([sum(test_env.history[i+1][1]) for i in range(len(test_env.history)-1)])
    T = np.array(range(1,test_env.time-1))

    xnew = np.linspace(T.min(), T.max(), 300)  
    spl = make_interp_spline(T, rewards, k=3)  # type: BSpline
    power_smooth = spl(xnew)
    plt.plot(xnew, power_smooth)
    plt.xlabel("Time Steps")
    plt.ylabel("Rewards")
    #plt.show()
    

    def budget_printer(campaign_group):
        for campaign in campaign_group:
            print(f'{campaign.id} has {campaign.roi} ROI')

    budget_printer(test_env.campaigns)
    return total_rewards


time_steps = 100
total_budget = 5000
results = {}
iterations = int(input('Select the number of iterations: '))
print('Calculating estimate of total computation time...')
print(f'It will take {(iterations*150)/60} minutes')
inp = input('Accept (a) or Cancel (c): ')
if inp=='c':
    raise("Experiment has been cancelled")
explore_q = [float(round(0.1*i+0.1,2)) for i in range(1,100)]
explore_visits = [i for i in range(1,100)]
actions = 0
combinations = len(explore_q)*len(explore_visits)*iterations
#HYPERPARAMETER TUNING:
start = time.time() #we need it to calculate time complexity of the program
for visit in range(len(explore_visits)):
    tests = [1]
    for q in range(len(explore_q)):
        a = round(100*actions/combinations,3)
        print(f"{a}% {int(a)*'#'}")
        #print(f"----q = {explore_q[q]} --- visits={explore_visits[visit]} --- result= {max(tests)} ")
        tests = []
        for i in range(iterations):
            actions +=1
            tests.append(test(time_steps,total_budget,explore_q[q],explore_visits[visit]))
        results[(explore_q[q],explore_visits[visit])] = Average(tests)
end = time.time() 
totalTime = end - start
print(f'The best configuration for complexity 10 campaigns has been {max(results.values())} with hyperparameters {get_key(max(results.values()),results)} => (q,visits)')
print(f'The experiment took: {totalTime/60}')


"""
#MODEL TESTING
for i in range(iterations):
    results.append(test(time_steps,total_budget))
print(f"Showing results after {iterations} iterations. Best: {max(results)}, Worst: {min(results)}, Average: {Average(results)}")

#ADVANCED VISUALIZATION OF MULTI-ARMED BANDIT
fig = plt.figure(figsize=[30,10])
ax1 = fig.add_subplot(121)
ax1.plot([sum(test_env.history[i][1]) for i in range(len(test_env.history))], label="cummulative rewards")
#ax1.plot(cum_rewards, label="cummulative rewards")
ax1.set_xlabel("Time steps")
ax1.set_ylabel("Cummulative rewards")
ax2 = fig.add_subplot(122)
ax2.bar([i for i in range(len(arm_counts))], arm_counts)
if os.path.exists("test.png"):
    os.remove('test.png')
plt.savefig('test.png')

plt.plot([test_env.history[i][0][1] for i in range(len(test_env.history))])
plt.plot([test_env.history[i][0][2] for i in range(len(test_env.history))])
plt.plot([test_env.history[i][0][3] for i in range(len(test_env.history))])
plt.show()
"""



