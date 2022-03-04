from unittest import TestResult
import os
from scipy.interpolate import make_interp_spline, BSpline

from main import *
from mab import *

#we should do the testing here
#i'll start manually but we should automate it
#···· IMPORTANT!!! ---> campaign_id must be equal to arm id for the mab algorithm to work. 
campaign1 = Campaign(0,0,0,0,0,2)
campaign2 = Campaign(1,0,0,0,0,0)
campaign3 = Campaign(2,0,0,0,0,3)

campaigns = [campaign1,campaign2,campaign3]

time = 200
total_budget = 500
test_env = State(total_budget,time,campaigns)

#
#random_agent = RandomAgent(test_env,200)
#random_agent_result = random_agent.act()

epsilon_agent = EpsilonGreedyAgent(test_env, 0.9, 0.9, 5,time)
epsilon_agent_result = epsilon_agent.act()

#softmax_agent = SoftmaxExplorationAgent(test_env, tau=0.01, max_iterations=100)
#softmax_agent_result = softmax_agent.act()

#optimistic_agent = OptimisticAgent(test_env,10,10,100)
#optimistic_agent_result = optimistic_agent.act()

#ucb_agent = UCBAgent(test_env, 1,100)
#ucb_agent_result = ucb_agent.act()

total_rewards = sum(epsilon_agent_result["rewards"])
print(f"Total Reward : {total_rewards}")

cum_rewards = epsilon_agent_result["cum_rewards"]
arm_counts = epsilon_agent_result["arm_counts"]
rewards = np.array([sum(test_env.history[i][1]) for i in range(len(test_env.history))])
T = np.array(range(0,test_env.time))

xnew = np.linspace(T.min(), T.max(), 300)  
spl = make_interp_spline(T, rewards, k=3)  # type: BSpline
power_smooth = spl(xnew)
plt.plot(xnew, power_smooth)
plt.show()



"""
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



