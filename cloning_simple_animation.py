# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 16:53:45 2021

@author: jacob
"""

from functions import *
from cloning import CloningPopulation
import numpy as np
import matplotlib.pyplot as plt

BETA=1

with open('random_map_dynamics.npy', 'rb') as f:
    P = np.load(f, allow_pickle=True)
    
with open('random_map_rewards.npy', 'rb') as f:
    rs = np.load(f, allow_pickle=True)

T = np.exp(BETA*rs)
T = np.diag(T.reshape(-1))


# A,S=4,29
# P=dynamic(A,S)
# T=twister(A,S)
# rs = np.log(np.diag(T)/BETA)

rho_true, _, v_true = Perron_info(P@T)

theta_true = -np.log(rho_true)
#%%
A=4
all_pops = []
NITEMS=50
initial_pop = np.zeros(P.shape[0])
start_state = 8
initial_pop[start_state*A:start_state*A+A]=NITEMS/A
# First evolve for a bit
NUM_POPS = 3
for pop_num in range(NUM_POPS):
    pop = CloningPopulation(P,rs.ravel(), N_items=NITEMS, population_0 = initial_pop, Nsim_steps=1, N_eq_steps=2_000, beta=BETA, refill_w_curr_dist = True)
    pop.equilibrate()
    all_pops.append(pop)
    printf('Population', pop_num, out_of = NUM_POPS)

    
avgd_pop = np.mean([pop.get_distribution() for pop in all_pops], axis=0)

#%%
count = 1
THRESHOLD = 0.97
STEPS = 2_000
thetas=np.zeros((NUM_POPS, STEPS))
animation_list=[]
for step in range(STEPS):
    printf('Step', step, out_of = STEPS)
    for popnum, pop in enumerate(all_pops):
        pop.run_sim()
        theta = -np.log(pop.free_energy)
        thetas[popnum][step] = theta
        if np.random.uniform() > THRESHOLD:
            count += 1 # <= DO NOT UNCOMMENT!
            # print(count)
            avgd_pop = (avgd_pop*(count - 1) + np.array(np.mean([ pop.get_distribution() for pop in all_pops], axis=0)))/count    
            animation_list.append(avgd_pop)
 
theta_std = thetas.std(axis=0)
#%%
avgd_theta = thetas.mean()
axis_length = STEPS
plt.figure()
plt.title('Free Energy')
plt.plot(thetas.mean(axis=0), 'b-', label='Population-avgd thetas')
# plt.errorbar(list(range(len(thetas.mean(axis=0)))) ,thetas.mean(axis=0), yerr=theta_std, fmt='b-', label='Thetas')
plt.plot([avgd_theta]*axis_length, 'r--', label='Averaged Theta')
plt.plot([theta_true]*axis_length, 'k-', label='True Theta')
plt.legend()
print(f'True theta: {theta_true}')
print(f'Estimated theta: {avgd_theta}')

plt.figure()
plt.title('Average simulated right eigenvector')
plt.plot(avgd_pop, 'b-', label='Avg. Simulated Distribution')
plt.plot(v_true, 'k-', label='True Distribution')
plt.legend()

plt.figure()
plt.title('Average of exp(beta*r) wrt v: Free energy')
plt.plot([-np.log(avgd_pop.dot(np.diag(T))) for avgd_pop in animation_list])

#%%
import matplotlib.animation as animation
# LAZY AFTER_THE_FACT ANIMATION
fig, ax = plt.subplots()
line, = ax.plot(animation_list[0])
ax.plot(v_true, 'k', label='True distribution')

count_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

def init():
    """initialize animation"""
    line.set_ydata([])
    count_text.set_text('')
    return line, count_text

def animate(i):
    """animate each step"""
    if i < len(animation_list):
        line.set_ydata(animation_list[i])  # update the data.
        count_text.set_text(f'Step number: {i}')
    else:
        count_text.set_text(f'Step number: {len(animation_list)}')
    return line, count_text
animate(0)
ani = animation.FuncAnimation(
    fig, animate, interval=50, blit=True, save_count=1, repeat=False, init_func=init, frames=len(animation_list))
