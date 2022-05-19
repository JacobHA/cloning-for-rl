# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 11:35:42 2021

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
    
A,S=4,15
mdp=MDP_dynamic(A,S)
T = twister(A,S)
qss = little_q(mdp, T)

qP, qT = PT_decomp(qss)

rho_true, _, v_true = Perron_info(qss)

theta_true = -np.log(rho_true)
#%%
all_pops = []
# First evolve for a bit
NUM_POPS = 5
for pop_num in range(NUM_POPS):
    pop = CloningPopulation(qP,np.log(np.diag(qT)), N_items=100, Nsim_steps=1, N_eq_steps=100, beta=BETA, refill_w_curr_dist = True)
    pop.equilibrate()
    all_pops.append(pop)
    printf('Population', pop_num, out_of = NUM_POPS)

    
avgd_pop = np.mean([pop.get_distribution() for pop in all_pops], axis=0)

#%%
count = 1
THRESHOLD = 0.95
STEPS = 1_000
thetas=np.zeros((NUM_POPS, STEPS))
animation_list=[]
for step in range(STEPS):
    printf('Step', step, out_of = STEPS)
    for popnum, pop in enumerate(all_pops):
        pop.run_sim()
        thetas[popnum][step] = pop.get_free_energy()
        if np.random.uniform() > THRESHOLD:
            count += 1
            # print(count)
            avgd_pop = (avgd_pop*(count - 1) + np.array(np.mean([ pop.get_distribution() for pop in all_pops], axis=0)))/count    
    animation_list.append(avgd_pop)
#%%
avgd_theta = thetas.mean()
axis_length = STEPS
plt.figure()
plt.plot(thetas.mean(axis=0), 'b-', label='Thetas')
plt.plot([avgd_theta]*axis_length, 'r--', label='Averaged Theta')
plt.plot([theta_true]*axis_length, 'k-', label='True Theta')
plt.legend()
print(f'True theta: {theta_true}')
print(f'Estimated theta: {avgd_theta}')

plt.figure()
plt.plot(avgd_pop, 'b-', label='Avg. Simulated Distribution')
plt.plot(v_true, 'k-', label='True Distribution')
plt.legend()

#%%
import matplotlib.animation as animation
# LAZY AFTER_THE_FACT ANIMATION
fig, ax = plt.subplots()
line, = ax.plot(animation_list[0])
ax.plot(Perron_info(P@T)[2], 'k', label='True distribution')

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
    fig, animate, interval=50, blit=True, save_count=1, repeat=False, init_func=init, frames=10_000)
