# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 16:53:45 2021

@author: jacob
"""

from functions import *
from cloning import CloningPopulation
import numpy as np
import matplotlib.pyplot as plt

def printf(label, i, out_of = 0):
    if out_of != 0:
        print(f'\r{label}: {i}/{out_of}', flush=True, end='')
    else:
        print(f'\r{label}: {i}', flush=True, end='')
    if out_of != 0:
        if i == out_of-1:
            print('\nFinished.')
    return

S=10
A=4
BETA=10

P=dynamic(A,S)
T=twister(A,S, beta=BETA)

rs = np.log(T.diagonal()) / BETA # WAS MISSING !!! AND BETA IN T DEF

with open('random_map_dynamics_BIG.npy', 'rb') as f:
    P = np.load(f, allow_pickle=True)
    
with open('random_map_rewardsBIG.npy', 'rb') as f:
    rs = np.load(f, allow_pickle=True)

T = np.exp(BETA*rs)
T=np.diag(T.reshape(-1))

theta_true, _, v_true = Perron_info(P@T)

#%%

def init_pops(NUM_POPS = 1, nitems = 1, nsteps_to_reach_eq=5_000, beta=BETA):
    all_pops = []
    # First evolve for a bit

    for pop_num in range(NUM_POPS):
        pop = CloningPopulation(P,rs.flatten(), N_items=nitems, Nsim_steps=nsteps_to_reach_eq, beta=beta, refill_w_curr_dist = True)
        pop.run_sim()
        all_pops.append(pop)
        printf('Population', pop_num, out_of = NUM_POPS)
            
    avgd_pop = np.mean([pop.get_distribution() for pop in all_pops], axis=0)
    return all_pops, avgd_pop

#%%
def RMSE(vec1, vec2):
    return np.sqrt( ((vec1 - vec2)**2).sum() )
    
max_population = 196
RMSE_THRESHOLD = 0.0001
steps_reqd = []
steps_std = []
for n in range(50,51,1): #range(40, max_population, 10):
    print(f'\nPopulation Value: {n}/{max_population}')
    avgstep = []
    for avgstep_samples in range(1):
        pop_list, avgd_pop = init_pops(NUM_POPS = 1, nitems = n, nsteps_to_reach_eq = 1)
        
        count = 1
        thetas=[np.mean([pop.get_free_energy() for pop in pop_list])]
        THRESHOLD = 0.9
    
        step = 1
        # while RMSE(v_true, avgd_pop) >= RMSE_THRESHOLD:
        while np.abs( (np.mean(thetas) - theta_true)/theta_true ) >= RMSE_THRESHOLD or step < 1000:
            print(f'\r{np.abs( (np.mean(thetas) - theta_true)/theta_true )}', end='', flush=True)
            # print(f'\r{RMSE(v_true, avgd_pop)}', end='', flush=True)
            for pop in pop_list:     
                pop.evolve()
                thetas.append(pop.get_free_energy())
                # if np.random.uniform() > THRESHOLD:
                    # thetas.append(pop.get_free_energy())

                #     count += 1
                #     avgd_pop = (avgd_pop*(count - 1) + np.array(np.mean([ pop.get_distribution() for pop in pop_list], axis=0)))/count    
                step += 1
        avgstep.append(step)
        print(step)
        
    steps_reqd.append(np.mean(avgstep))
    steps_std.append(np.std(avgstep))
    print(f'Avg steps required: {steps_reqd[-1]}')
    print(f'Std of steps required: {steps_std[-1]}')

    
#%%
    
plt.figure()
plt.title(f'Time taken to reach 5% absolute error in free energy. Total SA: {P.shape[0]}')
plt.xlabel('Amt of agents')
plt.ylabel('Steps required (after 1000 equilibration steps)')
plt.plot(list(range(40, max_population, 10)), steps_reqd, 'go-')
    
#%%
# Simple Data Storage:
# equilibrium time: 1000
x1=[30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190]
y1=[600.0, 301.8, 527.5, 191.7, 172.9, 148.8, 143.9, 120.8, 100.8, 111.7, 88.0, 70.5, 68.5, 90.2, 67.6, 56.5, 48.5]
std1=[203.111, 117.784, 818.885, 71.399, 89.52, 48.54, 65.207, 28.074, 41.566, 54.595, 38.787,26.235, 23.269, 45.552,27.167, 28.2, 21.551]

x2=[35, 40, 45, 50, 55, 60]
y2=[428.6, 305.4, 254.7, 249.0, 258.0, 221.0]
std2=[197.75247, 129.18529328062075, 134.3942335072454, 68.99420265500574, 67.1088667763061,77.33692520394123]

plt.figure()
plt.title(f'Time taken to reach 5% RMS error in v. Total SA: {P.shape[0]}')
plt.xlabel('Amt of agents')
plt.ylabel('Steps required (after 1000 equilibration steps)')
plt.errorbar(x2+x1[4:],y2+y1[4:],  yerr=std2+std1[4:],fmt='bo-', linewidth=1)
#%%
axis_length = len(thetas)
plt.figure()
plt.plot(thetas)
plt.plot([np.mean(thetas)]*axis_length, 'b-', label='Avg. Thetas')
plt.plot([Perron_info(P@T)[0]]*axis_length, 'k-', label='True Theta')
plt.legend()

plt.figure()
plt.plot(avgd_pop, 'b-', label='Avg. Simulated Distribution')
plt.plot(Perron_info(P@T)[2], 'k-', label='True Distribution')
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
