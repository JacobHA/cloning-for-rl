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
BETA=2

P=dynamic(A,S)
def T(beta):
    return twister(A,S, beta=BETA)

def rs(t, beta):
    return np.log(t.diagonal()) / BETA # WAS MISSING !!! AND BETA IN T DEF

with open('random_map_dynamics.npy', 'rb') as f:
    P = np.load(f, allow_pickle=True)
    
with open('random_map_rewards.npy', 'rb') as f:
    rs = np.load(f, allow_pickle=True)

def T(beta):
    t= np.exp(beta*rs)
    return np.diag(t.reshape(-1))
    

def theta_v(t):
    theta_true, _, v_true = Perron_info(P@t)
    
    return theta_true, v_true

#%%
plt.figure()
plt.ion()
theta_beta = []
for BETA in np.linspace(0.01,20,500):
    print(BETA)
    TT = T(BETA)
    # R = rs(TT,BETA).flatten()
    R = rs.flatten()
    ttru, vtru = theta_v(TT)
    all_pops = []
    # First evolve for a bit
    NUM_POPS = 50
    for pop_num in range(NUM_POPS):
        printf('Population', pop_num, out_of = NUM_POPS)
        pop = CloningPopulation(P,R, N_items=500, Nsim_steps=50, beta=BETA, refill_w_curr_dist = True)
        pop.run_sim()
        all_pops.append(pop)
        
    # avgd_pop = np.mean([pop.get_distribution() for pop in all_pops], axis=0)

    count = 1
    thetas=[]
    THRESHOLD = 0.8
    STEPS = 50
    for step in range(STEPS):
        theta_step = []
        printf('Step', step, out_of = STEPS)
        for pop in all_pops:
            pop.evolve()
            theta_step.append(pop.get_free_energy())
        thetas.append(np.mean(theta_step))
    theta_beta.append(np.mean(thetas)) # theta for fixed temp, BETA
    plt.scatter(BETA, -np.log(np.mean(thetas)))
    plt.show()
    plt.pause(0.0001)        
#%%
betas_list = np.linspace(0.01,20,500)[:len(theta_beta)]
plt.figure()
plt.xlabel('Inverse Temperature: beta')
plt.ylabel('Free energy: theta')
plt.plot(1/betas_list, -np.log(theta_beta), 'bo-', label='Free energy for maze')
plt.legend()



