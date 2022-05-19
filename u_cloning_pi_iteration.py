# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 16:49:51 2021

@author: jacob
"""


# Donsker Varadhan decomposition 




with open('random_map_dynamics.npy', 'rb') as f:
    P = np.load(f, allow_pickle=True)
    
with open('random_map_rewards.npy', 'rb') as f:
    rs = np.load(f, allow_pickle=True)


BETA=1
T = np.exp(BETA*rs)
T=np.diag(T.reshape(-1))
D=P@T

#%%
# First begin with random policy

from functions import random_prior, policy_from_u
S,A = rs.shape
pi0 = random_prior(A,S)

pop = CloningPopulation(P,rs.flatten(), N_items=20, Nsim_steps=1000, beta=BETA, refill_w_curr_dist = True)
pop.get_population()

def trans(population, pi):
    trans_pop = np.zeros(SA)
    # Transition the remaining agents:       
    # TODO : consider nonzeroing everywhere before loop for speedup?
    for i,amt_in_i in enumerate(population):
        a = (i+1) % A
        s = (i+1-A) // S
        amt_in_i = int(amt_in_i)
        if amt_in_i != 0:
            for agents in range(amt_in_i):
                # Get a random sample from all non-zero transitions possible:
                transitioned_to = np.random.choice(np.argwhere(P.T[i] != 0).flatten(),\
                                 p = P.T[i][P.T[i] != 0] * pi[i] / (P.T[i][P.T[i] != 0] * pi[i]).sum() )
                trans_pop[transitioned_to] += 1
    return trans_pop


#%%
import matplotlib.animation as animation
# LAZY AFTER_THE_FACT ANIMATION
fig, ax = plt.subplots()
line, = ax.plot(animation_list[0])
ax.plot(Perron_info(P@T)[1], 'k', label='True distribution')
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()

count_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

def init():
    """initialize animation"""
    line.set_ydata([])
    count_text.set_text('')
    return line, count_text

def animate(i):
    """animate each step"""
    if i < len(animation_list):
        line.set_ydata(animation_list[i]*NORMALIZATION)  # update the data.
        count_text.set_text(f'Step number: {i}')
    else:
        count_text.set_text(f'Step number: {len(animation_list)}')
    return line, count_text
animate(0)
ani = animation.FuncAnimation(
    fig, animate, interval=10, blit=True, save_count=1, repeat=False, init_func=init, frames=10_000)
