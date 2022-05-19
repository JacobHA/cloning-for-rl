# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 16:17:03 2021

@author: jacob
"""

# Donsker Varadhan decomposition 

from functions import *
from cloning import CloningPopulation
# from cloning_simple import printf

def PT_decomp(A):
    colsums = A.sum(axis=0)
    T = np.diag(colsums)
    P = A/colsums
    
    assert np.allclose(A, P@T), "Decomposition does not agree with original matrix"
    return P, T


with open('random_map_dynamics.npy', 'rb') as f:
    P = np.load(f, allow_pickle=True)
    
with open('random_map_rewards.npy', 'rb') as f:
    rs = np.load(f, allow_pickle=True)

S,A=rs.shape
BETA=1
T = np.exp(BETA*rs)
T=np.diag(T.reshape(-1))
D=P@T
Dt = D.T
Pt, Tt = PT_decomp(Dt)

theta_true, u_true, v_true = Perron_info(D)
theta_trueT, u_trueT, v_trueT = Perron_info(Dt)
NORMALIZATION=u_true[0]/v_trueT[0]

assert np.isclose(theta_true, theta_trueT)
assert np.allclose(u_true, v_trueT * NORMALIZATION)

rst = np.log(np.diag(Tt))/BETA

#%%

# First collapse into s-s
little_q(Pt)
#%%
all_pops = []
# First evolve for a bit

initial_population = np.zeros(S*A)
initial_population[0] = 100
NUM_POPS = 2
for pop_num in range(NUM_POPS):
    pop = CloningPopulation(Pt,rst.flatten(), population_0 = initial_population, N_items=100, Nsim_steps=2000, beta=BETA, refill_w_curr_dist = True)
    pop.run_sim()
    all_pops.append(pop)
    printf('Population', pop_num, out_of = NUM_POPS)

    
avgd_pop = np.mean([pop.get_distribution() for pop in all_pops], axis=0)

#%%
count = 1
thetas=[]
THRESHOLD = 0.8
STEPS = 2500
animation_list=[]
for step in range(STEPS):
    printf('Step', step, out_of = STEPS)
    for pop in all_pops:
        pop.evolve()
        thetas.append(pop.get_free_energy())
        if np.random.uniform() > THRESHOLD:
            count += 1
            # print(count)
            avgd_pop = (avgd_pop*(count - 1) + np.array(np.mean([ pop.get_distribution() for pop in all_pops], axis=0)))/count    
    animation_list.append(avgd_pop)
#%%
import matplotlib.pyplot as plt
axis_length = len(thetas)
plt.figure()
plt.plot(thetas)
plt.plot([np.mean(thetas)]*axis_length, 'b-', label='Avg. Thetas')
plt.plot([Perron_info(P@T)[0]]*axis_length, 'k-', label='True Theta')
plt.legend()

plt.figure()
plt.plot(avgd_pop*NORMALIZATION, 'b-', label='Avg. Simulated Distribution')
plt.plot(Perron_info(P@T)[1], 'k-', label='True Distribution')
plt.legend()

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
