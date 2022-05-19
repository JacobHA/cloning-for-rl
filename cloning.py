# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 15:54:37 2021

@author: jacob
"""

from functions import *
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import copy
matplotlib.rcParams.update({'font.size': 22})


# S=10
# A=4

# P=dynamic(A,S)
# T=twister(A,S)
# D=P@T

# S=256
# A=4
# with open('random_map_dynamics.npy', 'rb') as f:
#     P = np.load(f, allow_pickle=True)
    
# with open('random_map_rewards.npy', 'rb') as f:
#     rs = np.load(f, allow_pickle=True)

# beta=1
# T = np.exp(beta*rs)
# T=np.diag(T.reshape(-1))

# P=dynamic(A,S)
# T=twister(A,S, beta=1)

#%%
        
import numpy as np
rng = np.random.default_rng()
class CloningPopulation():
    def __init__(self, system_dynamics, system_rewards, population_0=None, N_items=100, N_eq_steps=100, Nsim_steps=10, beta=1, refill_w_curr_dist = False, verbose=False):

        self.system_dynamics = system_dynamics
        self.system_rewards = system_rewards
        self.population_0 = population_0
        self.N_items = N_items
        self.N_eq_steps = N_eq_steps
        self.Nsim_steps = Nsim_steps
        self.beta = beta
        self.refill_w_curr_dist = refill_w_curr_dist
        self.verbose = verbose

        self.step_number = 1
        self.free_energy = np.nan
        self.SA = self.system_dynamics.shape[0]
        
        # Initialize the population
        if self.population_0 is None:
            print('Options for initial population are: \"spike\", \"random\", or a specific initial array.')
            print('Exiting...')
            exit(0)
        
        try:
            if self.population_0.lower() == 'spike':
                self.population_present = self.generate_spike_init_pop()
            if self.population_0.lower() == 'random':
                self.population_present = self.generate_random_init_pop()

        except AttributeError: # arrays do not have lower attribute:
            self.population_present = population_0
            
        self.population_0 = self.population_present

        # Ensure initial population satisfies required number of items
        assert self.population_present.sum() == self.N_items
        assert len(self.population_present) == self.SA
        
        self.old_pop = self.population_present
        self.distribution = self.population_present / self.N_items
        
        
    def kill(self):

        # Kill off by exp(beta*r_i)
        self.old_pop = copy.deepcopy(self.population_present)
        next_pop = copy.deepcopy(self.population_present)
        for i,amt_in_i in enumerate(self.population_present):
            amt_in_i = int(amt_in_i) # TODO: make initial pop list int typed
            if amt_in_i != 0:
                for agents in range(amt_in_i):
                    # Flip a coin:
                    kill = (1-np.exp(self.beta * self.system_rewards[i])) > np.random.uniform() 
                    if kill: next_pop[i] -= 1
                    else: next_pop[i] -= 0
                    
        self.population_present = next_pop
        if self.population_present.sum() == 0:
            self.free_energy = 1e-9 # saves log
        else:
            self.free_energy = self.population_present.sum() / self.N_items
        
    def trans(self):
        trans_pop = np.zeros(self.SA)
        # Transition the remaining agents:       
        # TODO : consider nonzeroing everywhere before loop for speedup?
        for i,amt_in_i in enumerate(self.population_present):
            amt_in_i = int(amt_in_i)
            if amt_in_i != 0:
                for agents in range(amt_in_i):
                    # Get a random sample from all non-zero transitions possible:
                    transitioned_to = rng.choice(np.argwhere(self.system_dynamics.T[i] != 0).ravel(),\
                                     p = self.system_dynamics.T[i][self.system_dynamics.T[i] != 0])
                    trans_pop[transitioned_to] += 1
        self.population_present = trans_pop      

        
    def refill(self):
        refilled_pop = copy.deepcopy(self.population_present)
        
        # Re-fill the population with N_items
        remaining_agents = int(self.N_items - sum(self.population_present))
        self.population_ratio = remaining_agents/self.N_items

        if remaining_agents == self.N_items:
            # e.g. if ALL agents were killed off!
            # print("All agents killed off")
            self.population_present = self.population_0 # self.generate_init_pop()
            return # skip refill step
        distro = self.get_distribution()[self.population_present!=0]
        distro /= distro.sum()
        nonzero_loc = np.argwhere(self.population_present != 0).ravel()
        while remaining_agents > 0:
            if self.refill_w_curr_dist:
                
                sa_loc = rng.choice(nonzero_loc, p=distro)
            else:
                sa_loc = rng.choice(nonzero_loc)

            assert self.population_present[sa_loc] != 0, "attempted to add to zero pop spot"
            refilled_pop[sa_loc] += 1   
            remaining_agents -= 1
            # print(refilled_pop.sum())
            
        self.population_present = refilled_pop
        assert self.population_present.sum() == self.N_items, f"Not repopulated/killed properly.\nCurrent population: {self.population_present.sum()}"


        
    def evolve(self):
        self.kill()
        self.trans()
        self.refill()
        self.step_number +=1


    def equilibrate(self):
        for step in range(self.N_eq_steps):
            self.evolve()
        if self.verbose:
            print(f'Allowed to equilibrate after {self.N_eq_steps} steps.')
        
    def run_sim(self):       
        for step in range(self.Nsim_steps):
            self.evolve()

            if self.verbose:
                if step % 10 == 0:
                    printf('Step', step, out_of=self.Nsim_steps)
                    

    def generate_random_init_pop(self):
        initial_pop = np.zeros(self.SA)
        for agents in range(self.N_items):
            # randomly place each item
            sa_loc = round(np.random.uniform(0, self.SA - 1))
            initial_pop[sa_loc] += 1
        return initial_pop
    
    def generate_spike_init_pop(self):
        initial_pop = np.zeros(self.SA)
        # Choose a random state-action to begin in
        sa = rng.choice(self.SA)
        # Has to be valid, e.g. not a wall state 
        while self.system_dynamics[:][sa] != np.zeros_like(self.system_dynamics[:][sa]): # P_{ji} =/= all zeros
            sa = rng.choice(self.SA)
        initial_pop[sa] = self.N_item
        return initial_pop
    
    def get_distribution(self):
        return self.population_present / sum(self.population_present)
    def get_population(self):
        return self.population_present


#%%
# N_items = 100
# init_pop = np.zeros(S*A)
# for i in range(N_items):
#     # randomly place each item
#     sa_loc = round(np.random.uniform(0,S*A-1))
#     init_pop[sa_loc] += 1
    
    
# #%%    
# # assert init_pop.sum() == N_items
# pop = CloningPopulation(P,rs.flatten(), N_items=10, Nsim_steps=10000)
# def rolling_avg(arr_old, arr_new, step_num):
#     return (arr_old*(step_num-1) + arr_new)/step_num
# #%%
# MIN_STEPS = 200
# roots=[]
# for step in range(pop.Nsim_steps):
#     if step % 100 ==0:
#         print(step)
#     # evolve population
#     roots.append(pop.get_free_energy())
#     pop.evolve()
#     if step == MIN_STEPS:
#         roll = pop.get_distribution()

#     if step > MIN_STEPS:
#         roll = rolling_avg(roll, pop.get_distribution(), pop.get_current_step_number())

# plt.figure()
# plt.plot(roll)
# #%%
# fig, ax = plt.subplots()
# ax.plot(Perron_info(P@T)[2], 'k', label='True distribution')
# pop = CloningPopulation(P,rs.flatten(), N_items=3_000, Nsim_steps=1000, beta=4)
# # pop = CloningPopulation(P,np.log(np.diag(T.diag)), N_items=15_000, Nsim_steps=1000)

# roll = pop.get_distribution()

# count_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
# line, = ax.plot(pop.get_distribution())#, label=f'Simulated distribution: step 0')
# roots = []

# def init():
#     """initialize animation"""
#     line.set_ydata([])
#     count_text.set_text('')
#     return line, count_text

# def animate(i):
#     # if i % 3 == 1:
#     #     pop.kill()
#     #     text='Killing'
#     # if i % 3 == 2:
#     #     pop.trans()
#     #     text='Transitioning'
#     # if i % 3 == 0:
#     #     pop.refill()
#     #     text='Refilling'
#     pop.evolve()
#     text='Evolving'
#     roots.append(pop.get_free_energy())
#     line.set_ydata(pop.get_distribution())  # update the data.
#     # line.set_ydata(pop.rolling_avg())
#     count_text.set_text(f'Step number: {pop.get_current_step_number()}\n{text}')
#     # plt.legend()
#     return line, count_text
# animate(0)
# ani = animation.FuncAnimation(
#     fig, animate, interval=50, blit=True, save_count=1, repeat=False, init_func=init, frames=10_000)

# #%%
# all_pops = []
# # First evolve for a bit
# for pop_num in range(1):
    
#     pop = CloningPopulation(P,rs.flatten(), N_items=200, Nsim_steps=300, beta=1)
#     pop.run_sim()
#     all_pops.append(pop)
# # avgd_pop = all_pops[0]
# #%%
# count = 1
# thetas=[]
# THRESHOLD = 0.5
# avgd_pop = np.mean([pop.get_distribution() for pop in all_pops], axis=0)

# STEPS = 5000
# for step in range(STEPS):
#     for pop in all_pops:
#         pop.evolve()
#         thetas.append(pop.get_free_energy())
#         # if np.random.uniform() > THRESHOLD:
#         #    count += 1
#         #    print(count)
#         #    avgd_pop = (avgd_pop*(count - 1) + np.array(np.mean([ pop.get_distribution() for pop in all_pops])))/count    
# plt.figure()
# plt.plot(thetas)
# plt.plot([np.mean(thetas)]*STEPS, 'b-', label='Avg. thetas')
# plt.plot([Perron_info(P@T)[0]]*STEPS, 'k-', label='True theta')
# plt.legend()
# #%%
# THRESHOLD = 0.5 # choose rv threshold to take data sample
# fig, ax = plt.subplots()
# ax.plot(Perron_info(P@T)[2], 'k', label='True distribution')
# # global avgd_pop
# avgd_pop = np.mean([pop.get_distribution() for pop in all_pops], axis=0)
# line, = ax.plot(avgd_pop)#, label=f'Simulated distribution: step 0')

# global count
# count = 1


# def init():
#     """initialize animation"""
#     line.set_ydata([])
    
#     return line, 

 
# def animate(i, avgd_pop):
#     print(i)
#     # print(count)
#     for pop in all_pops:
#         pop.evolve()
#         if np.random.uniform() > THRESHOLD:
#             count += 1
#             avgd_pop = (avgd_pop*(count - 1) + np.array( np.mean([pop.get_distribution() for pop in all_pops],axis=0) ))/count
#             # avgd_pop /= avgd_pop.sum()
#             line.set_ydata(avgd_pop)  # update the data.

#     return line,
 
# animate(0, avgd_pop)
# ani = animation.FuncAnimation(
#     fig, animate, fargs=(avgd_pop,), interval=5, blit=True, save_count=1, repeat=False, init_func=init)


# #%%
# print(Perron_info(P@T)[2])
# print(pop.get_distribution())
# v=pop.get_distribution()
# rho = pop.get_free_energy()

# # useful for viewing roots over time
# def moving_average(a, n=3) :
#     ret = np.cumsum(a, dtype=float)
#     ret[n:] = ret[n:] - ret[:-n]
#     return ret[n - 1:] / n
# #%%
# populations = []
# for population_num in range(15):
#     populations.append(CloningPopulation(P,rs.flatten(), N_items=1000, Nsim_steps=500, track=0.8, beta=4))
# #%%
# pop_step = []
# root_step=[]
# roots_std=[]

# for num,population in enumerate(populations):
#     print(num)
#     population.run_sim()
#     root_step.append(population.get_free_energy())
#     pop_step.append(population.get_avgd_pop())
    
# root_step.append(np.mean(root_step))
# pop_avg = np.mean(pop_step, axis=0)
# roots_std.append(np.std(root_step))

# #%%
# # plt.figure()
# # plt.errorbar(np.arange(len(roots)), roots/population.N_items, roots_std/population.N_items)
# plt.figure()
# plt.plot(np.mean(pop_step, axis=0)/np.mean(pop_step, axis=0).sum())
# #%%

# # # After we have rho, should be (?) easier to get u
# # u_init = np.diagonal(T) #np.ones(S*A)
# # u_init = u_init / u_init.dot(v)

# # # must sample thru 
# # u = u_init
# # for i in range(100):
# #     j = np.random.choice(range(S*A), p=P[:][i]/sum(P[:][i])) # random sample
# #     u[i] += u[j] * T[i][i] / rho
# #     u / u.dot(v)