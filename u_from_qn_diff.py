# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 15:36:59 2021

@author: jacob
"""


from functions import *
import numpy as np
import matplotlib.pyplot as plt

BETA=1

with open('random_map_dynamics.npy', 'rb') as f:
    P = np.load(f, allow_pickle=True)
    
with open('random_map_rewards.npy', 'rb') as f:
A=4
S=4
P=dynamic(A,S)
T1=twister(A,S)
T2=twister(A,S)
    rs = np.load(f, allow_pickle=True)

T = np.exp(BETA*rs)
T = np.diag(T.reshape(-1))

rho_true, u_true, v_true = Perron_info(P@T)

theta_true = -np.log(rho_true)

def u_g(N):
    return np.log(np.linalg.matrix_power(P@T, N).sum(axis=0)) + N*theta_true

plt.figure()
for n in range(2,10):
    plt.plot(u_g(n), label=f'N={n}')
plt.plot(np.log(u_true), label='Log(u)')
plt.legend()

#%%
Tc=T1**0.5 * T2**0.5
D1=P@T1
D2=P@T2
Dc=P@Tc

print(np.linalg.eigvals(D1).sum())
print(np.linalg.eigvals(D2).sum())
print(np.linalg.eigvals(Dc).sum())


# print(np.linalg.eigvals(D1).sum()-np.linalg.eigvals(D1)[0])
# print(np.linalg.eigvals(D2).sum()-np.linalg.eigvals(D2)[0])
# print(np.linalg.eigvals(Dc).sum()-np.linalg.eigvals(Dc)[0])


# for rho_m1,rho_m2,rho_mc in zip(np.real_if_close(np.linalg.eigvals(D1)), np.real_if_close(np.linalg.eigvals(D2)), np.real_if_close(np.linalg.eigvals(Dc))):
#     print(np.abs(rho_m1)**0.4 * np.abs(rho_m2)**0.6 - rho_mc)

#%%
def DtoN(d,N):
    return np.linalg.matrix_power(d,N)

def q_n(d,n):
    return np.log(DtoN(d,n+1).sum(axis=0)) - np.log(DtoN(d,n).sum(axis=0))

# D=P@T
    
SA = P.shape[0]
A=4
S=int(SA/A)
P=dynamic(A,S)
T1=twister(A,S)
T2=twister(A,S)
a1=0.5
a2=1-a1
Tc=T1*a1 + T2*a2
Dc=P@Tc
D1=P@T1
D2=P@T2

r1,u1,v1=Perron_info(D1)
r2,u2,v2=Perron_info(D2)
rc,uc,vc=Perron_info(Dc)
theta1,theta2,thetac=-np.log(np.array([r1,r2,rc]))

start_n=0
N=8
delta=np.zeros((S*A,N))
for n in range(start_n, start_n + N):
    qn1,qn2,qnc = q_n(D1, n), q_n(D2, n), q_n(Dc, n)
    # delta.T[n - start_n] = qn1 * a1 + qn2 * a2 - qnc
    delta.T[n - start_n] = np.log(np.exp(qn1) * a1 + np.exp(qn2)* a2) - qnc
    
for i in range(10,16):
    plt.plot(delta[i][:], label=f'i = {i}')
    
# plt.plot([ -(theta1*a1 + theta2*a2 - thetac)]*N, 'k--', label='Theta difference')
plt.plot([ -(-np.log(r1*a1+r2*a2) - thetac)]*N, 'k--', label='Theta difference')

plt.legend()
# plt.yscale('log')
