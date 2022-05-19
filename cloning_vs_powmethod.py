# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 11:31:12 2021

@author: jacob
"""

from functions import twisted
import time 
from speedyPower import power_iteration

A=5
times4=[]
times3=[]
for S in range(1000,1001,1):
    print(S)
    # A=4
    # S=20
    matrices=[]
    for i in range(5):
        matrices.append(twisted(A,S))
    
    t1=time.time()
    for d in matrices:
        power_iteration(d)
    
    t2=time.time()
  
    times3.append(t2-t1)

    # print(f'Old total: {t2-t1}')
    # print(f'New total: {t3-t2}')
    
    
import matplotlib.pyplot as plt
plt.plot(times3, 'bo-')
plt.plot(times4, 'ro-')

# def function_to_repeat():
#     # ...

# duration = timeit.timeit(function_to_repeat, number=1000)