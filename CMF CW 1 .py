#!/usr/bin/env python
# coding: utf-8

# In[75]:


import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as si
import os


# In[2]:


data=yf.download('RR.L',start='2021-11-17',end='2023-11-17')


# # Rolls Royce

# In[3]:


data.head()


# In[4]:


data.tail()


# In[5]:


S = data['Adj Close'][-1]
print('The spot price is', round(S,2))


# # Descriptive Analysis of Rolls Royce's Adjusted Close

# In[10]:


data.describe().round(2)


# In[14]:


data['Adj Close'].plot(figsize=(10,7))
plt.title("RR.L's movement over a 2 year period")
plt.show()


# In[15]:


data['Daily_Return'] = data['Close'].pct_change()


# In[17]:


avg_return = data['Daily_Return'].mean() * 252  # Assuming 252 trading days in a year
std_dev = data['Daily_Return'].std() * (252 ** 0.5)

print(f"Annualized Average Return: {avg_return:.2%}")
print(f"Annualized Standard Deviation: {std_dev:.2%}")


# In[18]:


fig = plt.figure()
plt.plot(log_return * 100)
plt.xlabel('Days')
plt.ylabel('Percentage % ')
plt.title('Log Return')


# # Binomial Tree 

# In[90]:


S0 =244.5             # spot price = 244.5
K = 251.83            # strike price
T = 2                 # maturity
r = 0.05              # risk free rate 
sig = 0.4959             # volatility
N = 3                 # number of periods or number of time steps  
payoff = "put"        # payoff 

print(S0)


# In[25]:


dT = float(T) / N                             # Delta t
u = np.exp(sig * np.sqrt(dT))                 # up factor
d = 1.0 / u                                   # down factor 

S = np.zeros((N + 1, N + 1))
S[0, 0] = S0
z = 1
for t in range(1, N + 1):
    for i in range(z):
        S[i, t] = S[i, t-1] * u
        S[i+1, t] = S[i, t-1] * d
    z += 1


# In[26]:


print('The up factor u is ',(u))


# In[30]:


print('The down factor d is',(d))


# In[32]:


print('The binomial tree presenting Rolls Royce price over 3 time steps ', '\n', (S))


# In[33]:


a = np.exp(r * dT)    # risk free compound return
p = (a - d)/ (u - d)  # risk neutral up probability
q = 1.0 - p           # risk neutral down probability
p


# In[34]:


S_T = S[:,-1]
V = np.zeros((N + 1, N + 1))
if payoff =="call":
    V[:,-1] = np.maximum(S_T-K, 0.0)
elif payoff =="put":
    V[:,-1] = np.maximum(K-S_T, 0.0)
V


# In[35]:


# for European Option
for j in range(N-1, -1, -1):
    for i in range(j+1):
        V[i,j] = np.exp(-r*dT) * (p * V[i,j + 1] + q * V[i + 1,j + 1])
V


# In[36]:


print('European ' + payoff, str( V[0,0]))


# # Monte Carlo Simulation

# In[64]:


def mcs_simulation_np(m):       #m is the number of steps
    M = m
    I = m
    dt = T / M 
    S = np.zeros((M + 1, I))
    S[0] = S0 
    rn = np.random.standard_normal(S.shape) 
    for t in range(1, M + 1): 
        S[t] = S[t-1] * np.exp((r - sig ** 2 / 2) * dt + sig * np.sqrt(dt) * rn[t]) 
    return S


# In[65]:


S2 = mcs_simulation_np(1000)


# In[67]:


fig = plt.figure()
plt.plot(S2)
fig.suptitle('Monte Carlo Simulation: Rolls Royce')
plt.xlabel('Day')
plt.ylabel('Price')
plt.show()


# In[69]:


n, bins, patches = plt.hist(x=S[-1,:], bins='auto', color='#0504aa',alpha=0.7, rwidth=0.85)

plt.grid(axis='y', alpha=0.75)
plt.xlabel('S_T')
plt.ylabel('Frequency')
plt.title('Frequency distribution of the simulated end-of-period values')


# In[70]:


put = np.mean(np.maximum(K - S2[:,-1],0))
print('Monte Carlo Simulation & Option price - European put', str(put))


# # Black Scholes Merton Model

# In[71]:


def euro_option_bs(S0, K, T, r, vol, payoff):
    
    d1 = (np.log(S / K) + (r + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    if payoff == "call":
        option_value = S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0)
    elif payoff == "put":
        option_value = - S * si.norm.cdf(-d1, 0.0, 1.0) + K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0)
    
    return option_value


# In[78]:


euro_option_bs(S0, K, T, r, vol, 'put')
print('The BS put price is', round(put, 2))


# # Delta 

# In[97]:


import numpy as np
from scipy.stats import norm


# In[98]:


def delta(S, K, T, r, q, vol, payoff): # q = dividend = 0 
    
    d1 = (np.log(S / K) + (r - q + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    if payoff == "call":
        delta = np.exp(-q * T) * si.norm.cdf(d1, 0.0, 1.0)
    elif payoff == "put":
        delta =  - np.exp(-q * T) * si.norm.cdf(-d1, 0.0, 1.0)
    
    return delta


# In[96]:


delta(S, K, T, r, 0, vol, 'put')


# In[99]:


S11 = np.linspace(240,500,100)
Delta_Call = np.zeros((len(S11),1))
Delta_Put = np.zeros((len(S11),1))
for i in range(len(S11)):
    Delta_Put [i] = delta(S11[i], K, T, r, 0, vol, 'put')


# In[100]:


fig = plt.figure()

plt.plot(S11, Delta_Put, '--')
plt.grid()
plt.xlabel('Rolls Royce Price')
plt.ylabel('Delta')
plt.title('Delta')
plt.legend(['Delta for Put'])


# # Gamma

# In[111]:


import numpy as np
from scipy.stats import norm


# In[112]:


def gamma(S, K, T, r, q, vol, payoff):      #q = dividend = 0
    
    d1 = (np.log(S / K) + (r - q + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    gamma = np.exp(- q * T) * si.norm.pdf(d1, 0.0, 1.0) / (vol * S * np.sqrt(T))
    
    return gamma


# In[125]:


gamma(244.5,251.83,2,0.05,0, 0.4959,'put')


# In[133]:


S = np.linspace(240,500,100)
Gamma = np.zeros((len(S),2))
for i in range(len(S)):
    Gamma [i] = gamma(S[i],251.83, 2,0, 0.05, 0.4959, 'put')


# In[134]:


fig = plt.figure()
plt.plot(S, Gamma, '-')
plt.grid()
plt.xlabel('Stock Price')
plt.ylabel('Gamma')
plt.title('Gamma')
plt.legend(['Gamma for Put'])


# # Vega

# In[135]:


def vega(S, K, T, r, vol, payoff):
    
    d1 = (np.log(S / K) + (r + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    N_d1_prime=1/np.sqrt(2 * np.pi) * np.exp(-d1**2/2)
    vega = S * np.sqrt(T) * N_d1_prime
    
    return vega


# In[137]:


vega(244.5,251.83,2,0.05,0.4959,'put')


# In[143]:


vol = np.linspace(0.1,0.6,13)
Vega = np.zeros((len(vol),1))
for i in range(len(vol)):
    Vega [i] = vega(244.5,251.83,2,0.05, vol[i], 'put')


# In[144]:


fig = plt.figure()
plt.plot(vol, Vega, '-')
plt.grid()
plt.xlabel('Volatility')
plt.ylabel('Vega')
plt.title('Vega')
plt.legend(['Vega for Put'])


# # Theta

# In[141]:


def theta(S, K, T, r, vol, payoff):
    
    d1 = (np.log(S / K) + (r + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    N_d1_prime=1/np.sqrt(2 * np.pi) * np.exp(-d1**2/2)
    
    if payoff == "call":
        theta = - S * N_d1_prime * vol / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0)
    elif payoff == "put":
        theta = - S * N_d1_prime * vol / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0)
    
    return theta


# In[142]:


theta(244.5,251.83,2,0.05,0.4959,'put')


# In[145]:


T = np.linspace(0.25,3,12)
Theta_Call = np.zeros((len(T),1))
Theta_Put = np.zeros((len(T),1))
for i in range(len(T)):
    Theta_Put [i] = theta(244.5,251.83, T[i], 0.05, 0.25, 'put')


# In[147]:


fig = plt.figure()
plt.plot(T, Theta_Put, '-')
plt.grid()
plt.xlabel('Time to Expiry')
plt.ylabel('Theta')
plt.title('Theta')
plt.legend(['Theta for Put'])


# In[ ]:




