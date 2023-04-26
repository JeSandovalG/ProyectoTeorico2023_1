import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.integrate
import time

#Simulation parameters
gamma, beta, k = 1,1,1
desv = 2/(beta*gamma) #Noise-term correlation/standard deviation (assumed from a gaussian distribution)
u = 1/2 #Forward protocol
h = 0.01 #Time difference
tf = 10 #Final simulation time
t = np.arange(0,tf+h,h) #Time vector from 0 to 10 seconds
x0 = 0
xf = 4.5 #Final position for backward trajectories (from analytic mean solution)

#Auxiliar functions
def dxdt(t,x): #Differential equation for forward trajectories
    return -(k/gamma)*(x-u*t)

def dxdtR(t,x): #Differential equation for backward trajectories
    return -(k/gamma)*(x-u*(tf-t))

def solver(t,x,x0,dxdt,h):
    x[0] = x0 #Initial condition

    for i in range(len(x)-1): #Stochastic euler-forward solving algorithm implementation
        x[i+1] = x[i]+h*dxdt(t[i],x[i])+np.random.normal(loc=0,scale = np.sqrt(desv*h))

    return x

def dWFdt(x): #Work done by a given forward trajectory
    return -k*u*(x-u*t)

def dWBdt(x): #Work done by a given backward trajectory
    return -k*u*(x-u*(tf-t))

def generateData(N): #Generates N backward or forward trajectories 
    xs = [] #Forward trajectories vector
    labels = [] #Labels vector
    
    print("Algorithm running for {} trajectories.".format(N))
    start = time.time() #Initial running time
    
    for i in range(N): 
        rand = np.random.random()
        
        if rand > 0.5: #For random > 0.5, a forward trajectory is generated (label 1)
            tr = solver(t,np.zeros(len(t)),x0,dxdt,h) #Forward trajectory
            xs.append(tr)
            labels.append(1)
            
        else: #For random < 0.5, a backward trajectory is generated (label 0)
            trB = solver(t,np.zeros(len(t)),xf,dxdtR,h)
            xs.append(trB[::-1])
            labels.append(0)
        
    xs = np.array(xs)
    #Finding the work done in for each trajectory
    ws = [] #works vector

    for i in range(N):
        if labels[i] == 0:
            wf = scipy.integrate.simps(dWBdt(xs[i]),t) #Numerical integration using Simpson's rule
            ws.append(wf)
        
        else:
            wb = scipy.integrate.simps(dWFdt(xs[i]),t) 
            ws.append(wb)
    
    #Storing data in a .csv as a dictionary
    data = {}
    
    for i in range(len(xs)):
        data["Tr_{}".format(i)] = xs[i] #trajectory number
        data["Label_{}".format(i)] = labels[i] #trajectory label
        data["W_{}".format(i)] = ws[i] #trajectory work
        
    dataframe = pd.DataFrame(data)
    dataframe.to_csv("datosModelo.csv",index = False)
    
    end = time.time()
    print("Generated {} trajectories in {:.2f} seconds.".format(N,end-start)) #Prints the total time taken for the simulation
        
generateData(1000)            
            
        
        
        
    

