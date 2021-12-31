from scipy.integrate import odeint 
import time
import math
import numpy as np
import pylab as py
from math import *
import numpy as np
from scipy.optimize import newton
from matplotlib import animation, rc
from matplotlib import pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from IPython.display import HTML

#constants
m1 = 2                 # mass of pendulum 1 (in kg)
m2 = 1                 # mass of pendulum 2 (in kg)
L1 = 1.4                 # length of pendulum 1 (in meter)
L2 = 1                 # length of pendulum 2 (in meter)
g = 9.8                # gravitatioanl acceleration constant (m/s^2)

# u0 = [-np.pi/10.2, 0, np.pi/5.8, 0]    # initial conditions. 
# u[0] = angle of the first pendulum
# u[1] = angular velocity of the first pendulum
# u[2] = angle of the second pendulum
# u[3] = angular velocity of the second pendulum

tfinal = 25.0       # Final time. Simulation time = 0 to tfinal.
Nt = 751
t = np.linspace(0, tfinal, Nt)

# Differential equations describing the system
def double_pendulum(u,t,m1,m2,L1,L2,g):
    # du = derivatives
    # u = variables
    # p = parameters
    # t = time variable
    
    du = np.zeros(4)
    c = np.cos(u[0]-u[2])  # intermediate variables
    s = np.sin(u[0]-u[2])  # intermediate variables

    du[0] = u[1]   # d(theta 1)
    du[1] = ( m2*g*np.sin(u[2])*c - m2*s*(L1*c*u[1]**2 + L2*u[3]**2) - (m1+m2)*g*np.sin(u[0]) ) /( L1 *(m1+m2*s**2) )
    du[2] = u[3]   # d(theta 2)   
    du[3] = ((m1+m2)*(L1*u[1]**2*s - g*np.sin(u[2]) + g*np.sin(u[0])*c) + m2*L2*u[3]**2*s*c) / (L2 * (m1 + m2*s**2))
    
    return du

def makepath(theta1, theta2):
    u0 = [np.pi*theta1/180, 0, np.pi*theta2/180, 0]
    tfinal = 25.0       # Final time. Simulation time = 0 to tfinal.
    Nt = 751
    t = np.linspace(0, tfinal, Nt)
    sol = odeint(double_pendulum, u0, t, args=(m1,m2,L1,L2,g))
    return sol

sol = makepath(-20., 35.)

#the functions below are used to create the training set
train_window = 10

def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq

def preprocess_sol(sol, train_window):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_data_normalized = scaler.fit_transform(sol.reshape(-1, 4))
    train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)
    train_inout_seq= create_inout_sequences(train_data_normalized.reshape(751, 4), train_window)
    return train_inout_seq

train_inout_seq = preprocess_sol(sol, train_window)

class MyLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(4, 100)
        self.linear = nn.Linear(100, 4)
        self.c_h = (torch.zeros(1,1,100),
                    torch.zeros(1,1,100))
    def forward(self, x):
        h, self.c_h= self.lstm(x.view(len(x) ,1, -1), self.c_h)
        predictions = self.linear(h.view(len(x), -1))
        return predictions[-1]

use_saved_model = True

if use_saved_model:
    model = MyLSTM()
    model.load_state_dict(torch.load('lstm-pend2'))  #this model is in the stable region
    #model.load_state_dict(torch.load('lstm-pend3'))  #this model is from chaotic region                  
else:
    torch.save(model.state_dict(), 'lstm-pend3')

if use_saved_model == False:
    #model = MyLSTM()
    loss_function = nn.MSELoss()
    hidden_layer_size = 100
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    epochs =2002 #52

    for i in range(epochs):
        seq_err = 0.0
        for _ in range(500):
            num = np.random.randint(len(train_inout_seq))
            seq = train_inout_seq[num][0]
            labels = train_inout_seq[num][1]
            optimizer.zero_grad()
            model.c_h = (torch.zeros(1, 1, hidden_layer_size),
                            torch.zeros(1, 1, hidden_layer_size))

            y_pred = model(seq)

            single_loss = loss_function(y_pred.reshape(1,4), labels)
            single_loss.backward()
            optimizer.step()
            seq_err+= single_loss.item()
        if i%25 == 1:
            print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
            print(f'seq: {i:3} loss: {seq_err:10.8f}')
        if seq_err < 0.000007:
            break

    print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')
    torch.save(model.state_dict(), 'lstm-pend3')

#this is the full OneStep function.   it 
def OneStep(data, steps = 100):
    print('data set length =', len(data))
    train_window = 10
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_data_normalized = scaler.fit_transform(data.reshape(-1, 4))
    fut_pred = len(data) -train_window
    test_inputs = train_data_normalized[0:train_window].reshape(train_window,4).tolist()
    #print(test_inputs)
    s2 = train_data_normalized.reshape(len(data),4).tolist()
    realdata = data
    model.eval()
    preds = test_inputs.copy()
    t2 = test_inputs
    hidden_layer_size = 100
    x = 0
    for i in range(fut_pred):
        seq = torch.FloatTensor(t2[i:])
        model.c_h = (torch.zeros(1, 1, hidden_layer_size),
                        torch.zeros(1, 1, hidden_layer_size))
        x = model(seq)
        preds.append(x.detach().numpy())
        t2.append(x.detach().numpy())
    actual_predictions = scaler.inverse_transform(np.array(preds ).reshape(-1,4))
    print(len(actual_predictions))
    
    # the following will plot the lower mass path for steps using the actual ODE sover
    # and the predicitons
    plt.figure( figsize=(10,5))
    u0 = data[:,0]     # theta_1 
    u1 = data[:,1]     # omega 1
    u2 = data[:,2]     # theta_2 
    u3 = data[:,3]     # omega_2 
    up0 = actual_predictions[:,0]     # theta_1 
    up1 = actual_predictions[:,1]     # omega 1
    up2 = actual_predictions[:,2]     # theta_2 
    up3 = actual_predictions[:,3]     # omega_2 
    x1 = L1*np.sin(u0);          # First Pendulum
    y1 = -L1*np.cos(u0);
    x2 = x1 + L2*np.sin(u2);     # Second Pendulum
    y2 = y1 - L2*np.cos(u2);
    xp1 = L1*np.sin(up0);          # First Pendulum
    yp1 = -L1*np.cos(up0);
    xp2 = xp1 + L2*np.sin(up2);     # Second Pendulum
    yp2 = yp1 - L2*np.cos(up2);
    print(x2[0], y2[0])
    plt.plot(x2[0:steps], y2[0:steps], color='r')
    plt.plot(xp2[0:steps],yp2[0:steps] , color='g')
    err = 0.0
    errs = 0.0
    cnt = 0
    badday  =0.0
    errsq = 0.0
    maxerr = 0.0
    maxloc = 0
    
    #the following attempts to make an estimate of the error.  not very good.
    for i in range(len(actual_predictions)):
        er =np.linalg.norm(realdata[i]-actual_predictions[i])
        err += er
        if er > maxerr:
            maxerr = er
            maxloc = i
    print("mean error =", err/101)
    print('maxerr =', maxerr, ' at ', maxloc)
    return actual_predictions

actual_predictions =OneStep(sol, steps=100)

