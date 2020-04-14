# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 02:58:26 2020

@author: raphaelfeijao
"""
import simulation as sl
import tensorflow as tf
import numpy as np

'''
Constants
For (M = 1,000,000, d = 2, n = 1024), it takes 285 seconds == 4.75 minutes
'''
M = 1000
n = 9
d = 2
T = 3
S0 = 100
r = 0.05
delta = 0.1
sigma = 0.2
pho = 0

X = sl.simulate_x (M, n+1, d, T, S0, r, delta, sigma, pho)

'''
Payoff
'''
K = 10
def payoff (tau, x, K = K, T = T, n=n, r=r):
    '''
    The payoff of the option
    Tau will be one of the positions in the array of x
    '''
    P = tf.math.reduce_max(x, axis = 0) - K
    I = tf.math.greater(P[tau],0)
    
    P = tf.cond(I, lambda: P[tau], lambda: tf.convert_to_tensor(0.0, dtype = tf.float64))
    
    return tf.convert_to_tensor(np.exp(-r * tau*T/n), dtype = tf.float64)*P


'''
Training
For each step from 0 to N-1 we have to train a NN. We will have N neural nets
'''

class model:
    
    def __init__(self):
        #xavier=tf.keras.initializers.GlorotUniform()
        self.l1=tf.keras.layers.Dense(2,activation=tf.nn.relu,input_shape=[1000,2,10], dtype = tf.float64)
        self.l2=tf.keras.layers.Dense(1,activation=tf.nn.relu, dtype = tf.float64)
        self.out=tf.keras.layers.Dense(1,activation = tf.nn.sigmoid, dtype = tf.float64)
        self.train_op = tf.keras.optimizers.Adagrad(0.1)
        self.dim = 1000
    # Running the model
    def run(self,X): 
        boom=self.l1(X)
        boom1=self.l2(boom)
        boom2=self.out(boom1)
        return boom2
      
    #Custom loss fucntion
    #Change this for each n
    def get_loss(self,X):
        boom=self.l1(X)
        boom1=self.l2(boom)
        boom2=self.out(boom1)
        r = 0
        for i in range(self.dim):
            r += payoff(n-1,X[i])* boom2 + payoff(n,X[i])*(1-boom2)
        return -r/self.dim
      
    # get gradients
    def get_grad(self,X):
        with tf.GradientTape() as tape:
            tape.watch(self.l1.variables)
            tape.watch(self.l2.variables)
            tape.watch(self.out.variables)
            L = self.get_loss(X)
            g = tape.gradient(L, [self.l1.variables[0],self.l1.variables[1],self.l2.variables[0],self.l2.variables[1],self.out.variables[0],self.out.variables[1]])
        return g
      
    # perform gradient descent
    def network_learn(self,X):
        g = self.get_grad(X)
        print(g)
        print(self.l2.variables[0])
        print(self.l2.variables[1])
        self.train_op.apply_gradients(zip(g, [self.l1.variables[0],self.l1.variables[1],self.l2.variables[0],self.l2.variables[1],self.out.variables[0],self.out.variables[1]]))
        
        
model_test = model()

input_nn = tf.Variable(X, name = 'x')
model_test.network_learn(input_nn)