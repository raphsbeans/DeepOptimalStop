import tensorflow as tf
import time as t
from scipy import stats
import numpy as np

def multiBrownian(M, N, dim, T):
    '''
    A multidimensional independent Brownian motion.
    M: Number of samples.
    N: Number of periods.
    dim: Dimension of the brownian motion.
    T: Time interval
    '''
    
    dt = tf.convert_to_tensor(T / (N-1), dtype=tf.float64)
    Z = tf.math.sqrt(dt) * tf.random.normal((M, N, dim), dtype=tf.float64)
    return tf.math.cumsum(Z, axis=1)

def geometricBM(nb_samples, nb_periods, dim, T, S0, rate, div_yield, sigma, corr):
    '''
    This function will simulate a geometric BM
    
    S0: Initial value. shape = (dim)
    rate: Risk free interest rate (scalar).
    div_yield: Dividends yields. shape = (dim)
    sigma: Volatilities. shape = (dim)
    corr: Correlation matrix. shape = (dim, dim) 
    '''
    # convert to tensor
    S0 = tf.convert_to_tensor(S0, dtype=tf.float64)
    div_yield = tf.convert_to_tensor(div_yield, dtype=tf.float64)
    sigma = tf.convert_to_tensor(sigma, dtype=tf.float64)
    corr = tf.convert_to_tensor(corr, dtype=tf.float64)
        
    # time grid
    t = tf.range(0, T + T / nb_periods, T / (nb_periods - 1), dtype=tf.float64) 
    t = tf.reshape(t, [nb_periods, 1])
    
    # drift
    u = rate - div_yield - sigma ** 2 / 2
    u = tf.reshape(u, [1, dim])

    # get brownian motion
    BM = multiBrownian(nb_samples, nb_periods, dim, T)    
    
    if dim > 1:
        # covariance matrix -------------------
        #temp = sigma[None] * sigma[:, None]
        #cov = tf.multiply(temp, corr)
        #A = tf.linalg.cholesky(cov)

        # or
        sigma_ = tf.reshape(sigma, [dim, 1])
        A = tf.linalg.cholesky(corr)
        A = tf.multiply(A, sigma_)
        # -------------------------------------        
        diffusion_term = tf.linalg.matvec(A, BM)  
    else:
        diffusion_term = sigma*BM
    
    res = tf.math.exp(u*t  + diffusion_term)    
    return S0 * res

def payoff (tau, x, var_compute = False):
    '''
    The payoff of the option
    Tau will be a vector one of the positions in the array of x
    '''
    P = tf.math.reduce_max(x, axis = 1) - K
    t2 = tf.cast(tau, float_type)        
    
    I = tf.math.greater(P,0)
    I = tf.where(I, 1.0, 1.0*0)    
    I = tf.cast(I, float_type)
    pay = tf.convert_to_tensor(tf.exp(-r * t2*T/n), dtype = float_type)*tf.multiply(I,P)
    if var_compute:
        return pay
    else:
        return tf.reduce_mean(pay)

def confident_interval(sigma, K, alpha = 0.05):
    '''
    This function will return
    z_(alpha/2) * sigma/sqrt(K)
    '''
    return stats.norm.ppf(1-alpha/2)*sigma/np.sqrt(K)

class Pricer:
    '''
    This class will initializate each child with a diferent Neural Network
    This way it can change each child's loss function
    '''
    
    def __init__(self, n):
        self.NN = [NeuralNet(i + 1) for i in range(n)]
        self.n = n
        self.this_stop_time = self.n * tf.ones(batch_size*training_size, dtype = int_type)
        
    def update_stop_time (self, n, X):
        f = tf.squeeze(self.NN[n].run(X[:, n, :], batch_size, training_size))
        f = tf.cast(f, dtype = int_type)
        self.this_stop_time = n * f + self.this_stop_time * (1 - f)
    
    def train(self):
        print('Generating simulated samples X')
        start = t.time()
        X = geometricBM(batch_size * training_size, n+1, d, T, S0, r, delta, sigma, pho)
        
        input_nn = tf.Variable(X, name = 'x', dtype = float_type)
        for i in range(self.n - 1, -1, -1):
            print('Training layer {} ...'.format(i))
            self.NN[i].network_learn(input_nn, self.this_stop_time)
            
            self.update_stop_time(i, input_nn)
        print('End of training - Elapsed time = {}s'.format(t.time() - start))

    def lower_bound(self, Kl, t_size):
        '''
        Return a lower bound estimate together with the variance for this estimate: (lower_bound, variance).
        The actual sample size is Kl * t_size. 
        Use t_size to divide the simulation in t_size batches of size Kl. This is preferred over a single batch of size Kl * t_size if do not have enough ram. 
        '''

        X = geometricBM(Kl * t_size, n+1, d, T, S0, r, delta, sigma, pho)
            
        input_nn = tf.Variable(X, name = 'x', dtype = float_type)
        s = []
        for i in range(self.n):
            s.append( self.NN[i].run(input_nn[:, i, :], Kl, t_size))
        s.append(tf.ones([Kl * t_size], dtype = int_type)) 
        
        tau = 0
        for i in range(self.n + 1):
            p = 1
            for j in range(i):
                p *= 1 - s[j]
            
            tau += i * s[i] * p
        
        tau = tf.squeeze(tau)
        t1 = tf.range(input_nn.shape[0], dtype = int_type)
        Indx = tf.stack((t1, tau), axis=1)
        X_stoped = tf.gather_nd(input_nn,Indx)
        print(tf.math.reduce_variance(tau))

        L = payoff (tf.squeeze(tau), X_stoped, var_compute = True)
        
        L_mean = tf.reduce_mean(L)
        
        variance = (L - L_mean)**2
        variance = tf.sqrt(tf.math.reduce_sum(variance)/(Kl * t_size - 1))
        
        return L_mean, variance

    def upper_bound(self, Ku, J):
        '''
        Return an upper bound estimate together with the variance for this estimate: (upper_bound, variance).
        The actual sample size is Kl * t_size. 
        Use t_size to divide the simulation in t_size batches of size Kl. This is preferred over a single batch of size Kl * t_size if do not have enough ram. 
        '''

        Z = geometricBM(Ku, n + 1, d, T, S0, r, delta, sigma, pho)

        # get payoff for all paths and time periods
        # change this - implement this as a function call to payoff function
        time_range = tf.range(0, n+1, dtype=float_type)
        G = tf.math.reduce_max(Z, axis = 2) - K
        I = tf.cast(tf.where(G > 0, 1.0, 0.0), float_type)        
        G = tf.exp(-r * time_range * T/n) * tf.multiply(I, G)

        C = []
        for i in range(0, n):
            # sample paths ZZ starting at 1 and with only the duration remaining for Z from point i
            ZZ = geometricBM(Ku * J, n + 1 - i, d, (n - i) * T / n, 1, r, delta, sigma, pho)
            ZZ = tf.reshape(ZZ, (Ku, J, n + 1 - i, d)) 
            
            # get initial position
            Z0 = Z[:, i, :]
            Z0 = tf.reshape(Z0, [Ku, 1, 1, d])

            # conditional paths starting from i (included) - shape (Ku, J, n-i+1, d)
            ZZZ = tf.multiply(Z0, ZZ)
            ZZZ = tf.reshape(ZZZ, [Ku * J, n-i+1, d])

            # get stopping decisions - sd
            sd = []
            for t in range(i+1, n):
                # choose positions of J and Ku in the parameters so that the data comes out in with the same alignment
                sd.append(self.NN[t].run(ZZZ[:, t-i, :], Ku * J, 1)) 
            sd.append(tf.ones([Ku * J], dtype = int_type)) # <= sd has lenght of (n - i)
            
            # reshape ZZZ one more time
            ZZZ = tf.reshape(ZZZ, [Ku, J, n-i+1, d])

            # convert to tensor
            sd_t = tf.convert_to_tensor(sd, dtype=int_type) # shape (n-i, Ku * J) -> can be reshaped to (n-i, Ku, J)
            sd_t = tf.reshape(sd_t, [n-i, Ku, J])
            sd_t = tf.transpose(sd_t, perm=[1, 2, 0]) # -> transpose to (Ku, J, n-i)

            # this is used to find the first stopping decision == 1
            def index1d(t):
                return tf.reduce_min(tf.where(tf.equal(t, 1)))

            # calculate continuation values
            G_ZZZ = tf.math.reduce_max(ZZZ, axis = 3) - K
            print(G_ZZZ.shape)
            C_k = []
            for k in range(Ku):
                payoff_sum = 0
                for j in range(J):
                    stopping_time = i + 1 + tf.reduce_min(tf.where(tf.equal(sd_t[k, j], 1)))
                    print(stopping_time)
                    st = tf.cast(stopping_time, dtype=float_type)
                    payoff_sum += tf.math.maximum(G_ZZZ[k, j, stopping_time], 0.0) * tf.exp(-r * st * T/n)
                
                payoff_avg = payoff_sum / J
                C_k.append(payoff_avg)
            C.append(tf.convert_to_tensor(C_k, dtype=float_type))

class NeuralNet:
    '''
    This class will be responsable for each individual neural network
    In our model, it will be the F^(theta_n)
    '''
    def __init__(self, n):
        xavier=tf.keras.initializers.GlorotUniform()
        self.l1=tf.keras.layers.Dense(40 + d, kernel_initializer = xavier, activation=tf.nn.relu,input_shape=[d], dtype = float_type)
        self.l2=tf.keras.layers.Dense(40 + d, kernel_initializer = xavier,activation=tf.nn.relu, dtype = float_type)
        self.out=tf.keras.layers.Dense(1,kernel_initializer = xavier,activation = tf.nn.sigmoid, dtype = float_type)
        self.train_op = tf.keras.optimizers.Adam(0.05)
        self.n = n
        self.stop_time = []
        self.stop_X = []
        
        
    # Running the model
    def run(self, X, b_size, t_size): 
      ans = []
      
      for i in range(t_size):
        X_test = X[i*b_size:(i + 1)*b_size]
        boom=self.l1(X_test)
        boom1=self.l2(boom)
        boom2=self.out(boom1)
        
        boom2 = tf.where(tf.greater(boom2, 0.5), 1, 0)
        ans.append(tf.cast(boom2, int_type)[:,0])
      
      return tf.concat(ans, axis = 0)
      
    #Custom loss fucntion
    #Change this for each n
    def get_loss(self, X):
        boom=self.l1(X)
        boom1=self.l2(boom)
        boom2=self.out(boom1)
        
        r = payoff(self.n-1,X)*boom2 +  payoff(self.stop_time,self.stop_X)*(1-boom2)
        return -r
      
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
    def network_learn(self,X, stop_time):   
        t1 = tf.range(X.shape[0], dtype = int_type)
        Indx = tf.stack((t1, stop_time), axis=1)
        X_stoped = tf.gather_nd(X,Indx)
        
        for i in range(training_size):
            self.stop_time = stop_time[i*batch_size:(i + 1)*batch_size]
            self.stop_X = X_stoped[i * batch_size: (i + 1) * batch_size]
            
            g = self.get_grad(X[i * batch_size: (i + 1) * batch_size, self.n-1])
            self.train_op.apply_gradients(
                zip(g, [self.l1.variables[0],
                        self.l1.variables[1],
                        self.l2.variables[0],
                        self.l2.variables[1],
                        self.out.variables[0],
                        self.out.variables[1]]))
        return


if __name__ == '__main__':

    float_type = tf.float64
    int_type = tf.int64

    n = 9
    d = 100
    T = 3
    S0 = 100*tf.ones(d, dtype=float_type)
    r = 0.05
    delta = 0.1*tf.ones(d, dtype=float_type) # div yield
    sigma = 0.2*tf.ones(d, dtype=float_type)
    pho = tf.linalg.diag(tf.ones(d, dtype = float_type))
    K = 100

    batch_size = 2**10
    training_size = 100
    P = Pricer(n) 

    # train model
    P.train()

    # get lower bound estimate
    Kl = 10000
    L, var = P.lower_bound(Kl, 10)
    sig = confident_interval(var, Kl)

    print('\nThe price is: {}'.format(L))  
    print("Variance is {}".format(var))
    print("Confidence Interval: {}".format(confident_interval(sig, Kl)))