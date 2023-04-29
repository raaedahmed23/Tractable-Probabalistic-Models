
from __future__ import print_function
import numpy as np
import sys
import time
from Util import *
from CLT_class import CLT

class MIXTURE_CLT():
    
    def __init__(self):
        self.n_components = 0 # number of components
        self.mixture_probs = None # mixture probabilities
        self.clt_list = []   # List of Tree Bayesian networks
        

    '''
        Learn Mixtures of Trees using the EM algorithm.
    '''
    def learn(self, dataset, n_components=2, max_iter=50, epsilon=1e-5):
        # For each component and each data point, we have a weight
        weights = np.zeros((n_components,dataset.shape[0]))
        self.n_components = n_components

        # Randomly initialize the chow-liu trees and the mixture probabilities
        # Your code for random initialization goes here
        mix_probs = np.random.random(n_components)
        mix_probs /= mix_probs.sum()
        self.mixture_probs = mix_probs

        # initializing variable to store Chow-Liu probabilities
        T = np.zeros((n_components, dataset.shape[0]))

        self.clt_list = [CLT() for _ in range(n_components)]
        for clt in self.clt_list:
            clt.learn(dataset)

        for itr in range(max_iter):
            for k in range(n_components):

                #E-step: Complete the dataset to yield a weighted dataset
                # We store the weights in an array weights[ncomponents,number of points]
                # Your code for E-step here

                # getProb i.e weight for each sample
                for i, sample in enumerate(dataset):
                    T[k][i] = self.clt_list[k].getProb(sample)
                # T[k] = np.array([self.clt_list[k].getProb(sample) for sample in dataset])
                
                # denominator = np.sum([np.multiply(self.mixture_probs[k], T[k]) for k in range(n_components)])
                weights[k] = np.multiply(self.mixture_probs[k], T[k]) / np.sum(np.multiply(self.mixture_probs[k], T[k]))

                # M-step: Update the Chow-Liu Trees and the mixture probabilities
                # Your code for M-Step here

                self.clt_list[k].update(dataset, weights[k])

            if itr == 0:
                curr_ll = self.computeLL(dataset) / dataset.shape[0]
            else:
                new_ll = self.computeLL(dataset) / dataset.shape[0]
                if abs(new_ll - curr_ll) < epsilon:
                    return
                curr_ll = new_ll
    
    """
        Compute the log-likelihood score of the dataset
    """
    def computeLL(self, dataset):
        ll = 0.0
        # Write your code below to compute likelihood of data
        #   Hint:   Likelihood of a data point "x" is sum_{c} P(c) T(x|c)
        #           where P(c) is mixture_prob of cth component and T(x|c) is the probability w.r.t. chow-liu tree at c
        #           To compute T(x|c) you can use the function given in class CLT
        l = 0.0
        for sample in range(dataset.shape[0]):
            for k in range(self.n_components):
                l += np.multiply(self.mixture_probs[k], self.clt_list[k].getProb(dataset[sample]))
            ll += np.log(l)
        return ll
    

    
'''
    After you implement the functions learn and computeLL, you can learn a mixture of trees using
    To learn Chow-Liu trees, you can use
    mix_clt=MIXTURE_CLT()
    ncomponents=10 #number of components
    max_iter=50 #max number of iterations for EM
    epsilon=1e-1 #converge if the difference in the log-likelihods between two iterations is smaller 1e-1
    dataset=Util.load_dataset(path-of-the-file)
    mix_clt.learn(dataset,ncomponents,max_iter,epsilon)
    
    To compute average log likelihood of a dataset w.r.t. the mixture, you can use
    mix_clt.computeLL(dataset)/dataset.shape[0]
'''

if __name__ == '__main__':

    # dataset_list = ['accidents', 'baudio', 'bnetflix', 'jester', 'kdd', 'msnbc', 'nltcs', 'plants', 'pumsb_star', 'tretail']
    dataset_list = ['accidents']
    k_values = [2, 5, 10]
    ll_vals = list()
    for dataset_name in dataset_list:
        for k in k_values:
            mix = MIXTURE_CLT()

            training = '../dataset/' + dataset_name + '.ts.data'
            dataset=Util.load_dataset(training)
            mix.learn(dataset, n_components= k, max_iter=1, epsilon= 1e-1)

            validation = training = '../dataset/' + dataset_name + '.valid.data'
            valid = Util.load_dataset(validation)

            ll_vals.append(mix.computeLL(valid)/valid.shape[0])

        ll = np.asarray(ll_vals)
        print(f'K for {dataset_name}: {k_values[np.argmin(ll)]}')



    # mix_clt=MIXTURE_CLT()
    # ncomponents=10 #number of components
    # max_iter=50 #max number of iterations for EM
    # epsilon=1e-1 #converge if the difference in the log-likelihods between two iterations is smaller 1e-1
    # dataset=Util.load_dataset('../dataset/accidents.ts.data')
    # mix_clt.learn(dataset,ncomponents,max_iter,epsilon)

    # # To compute average log likelihood of a dataset w.r.t. the mixture, you can use
    # dataset=Util.load_dataset('../dataset/accidents.test.data')
    # mix_clt.computeLL(dataset)/dataset.shape[0]


    