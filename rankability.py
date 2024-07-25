#!/usr/bin/env python3 
# -*- coding: utf-8 -*-

"""
Created on Mon Mar 27 16:10:41 2023
@author: Alexandre José Ferreira
"""

import numpy as np
from numpy.linalg import norm
       
class DataRankable():
    ''' Class for estimating data rankability using the rho method 
    proposed by Alexandre and Amit.
    
    requirements : numpy
    input: alternative-criteria dataframe WITHOUT alternative column name

     OBS: Each criterion can be a benefit or a cost. However, to specifically 
        analyze dominance and efficiency, increasing numerical order of the 
        criterion results in greater dominance, that is, the higher the value 
        of the criterion the better (benefit maximization).
    '''
    
    def __init__(self, df = None, dfalt = None):
        
        for columnname in df.keys():
            try:
                df[columnname].astype(float)
            except TypeError:
                print('could not convert string to float!')
                
        self.df = df
        self.numberOfAlternatives = self.df.shape[0]
        self.numberOfCriteria = self.df.shape[1]
    
    def __completeDominanceVector__(self):
        
        ''' Maximum dominance strength or
        complete dominance set 
                
        Returns
        -------
        completeDominanceVector : minimum non-dominance vector.
        TYPE: array
        '''
        x = np.array(range(self.numberOfAlternatives))
        x = np.flip(x)
        x = [float(self.numberOfCriteria*xi + 0) for xi in x]
        self.completeDominanceVector= np.array(x)
        
        return self.completeDominanceVector
        
    def __completeNonDominanceVector__(self):
        ''' Minimum dominance strength or
        set with no dominated alternative
        
        Returns
        -------
        completeNonDominanceVector : maximum non-dominance vector.
        TYPE: array
        '''
        self.completeNonDominanceVector = np.ones(self.numberOfAlternatives)* \
            self.__completeDominanceVector__().sum() \
                    /self.numberOfAlternatives
        
        return self.completeNonDominanceVector

    def __dominanceVector__(self, vec):
        '''
        Parameters
        ----------
        vec : criterion.

        Returns
        -------
        dom_each_crtrion : dominance vector for each criterion.
        '''
        dom_each_crtrion = np.zeros(self.numberOfAlternatives)
        i=0;
        for x in vec:
            for y in vec:
                if x>y: #=
                    dom_each_crtrion[i]=dom_each_crtrion[i]; #OR don[i]=don[i]+x-y;
                if x<y: #=
                    dom_each_crtrion[i]=dom_each_crtrion[i]+1; #OR don[i]=don[i]+y-x;
                i=i+1;
            i=0;
            
        return dom_each_crtrion
    
    def __dominanceEfficiencyVector__(self, vec):
        '''
        Parameters
        ----------
        vec : criterion.

        Returns
        -------
        eff_each_crtrion : dominance efficiency vector for each criterion.
        '''
        eff_each_crtrion = np.zeros(self.numberOfAlternatives)
        i=0;
        for x in vec:
            for y in vec:
                if x>y: #=
                    eff_each_crtrion[i]=eff_each_crtrion[i]; #OR don[i]=don[i]+x-y;
                if x<y: #=
                    eff_each_crtrion[i]=eff_each_crtrion[i]+y-x;
                i=i+1;
            i=0;
        
        return eff_each_crtrion
    
    def __sumDominanceVectorAllCriteria__(self):
        '''
        Returns
        -------
        dominanceVectorAllCriteria : dominance vector for all criteria.
        TYPE: array
        '''
        dom = np.zeros(self.numberOfAlternatives)
        for col in self.df.keys():
        	dom = dom + self.__dominanceVector__(self.df[col])
            
        self.df['dominance'] = dom.astype('int64')
        
        # there is symmetry in the criteria space that's why sort and flip
        dom = np.sort(dom)
        dom = np.flip(dom)
        if not dom.any(): 
            dom=self.__completeNonDominanceVector__()
        self.dominanceVectorAllCriteria = dom
        
        return self.dominanceVectorAllCriteria
    
    def __sumDomEffVectorAllCriteria__(self):
        '''
        Returns
        -------
        dominance_eff_vector : dominance efficiency vector for all criteria.
        '''
        eff = np.zeros(self.numberOfAlternatives)
        for col in self.df.keys():
        	eff = eff + self.__dominanceEfficiencyVector__(self.df[col])
            
        self.df['efficiency'] = eff
        
        # there is symmetry in the criteria space that's why sort and flip
        eff = np.sort(eff)
        eff = np.flip(eff)
        if not eff.any(): 
            eff=self.__completeNonDominanceVector__()
        self.dominanceEffVectorAllCriteria = eff
        
        self.df = self.df.sort_values(['dominance', 'efficiency'], ascending = [False, False])
        
        return self.dominanceEffVectorAllCriteria
    
    def __rankabilityIndex__(self):
        '''
        rescale 0-1 cosine similarity

        Returns
        -------
        rankability_index : rankability index estimation
        '''
        a=self.__completeDominanceVector__()
        b=self.__completeNonDominanceVector__()
        c=self.__sumDominanceVectorAllCriteria__()
        cos_sim_1 = np.dot(a, b) / (norm(a) * norm(b))
        cos_sim_2 = np.dot(c, a) / (norm(c) * norm(a))
        r = np.abs(cos_sim_2 - cos_sim_1) \
                            /(1.0 - cos_sim_1)
        self.rankabilityIndex = r
    
        return self.rankabilityIndex
    
    def __dominanceLevelVector__(self):
        '''
        Returns
        -------
        dominance_level : dominance levels where 0 is the Pareto level.
        '''
        self.df = self.df.sort_values(['dominance', 'efficiency'], ascending = [False, False])

        # caso m << n e m >=2
        level=0 # nivel de dominancia...faixa pareto
        vec=[]
        dom = self.dominanceVectorAllCriteria
        m = self.numberOfCriteria 
        for i in dom:
            for j in dom:
                if i-j >= m:
                    level+=1
                    
            vec.append(level)
            level=0
        
        self.df['level'] = vec
        
        self.dominanceLevelVector = vec
        
        return self.dominanceLevelVector
