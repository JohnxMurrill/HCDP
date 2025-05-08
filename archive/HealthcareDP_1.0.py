# -*- coding: utf-8 -*-
"""
Created on Mon Oct 03 13:30:42 2016

@author: John

When running the Program make sure that the input file is in the same directory as the program.
"""

import datetime as dt
import numpy as np
import math
import time
import json
import sys

class HealthCareDP():
    #Init function. Declares global variables and parameters as well as caches.
    def __init__(self, state, numRounds, gamma, sigma, r, alpha, beta, mu, c):
        self.gamma = gamma
        self.sigma = sigma
        self.start = state
        self.r = r
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.c = c
        self.numRounds = numRounds
        self.cache = {}
        self.strategy = []
        self.EnumCache = {}
        self.StratCache = {}
        #[1,69,79]
    
    #Currently 16+Round. Subject to change.
    def HealthDegeneration(self, currentHealth, currentRound): 
      #9 period
      #return currentHealth - (15 + currentRound)
      #18 period
      return currentHealth - int(round(7.625 + (currentRound/4.0)))
    
    #Parameters: gamma, sigma, r
    def HealthRegained(self, investment):
        regain = int(self.gamma*((1 - math.exp((-1)*self.sigma*investment))/(1 + math.exp(((-1)*self.sigma*(investment - self.r))))))
        return regain
    #Paramaters: alpha, beta, mu
    def LifeEnjoyment(self, investment, currentHealth):
        enjoy = int(self.c*(self.beta*(currentHealth/100.0) + self.mu)*(1 - math.exp((-1)*self.alpha*investment)))
        return enjoy
        
    #Note for all functions below: State = [Round, Health, Cash]

    #Transition function. Moves from state X -> X', before investments.
    def Transition(self, state):
        #93.6 and 46.8 are the expected harvest. Adjustable.
        #9 period
        #return [state[0]+1, self.HealthDegeneration(state[1],state[0]), state[2] + int(round(93.622*state[1]/100))]
        #18 period
        return [state[0]+1, self.HealthDegeneration(state[1],state[0]), state[2] + int(round(46.811*state[1]/100))]
    
    #Investment function. Simulates the investment phase for state X'
    def Invest(self, state, investment):
        return [[state[0], min(100,state[1]+self.HealthRegained(investment[0])), investment[2]],self.LifeEnjoyment(investment[1],state[1])]
        
    #InvestmentEnum and StateEnum together generate every possible state that can be transitioned to from the provided state
    #StateEnum takes in every potential division of the available cash and calls Invest to generate every potential state
    #that can be achieved from the current state.
    def StateEnum(self, state):
        investments = self.InvestmentEnum(state[2])
        allStates = []
        for i in range(len(investments)):
            newState = self.Invest(state,investments[i])
            if newState not in allStates:
                allStates.append(newState)
        return allStates
        
    #InvestmentEnum generates every potential division of the available cash for StateEnum use.
    def InvestmentEnum(self, cash):
        potentialStates = []
        prev =-1
        if str(cash) not in self.EnumCache:
            for i in xrange(cash+1):
                hr = self.HealthRegained(i)
                if not hr == prev:
                    prev = hr
                    for j in xrange(min(100,cash+1-i)):
                        #Note: Round_down is currently pruning the number of states to X/10. This increases the runspeed 8-10x
                        #but is potentially sacrificing some accuracy. Can be lowered for more accuracy, but effect is
                        #negligable below ~3-5. Increase in runspeed is substantial.
                        potentialStates.append([i,cash-i-j,round_down(j,10)])
            self.EnumCache[str(cash)] = potentialStates
        return self.EnumCache[str(cash)]
    
    #Solve function calls itself recursively to enumerate every potential state of the environment. Once a state is seen it is
    #cached. The cache is checked before any state is evaluated to make sure that it has not been seen before.
    def Solve(self, currentState):
        newState = self.Transition(currentState)
        if newState[0] > self.numRounds or newState[1] <= 0:
            return [newState, 0]
        elif str(newState) in self.cache:
            return self.cache[str(newState)]
        else: 
            investments = self.StateEnum(newState)
            highestReturn = [0,0]
            for i in range(len(investments)):
                totalValue = investments[i][1] + self.Solve(investments[i][0])[1]
                if totalValue > highestReturn[1]:
                    highestReturn = [investments[i][0],totalValue]
                    earned = [investments[i][0],totalValue, investments[i][1]]
            self.cache[str(newState)] = highestReturn
            self.StratCache[str(newState)] = earned
        return highestReturn
    
    #Calls the solve function iteratively on the max returned state. This recreates the optimal path through the tree from
    #Whatever start state is provided all the way to completion. Note than in the returned strategy, the LE earned each round
    #is the difference between each period. The number shown is the remaining LE in the game that can be earned.    
    def FindStrat(self, state):
        cur = state
        for i in range(int(self.numRounds)):
                cur = self.Solve(cur)[0]
                self.strategy.append(cur)
    
    def AnalyzeStrat(self, strategy):
        alternate = []
        losses = []
        for i in strategy[:-1]:
            alternate.append(self.Solve(i[0]))
        print alternate[0][1]
        for i in range(len(strategy[:-2])):
            losses.append(((alternate[i+1][1]+(strategy[i][1]-strategy[i+1][1]))-alternate[i][1])/float(alternate[0][1]))
        for i in range(len(alternate)-1):
            print alternate[i], losses[i]
        print alternate[-1]
        print sum(losses)

def round_down(num, divisor):
    return num - (num%divisor)


#Main function for file input and initing the program
def main():
    #print "Please enter the parameter filename. File must be in the same directory and name is case sensitive"
    fileName = sys.argv[1]#raw_input()
    dummyStrat = [[[1,86,0], 1696],[[2,86,0], 1604],[[3,85,0], 1472],[[4,86,0], 1380],[[5,85,0], 1344],[[6,80,0], 1207],[[7,79,0], 1069],[[8,78,0], 937],[[9,74,0], 892],[[10,68,0], 725],[[11,64,0], 573],[[12,59,0], 479],[[13,51,0], 430],[[14,44,0], 303],[[15,35,0], 252],[[16,26,0], 174],[[17,15,0], 93],[[18,3,0], 46],[[19,0,0],0]]
    lines = open(fileName, "rb")
    params = []
    for i in lines:params.append(str.split(i));
    params = params[10:]
    for i in range(len(params)):
        try:
            params[i] = float(params[i][0])
        except:
            params[i] = json.loads(params[i][0])
    HCDP = HealthCareDP(params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7],params[8])
    start = time.time()
    HCDP.FindStrat(params[0])
    print HCDP.strategy
    end = time.time()
    print(end - start)
    HCDP.AnalyzeStrat(dummyStrat)
    outputfilename = 'output_{}.csv'.format(fileName[:-4])
    with open(outputfilename,'w') as f:
        for row in HCDP.strategy:
            f.write("%s\n" %row)
main()