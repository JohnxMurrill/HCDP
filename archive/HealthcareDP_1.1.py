# -*- coding: utf-8 -*-
"""
Created on Mon Oct 03 13:30:42 2016

@author: John

When running the Program make sure that the input file is in the same directory as the program.
"""

import csv
import collections
import math
import time
import json
import numpy as np
from  pprint import pprint

class RegenerationStrategy():
    def __init__(self, gamma, sigma, r):
        self.gamma = gamma
        self.sigma = sigma
        self.r = r

    def HealthRegained(self, investment):
        regain = int(self.gamma*((1 - math.exp((-1)*self.sigma*investment))/
                                 (1 + math.exp(((-1)*self.sigma*(investment - self.r))))))
        return regain

class LifeEnjoymentStrategy():
    def __init__(self, alpha, beta, mu, c):
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.c = c

    def LifeEnjoyment(self, investment, currentHealth):
        enjoy = self.c*(self.beta*(currentHealth/100.0) + self.mu)*(1 - math.exp((-1)*self.alpha*investment))
        return enjoy

class DegenerationStrategy():
    def __init__(self, intercept, slope):
        self.intercept = intercept
        self.slope = slope

    def HealthDegeneration(self, currentHealth, currentRound):
        return currentHealth - int(round(self.intercept + (self.slope * currentRound)))

class HarvestStrategy():
    def __init__(self, maxHarvest):
        self.maxHarvest = maxHarvest

    def HarvestAmount(self, currentHealth):
        return int(round(self.maxHarvest*currentHealth/100))

DPState = collections.namedtuple('DPState', 'period, health, cash')
Investment = collections.namedtuple('Investment', 'healthExpenditure, lifeExpenditure, cashRemaining')

class HealthCareDP():
    #Init function. Declares global variables and parameters as well as caches.
    def __init__(self, state, numRounds, regenStrat, enjoymentStrat, degenStrat, harvestStrat):
        self.start = state
        self.regenStrat = regenStrat
        self.enjoymentStrat = enjoymentStrat
        self.degenStrat = degenStrat
        self.harvestStrat = harvestStrat
        self.numRounds = numRounds
        self.cache = {}
        self.EnumCache = {}
        self.StratCache = {}
        #[1,69,79]

    def HealthDegeneration(self, currentHealth, currentRound):
        return self.degenStrat.HealthDegeneration(currentHealth, currentRound)
    
    #Parameters: gamma, sigma, r
    def HealthRegained(self, investment):
        return self.regenStrat.HealthRegained(investment)

    #Paramaters: alpha, beta, mu
    def LifeEnjoyment(self, investment, currentHealth):
        return self.enjoymentStrat.LifeEnjoyment(investment, currentHealth)
        
    #Transition function. Moves from state X -> X', before investments.
    def Transition(self, state):
        nextPeriod = state.period + 1
        return DPState(nextPeriod,
                       self.degenStrat.HealthDegeneration(state.health, nextPeriod),
                       state.cash + self.harvestStrat.HarvestAmount(state.health))
    
    #Investment function. Simulates the investment phase for state X'
    def Invest(self, state, investment):
        endHealth = min(100, state.health + self.regenStrat.HealthRegained(investment.healthExpenditure))
        return (DPState(state.period,
                        endHealth,
                        investment[2]),
                self.enjoymentStrat.LifeEnjoyment(investment.lifeExpenditure, endHealth))
    
    #InvestmentEnum and StateEnum together generate every possible state that can be transitioned to from the provided state
    #StateEnum takes in every potential division of the available cash and calls Invest to generate every potential state
    #that can be achieved from the current state.
    def StateEnum(self, state):
        investments = self.InvestmentEnum(state.cash)
        allStateEnjoyments = []
        for investment in investments:
            newStateEnjoyment = self.Invest(state, investment)
            if newStateEnjoyment not in allStateEnjoyments:
                allStateEnjoyments.append(newStateEnjoyment)
        return allStateEnjoyments
        
    #InvestmentEnum generates every potential division of the available cash for StateEnum use.
    def InvestmentEnum(self, cash):
        potentialStates = []
        prev =-1
        if str(cash) not in self.EnumCache:
            for healthExpenditure in xrange(cash+1):
                hr = self.regenStrat.HealthRegained(healthExpenditure)
                if not hr == prev:
                    prev = hr
                    for lifeExpenditure in xrange(max(cash-healthExpenditure-20, 0), cash+1-healthExpenditure):
                        potentialStates.append(Investment(healthExpenditure, lifeExpenditure, cash-healthExpenditure-lifeExpenditure))
            self.EnumCache[str(cash)] = potentialStates
        return self.EnumCache[str(cash)]
    
    #Solve function calls itself recursively to enumerate every potential state of the environment. Once a state is seen it is
    #cached. The cache is checked before any state is evaluated to make sure that it has not been seen before.
    def Solve(self, currentState):
        newState = self.Transition(currentState)
        if newState.period > self.numRounds or newState.health <= 0:
            return (newState, 0)
        elif newState in self.cache:
            return self.cache[newState]
        else: 
            stateEnjoyments= self.StateEnum(newState)
            highestReturn = (0, 0)
            for (state, enjoyment) in stateEnjoyments:
                future = self.Solve(state)[1]
                totalValue = enjoyment + future
                if totalValue > highestReturn[1]:
                    highestReturn = (state, totalValue, round(enjoyment,1))
            self.cache[newState] = highestReturn
        return highestReturn
    
    #Calls the solve function iteratively on the max returned state. This recreates the optimal path through the tree from
    #Whatever start state is provided all the way to completion. Note than in the returned strategy, the LE earned each round
    #is the difference between each period. The number shown is the remaining LE in the game that can be earned.    
    def FindStrat(self, state):
        cur = state
        strategy=[]
        for i in range(int(self.numRounds)):
                cur = self.Solve(cur)[0]
                strategy.append(cur)
        return strategy
    
    def AnalyzeStrat(self, strategy, outfile):
        alternate = []
        losses = []
        for i in strategy[:-1]:
            alternate.append(self.Solve(i[0]))
        for i in range(len(strategy[:-2])):
            losses.append(((alternate[i+1][1]+(strategy[i+1][1]-strategy[i+2][1]))-alternate[i][1])/float(alternate[0][1]))
        
        output = []
        for i in range(len(alternate)-1):
            output.append([alternate[i], strategy[i+1][0], (alternate[i+1][1]+(strategy[i+1][1]-strategy[i+2][1])), losses[i], np.cumsum(losses[:i+1])[i]])    
        open(outfile, 'w').close()
        with open(outfile,'r+b') as f:
            fieldnames = ['Optimal State','Earned Life Enjoyment','Remaining Available','% Loss','Accumulated Loss']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in output:
               writer.writerow({'Optimal State':row[0],'Earned Life Enjoyment':row[1],'Remaining Available':row[2],'% Loss':row[3], 'Accumulated Loss':row[4]})

def round_down(num, divisor):
    return num - (num%divisor)

def readInFile(data_file, size):
    with open(data_file) as f:    
        data = json.load(f)
    if size == 9:
        return [data[j:j+9] for j in xrange(0,len(data),9)]
    else:
        return [data[j:j+18] for j in xrange(0,len(data),18)]
    
def BatchRun(data, startState ,HCDP, outfile):
    for i in data:
        pad = [[startState,i[0][2][1]]]+ [k[2] for k in i] +[[[19,0,0],0]]
        for j in xrange(len(pad)):
            pad[j][1] = pad[-2][1] - pad[j][1]
        pad = [[DPState(val[0][0],val[0][1],val[0][2]),val[1]] for val in pad]
        HCDP.AnalyzeStrat(pad, outfile)

#Main function for file input and initing the program
def main():
    #print "Please enter the parameter filename. File must be in the same directory and name is case sensitive"
    #fileName = sys.argv[1]#
    fileName = raw_input()
    lines = open(fileName, "rb")
    params = []

    for i in lines:params.append(str.split(i));
    params = params[10:]
    for i in range(len(params)):
        try:
            params[i] = float(params[i][0])
        except:
            params[i] = json.loads(params[i][0])

    regenStrat = RegenerationStrategy(params[2],params[3],params[4])
    enjoymentStrat=LifeEnjoymentStrategy(params[5],params[6],params[7],params[8])
    degenStrat18 = DegenerationStrategy(7.625, 0.25)
    degenStrat9 = DegenerationStrategy(15,1)
    harvestStrat18 = HarvestStrategy(46.811)
    harvestStrat9 = HarvestStrategy(93.622)
    startState = params[0] = DPState(params[0][0], params[0][1], params[0][2])

    HCDP = HealthCareDP(startState,params[1],regenStrat,enjoymentStrat, degenStrat18, harvestStrat18)
          
    start = time.time()
    output = []
    for val in [startState] + HCDP.FindStrat(startState):
        o = HCDP.Solve(val)
        output.append(o)
        print o
        
    end = time.time()
    print(end - start)
    
    BatchRun(readInFile('EighteenRound_inFile.txt', 18)[:10],startState,HCDP, 'EighteenRoundOut.csv')

    outputfilename = 'output_{}.csv'.format(fileName[:-4])
    with open(outputfilename,'wb') as f:
        fieldnames = ['Round','Health','CashonHand','LERemaining','LEEarned']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in output:
            if row[0][0]<19:
                writer.writerow({'Round':row[0][0],'Health':row[0][1],'CashonHand':row[0][2],'LERemaining':row[1],'LEEarned':row[2] })
main()
