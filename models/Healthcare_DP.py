"""
Healthcare Dynamic Programming Model

This program implements a dynamic programming solution for an experimental healthcare
decision-making game where participants manage health and financial resources over
multiple rounds to maximize their "life enjoyment" score.

Author: Original by John Murrill - 2016, documentation added later

Authors Note:
This code represents the final version of the non-stochastic model which I helped to develop
with added documentation. I lost access to the original Github account I uploaded it to, and forked
this verison later on. I graduated at the end of 2017, any subsequent development was done on
forks. 

Game mechanics:
- Players start with a health score (typically 85)
- Each round, players earn money proportional to their health
- Health decreases each round at an increasing rate
- Players can invest in health regeneration, spend on life enjoyment, or save money
- The objective is to maximize total life enjoyment over all rounds
"""

import csv
import collections
import math
import time
import json
import numpy as np
from pprint import pprint

class RegenerationStrategy:
    """
    Controls how investments in health translate to actual health gains.
    
    Parameters:
    - gamma: Maximum possible health gain
    - sigma: Controls the steepness of the health gain curve
    - r: The investment amount at which 50% of max health gain is achieved
    """
    def __init__(self, gamma, sigma, r):
        self.gamma = gamma
        self.sigma = sigma
        self.r = r

    def HealthRegained(self, investment):
        """
        Calculate how much health is regained from a given investment.
        
        The formula uses a sigmoid curve to model diminishing returns on investment.
        """
        regain = int(self.gamma * ((1 - math.exp((-1) * self.sigma * investment)) /
                                (1 + math.exp(((-1) * self.sigma * (investment - self.r))))))
        return regain

class LifeEnjoymentStrategy:
    """
    Determines how investments in life enjoyment translate to score/utility.
    
    Parameters:
    - alpha: Controls the steepness of the enjoyment curve
    - beta: Weight of the health component
    - mu: Base enjoyment independent of health
    - c: Overall scaling factor
    """
    def __init__(self, alpha, beta, mu, c):
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.c = c

    def LifeEnjoyment(self, investment, currentHealth):
        """
        Calculate the life enjoyment (score) gained from a given investment.
        
        The formula models diminishing returns and scales with current health.
        """
        enjoy = self.c * (self.beta * (currentHealth / 100.0) + self.mu) * (1 - math.exp((-1) * self.alpha * investment))
        return enjoy

class DegenerationStrategy:
    """
    Controls how health declines over time.
    
    Parameters:
    - intercept: Base health loss per round
    - slope: Additional health loss per round as rounds progress
    - horizon: Round number that triggers increased degeneration
    """
    def __init__(self, intercept, slope, horizon):
        self.intercept = intercept
        self.slope = slope
        self.horizon = horizon

    def HealthDegeneration(self, currentHealth, currentRound):
        """
        Calculate how much health is lost in a given round.
        
        If the current round is beyond the horizon, degeneration rate doubles.
        """
        if currentRound <= self.horizon:
            return max(currentHealth - int(round(self.intercept + (self.slope * currentRound))), 0)
        else:
            return max(currentHealth - int(round(self.intercept + (2 * self.slope * currentRound))), 0)

class HarvestStrategy:
    """
    Determines how much money is earned each round based on current health.
    
    Parameters:
    - maxHarvest: Maximum possible harvest at 100 health
    """
    def __init__(self, maxHarvest):
        self.maxHarvest = maxHarvest

    def HarvestAmount(self, currentHealth):
        """
        Calculate the amount of money earned in a round based on current health.
        """
        return int(round(self.maxHarvest * currentHealth / 100))

# Named tuples for state representation
DPState = collections.namedtuple('DPState', 'period, health, cash')
Investment = collections.namedtuple('Investment', 'healthExpenditure, lifeExpenditure, cashRemaining')

class HealthCareDP:
    """
    Main dynamic programming class that finds optimal strategies for the healthcare game.
    
    This class implements the core algorithm to explore all possible states and determine
    the optimal decision at each state to maximize total life enjoyment.
    """
    
    def __init__(self, state, numRounds, regenStrat, enjoymentStrat, degenStrat, harvestStrat):
        """
        Initialize the dynamic programming solver.
        
        Parameters:
        - state: Starting state (DPState tuple)
        - numRounds: Total number of rounds in the game
        - regenStrat: RegenerationStrategy instance
        - enjoymentStrat: LifeEnjoymentStrategy instance
        - degenStrat: DegenerationStrategy instance
        - harvestStrat: HarvestStrategy instance
        """
        self.start = state
        self.regenStrat = regenStrat
        self.enjoymentStrat = enjoymentStrat
        self.degenStrat = degenStrat
        self.harvestStrat = harvestStrat
        self.numRounds = numRounds
        self.cache = {}  # Cache for dynamic programming
        self.EnumCache = {}  # Cache for enumerated states
        self.StratCache = {}  # Cache for strategies
    
    def HealthDegeneration(self, currentHealth, currentRound, horizon):
        """Wrapper for the degeneration strategy"""
        return max(self.degenStrat.HealthDegeneration(currentHealth, currentRound, horizon), 0)
    
    def HealthRegained(self, investment):
        """Wrapper for the regeneration strategy"""
        return max(self.regenStrat.HealthRegained(investment), 0)

    def LifeEnjoyment(self, investment, currentHealth):
        """Wrapper for the life enjoyment strategy"""
        return self.enjoymentStrat.LifeEnjoyment(investment, currentHealth)
        
    def Transition(self, state):
        """
        Transition function: Move from current state to next state before investments.
        
        This simulates the passage of one round:
        - Period increases by 1
        - Health degenerates according to the degeneration strategy
        - Cash increases by the harvest amount based on current health
        
        Returns:
        - New DPState tuple after transition
        """
        nextPeriod = state.period + 1
        return DPState(nextPeriod,
                       self.degenStrat.HealthDegeneration(state.health, nextPeriod),
                       state.cash + self.harvestStrat.HarvestAmount(state.health))
    
    def Invest(self, state, investment):
        """
        Investment function: Simulate the effects of an investment decision.
        
        Parameters:
        - state: Current state (DPState)
        - investment: Investment decision (Investment tuple)
        
        Returns:
        - Tuple of (new state after investment, life enjoyment gained)
        """
        endHealth = min(100, state.health + self.regenStrat.HealthRegained(investment.healthExpenditure))
        return (DPState(state.period,
                        endHealth,
                        investment[2]),
                self.enjoymentStrat.LifeEnjoyment(investment.lifeExpenditure, endHealth))
    
    def StateEnum(self, state):
        """
        Generate all possible state transitions from the current state.
        
        This function enumerates all possible investment decisions and their resulting states.
        
        Parameters:
        - state: Current state (DPState)
        
        Returns:
        - List of tuples (new state, life enjoyment gained) for all possible investments
        """
        investments = self.InvestmentEnum(state.cash)
        allStateEnjoyments = []
        for investment in investments:
            newStateEnjoyment = self.Invest(state, investment)
            if newStateEnjoyment not in allStateEnjoyments:
                allStateEnjoyments.append(newStateEnjoyment)
        return allStateEnjoyments
        
    def InvestmentEnum(self, cash):
        """
        Generate all possible investment decisions for a given amount of cash.
        
        This function creates a list of all unique health/enjoyment investment combinations.
        It optimizes by only considering health investments that result in different health gains.
        
        Parameters:
        - cash: Amount of cash available for investment
        
        Returns:
        - List of Investment tuples representing all possible investment decisions
        """
        potentialStates = []
        prev = -1
        if str(cash) not in self.EnumCache:
            for healthExpenditure in range(cash + 1):
                hr = self.regenStrat.HealthRegained(healthExpenditure)
                if not hr == prev:
                    prev = hr
                    for lifeExpenditure in range(max(cash - healthExpenditure - 20, 0), cash + 1 - healthExpenditure):
                        potentialStates.append(Investment(healthExpenditure, lifeExpenditure, cash - healthExpenditure - lifeExpenditure))
            self.EnumCache[str(cash)] = potentialStates
        return self.EnumCache[str(cash)]
    
    def Solve(self, currentState):
        """
        Core dynamic programming function to find the optimal strategy.
        
        This recursive function:
        1. Transitions to the next state
        2. Checks if the game is over or if the state is already cached
        3. Enumerates all possible investment decisions
        4. Recursively finds the value of each resulting state
        5. Returns the state and decision that maximize total future life enjoyment
        
        Parameters:
        - currentState: Current state (DPState)
        
        Returns:
        - Tuple of (optimal next state, total life enjoyment, immediate life enjoyment)
        """
        newState = self.Transition(currentState)
        if newState.period > self.numRounds or newState.health <= 0:
            return (newState, 0, 0)
        elif newState in self.cache:
            return self.cache[newState]
        else: 
            stateEnjoyments = self.StateEnum(newState)
            highestReturn = (0, 0)
            for (state, enjoyment) in stateEnjoyments:
                future = self.Solve(state)[1]
                totalValue = enjoyment + future
                if totalValue > highestReturn[1]:
                    highestReturn = (state, totalValue, round(enjoyment, 1))
            self.cache[newState] = highestReturn
        return highestReturn
    
    def FindStrat(self, state):
        """
        Generate the optimal path through the state space from the given starting state.
        
        Parameters:
        - state: Starting state (DPState)
        
        Returns:
        - List of states representing the optimal path
        """
        cur = state
        strategy = []
        for i in range(int(self.numRounds)):
            cur = self.Solve(cur)[0]
            strategy.append(cur)
        return strategy
    
    def AnalyzeStrat(self, strategy, ID, life, outfile):
        """
        Analyze how an actual strategy deviates from the optimal strategy.
        
        This function compares actual player decisions against optimal decisions and
        calculates the percentage loss in life enjoyment.
        
        Parameters:
        - strategy: List of actual states from a player's game
        - ID: Player identifier
        - life: Game/lifetime identifier
        - outfile: Output file for analysis results
        """
        alternate = []
        losses = []
        for i in strategy[:-1]:
            alternate.append(self.Solve(i[0]))
        for i in range(len(strategy[:-2])):
            losses.append(((alternate[i+1][1] + (strategy[i+1][1] - strategy[i+2][1])) - alternate[i][1]) / float(alternate[0][1]))
        
        output = []
        for i in range(len(alternate) - 1):
            output.append([alternate[i], strategy[i+1][0], (alternate[i+1][1] + (strategy[i+1][1] - strategy[i+2][1])), 
                          losses[i], np.cumsum(losses[:i+1])[i], strategy[i], (strategy[i][1] - strategy[i+1][1])])    
        with open(outfile, 'a+', newline='') as f:
            fieldnames = ['ID', 'Lifetime', 'Period', 'Optimal Health', 'Optimal Cash on Hand', 'Remaining Max',
                          'Optimal Earnings This Period', 'Realized Health', 'Realized Cash on Hand', 'Current LE',
                          'Earned This Period', 'Remaining Available', '% Loss', 'Accumulated Loss']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            for row in output:
               writer.writerow({'ID': ID, 'Lifetime': life, 'Period': row[0][0][0], 'Optimal Health': row[0][0][1], 
                               'Optimal Cash on Hand': row[0][0][2], 'Remaining Max': int(row[0][1]),
                               'Optimal Earnings This Period': row[0][2], 'Realized Health': row[5][0][1], 
                               'Realized Cash on Hand': row[5][0][2], 'Current LE': row[5][1], 
                               'Earned This Period': row[6], 'Remaining Available': int(row[2]), 
                               '% Loss': '%.3f' % row[3], 'Accumulated Loss': '%.3f' % row[4]})

def round_down(num, divisor):
    """Helper function to round down to nearest multiple of divisor"""
    return num - (num % divisor)

def readInFile(data_file, size):
    """
    Read player data from a JSON file.
    
    Parameters:
    - data_file: Path to the JSON file
    - size: Number of rounds per game (9 or 18)
    
    Returns:
    - List of player data grouped by game
    """
    with open(data_file) as f:    
        data = json.load(f)
    if size == 9:
        return [data[j:j+9] for j in range(0, len(data), 9)]
    else:
        return [data[j:j+18] for j in range(0, len(data), 18)]
    
def BatchRun(data, startState, HCDP, outfile):
    """
    Analyze a batch of player strategies against optimal strategies.
    
    Parameters:
    - data: List of player data
    - startState: Starting state (DPState)
    - HCDP: HealthCareDP instance
    - outfile: Output file for analysis results
    """
    with open(outfile, 'w+', newline='') as f:
        fieldnames = ['ID', 'Lifetime', 'Period', 'Optimal Health', 'Optimal Cash on Hand', 'Remaining Max',
                      'Optimal Earnings This Period', 'Realized Health', 'Realized Cash on Hand', 'Current LE',
                      'Earned This Period', 'Remaining Available', '% Loss', 'Accumulated Loss']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
    
    for i in data:
        total = i[-1][2][1]
        pad = [k[2] for k in i] + [[[19, 0, 0], 0]]
        for j in range(len(pad)):
            pad[j][1] = max(pad[-2][1] - pad[j][1], 0)
        pad = [[startState, total]] + pad
        pad = [[DPState(val[0][0], val[0][1], val[0][2]), val[1]] for val in pad]
        HCDP.AnalyzeStrat(pad, i[0][0], i[0][1], outfile)

def main():
    """
    Main function to run the HealthCareDP program.
    
    This function:
    1. Reads parameter file
    2. Initializes strategy objects
    3. Creates the DP solver
    4. Finds the optimal strategy
    5. Optionally runs batch analysis on player data
    """
    # Read parameter file
    fileName = input("Enter parameter filename: ")
    lines = open(fileName, "r")
    params = []

    for i in lines:
        params.append(str.split(i))
    params = params[10:]
    for i in range(len(params)):
        try:
            params[i] = float(params[i][0])
        except:
            params[i] = json.loads(params[i][0])

    # Initialize strategy objects
    regenStrat = RegenerationStrategy(params[2], params[3], params[4])
    enjoymentStrat = LifeEnjoymentStrategy(params[5], params[6], params[7], params[8])
    
    # Create different strategies for 9-round and 18-round games
    degenStrat18 = DegenerationStrategy(7.625, 0.25, 18)
    degenStrat9 = DegenerationStrategy(15, 1, 9)
    harvestStrat18 = HarvestStrategy(46.811)
    harvestStrat9 = HarvestStrategy(93.622)
    
    # Set starting state
    startState = params[0] = DPState(params[0][0], params[0][1], params[0][2])

    # Create DP solver (using 18-round parameters by default)
    HCDP = HealthCareDP(startState, params[1], regenStrat, enjoymentStrat, degenStrat18, harvestStrat18)
          
    # Find optimal strategy and measure execution time
    start = time.time()
    output = []
    for val in [startState] + HCDP.FindStrat(startState):
        o = HCDP.Solve(val)
        output.append(o)
        print(o)
        
    end = time.time()
    print(f"Execution time: {end - start} seconds")
    
    # Uncomment to run batch analysis on player data
    # BatchRun(readInFile('EighteenRound_inFile.txt', 18)[:10], startState, HCDP, 'EighteenRoundOut.csv')

    # Write output to CSV
    outputfilename = f'analysis\\output_{fileName[:-4]}.csv'
    with open(outputfilename, 'w+', newline='') as f:
        fieldnames = ['Round', 'Health', 'CashonHand', 'LERemaining', 'LEEarned']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in output:
            if row[0].period < 19:  # Only include actual game rounds
                writer.writerow({'Round': row[0].period, 'Health': row[0].health, 
                               'CashonHand': row[0].cash, 'LERemaining': row[1], 
                               'LEEarned': row[2]})

if __name__ == "__main__":
    main()