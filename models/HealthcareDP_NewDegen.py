"""
Healthcare Dynamic Programming Model with Alternative Degeneration/Regeneration

This program implements a dynamic programming solution for an experimental healthcare
decision-making game that uses alternative mathematical formulations for health
regeneration, degeneration, and life enjoyment functions.

Game mechanics:
- Players start with a health score (typically 85)
- Each round, players earn money proportional to their health
- Health decreases each round at a fixed rate, with larger drops on specific rounds
- Players can invest in health regeneration, spend on life enjoyment, or save money
- The objective is to maximize total life enjoyment over all rounds

Key differences from standard model:
- Different health regeneration formula that scales with current health deficit
- Different life enjoyment formula with hyperbolic scaling
- Special health shock rounds (rounds 3 and 9)
- Different investment enumeration with fixed banking increments
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
    Alternative health regeneration strategy that scales with health deficit.
    
    This model uses a different formula than the standard sigmoid curve,
    basing health gains on: (1) the current health deficit and 
    (2) a ratio of investment vs investment+constant.
    
    Parameters:
    - d: Scaling factor for health deficit
    - k: Half-saturation constant for investment
    """
    def __init__(self, d, k):
        self.d = d
        self.k = k

    def HealthRegained(self, investment, health):
        """
        Calculate how much health is regained from a given investment.
        
        Parameters:
        - investment: Amount of money invested in health
        - health: Current health level
        
        Returns:
        - Integer amount of health points regained
        
        Formula: health_gain = (100-health) * (investment-d*(1-(health/100)))/(investment+k)
        - Scales with health deficit (100-health)
        - Reduced by factor d*(1-(health/100)) for low health
        - Hyperbolic scaling with investment/(investment+k)
        """
        if investment <= 0:
            return 0
        # Calculate health gain based on health deficit and investment ratio
        regain = int((100-health)*((investment-self.d*(1-(health/100)))/(investment+self.k)))
        return max(regain, 0)  # Ensure non-negative health gain

class LifeEnjoymentStrategy:
    """
    Alternative life enjoyment strategy with hyperbolic scaling.
    
    This model uses a simpler formula than the standard model,
    using a hyperbolic function of investment scaled by health.
    
    Parameters:
    - j: Half-saturation constant for enjoyment investment
    """
    def __init__(self, j):
        self.j = j

    def LifeEnjoyment(self, investment, currentHealth):
        """
        Calculate the life enjoyment gained from a given investment.
        
        Parameters:
        - investment: Amount invested in life enjoyment
        - currentHealth: Current health level
        
        Returns:
        - Life enjoyment score gained
        
        Formula: enjoyment = currentHealth * (investment/(investment+j))
        - Scales linearly with health
        - Hyperbolic scaling with investment
        """
        if investment <= 0:
            return 0
        # Calculate enjoyment as fraction of health based on investment ratio
        enjoy = currentHealth*(investment/(investment+self.j))
        return enjoy

class DegenerationStrategy:
    """
    Alternative degeneration strategy with special rounds.
    
    This model uses a fixed degeneration rate with special
    rounds (3 and 9) that have larger health drops.
    
    Parameters:
    - degen: Base health degeneration per round
    """
    def __init__(self, degen):
        self.degen = degen

    def HealthDegeneration(self, currentHealth, currentRound):
        """
        Calculate health after degeneration for a given round.
        
        This model features special rounds (3 and 9) with larger health drops.
        
        Parameters:
        - currentHealth: Current health level
        - currentRound: Current round number
        
        Returns:
        - Health after degeneration (never below 0)
        """
        # Special rounds with increased degeneration (+50)
        if currentRound == 3 or currentRound == 9:
            return max(currentHealth - (self.degen+50), 0)
        else:
            return max(currentHealth - self.degen, 0)

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
        
        Parameters:
        - currentHealth: Current health level
        
        Returns:
        - Integer amount of money earned
        """
        return int(round(self.maxHarvest * currentHealth / 100))

# Named tuples for state representation
DPState = collections.namedtuple('DPState', 'period, health, cash')
Investment = collections.namedtuple('Investment', 'healthExpenditure, lifeExpenditure, cashRemaining')

class HealthCareDP:
    """
    Main dynamic programming class that finds optimal strategies for the healthcare game
    with alternative regeneration and degeneration models.
    
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
    
    def HealthDegeneration(self, currentHealth, currentRound):
        """
        Calculate health after degeneration, ensuring health doesn't go below 0.
        
        Parameters:
        - currentHealth: Current health level
        - currentRound: Current round number
        
        Returns:
        - Health after degeneration
        """
        return self.degenStrat.HealthDegeneration(currentHealth, currentRound)
    
    def HealthRegained(self, investment, health):
        """
        Calculate health gained from investment, ensuring it's not negative.
        
        Parameters:
        - investment: Amount invested in health
        - health: Current health level
        
        Returns:
        - Health points gained
        """
        return max(self.regenStrat.HealthRegained(investment, health), 0)

    def LifeEnjoyment(self, investment, currentHealth):
        """
        Calculate life enjoyment gained from investment.
        
        Parameters:
        - investment: Amount invested in life enjoyment
        - currentHealth: Current health level
        
        Returns:
        - Life enjoyment score gained
        """
        return self.enjoymentStrat.LifeEnjoyment(investment, currentHealth)
        
    def Transition(self, state):
        """
        Transition function: Move from current state to next state before investments.
        
        This simulates the passage of one round:
        - Period increases by 1
        - Health degenerates according to the degeneration strategy
        - Cash increases by the harvest amount based on current health
        
        Parameters:
        - state: Current state (DPState)
        
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
        # Calculate new health after regeneration, capped at 100
        endHealth = min(100, state.health + self.regenStrat.HealthRegained(investment.healthExpenditure, state.health))
        
        # Create new state with updated health and cash
        newState = DPState(state.period,
                          endHealth,
                          investment.cashRemaining)
        
        # Calculate life enjoyment gained from investment
        enjoyment = self.enjoymentStrat.LifeEnjoyment(investment.lifeExpenditure, endHealth)
        
        return (newState, enjoyment)
    
    def StateEnum(self, state):
        """
        Generate all possible state transitions from the current state.
        
        This function enumerates all possible investment decisions and their resulting states.
        
        Parameters:
        - state: Current state (DPState)
        
        Returns:
        - List of tuples (new state, life enjoyment gained) for all possible investments
        """
        investments = self.InvestmentEnum(state.cash, state.health)
        allStateEnjoyments = []
        for investment in investments:
            newStateEnjoyment = self.Invest(state, investment)
            if newStateEnjoyment not in allStateEnjoyments:
                allStateEnjoyments.append(newStateEnjoyment)
        return allStateEnjoyments
        
    def InvestmentEnum(self, cash, health):
        """
        Generate all possible investment decisions for a given amount of cash and health.
        
        This alternative enumeration approach:
        1. Uses fixed increments for banking (every 10 units)
        2. Only considers health investments that would be beneficial given current health
        
        Parameters:
        - cash: Amount of cash available for investment
        - health: Current health level
        
        Returns:
        - List of Investment tuples representing all possible investment decisions
        """
        potentialStates = []
        
        # Use cache if available for this cash amount
        if str(cash) in self.EnumCache:
            return self.EnumCache[str(cash)]
        
        # Enumerate all possible health expenditures
        for healthExpenditure in range(cash + 1):
            # Only consider banking in fixed increments of 10
            # This reduces state space but maintains reasonable granularity
            for bankedCash in range(0, min(111, cash-healthExpenditure+1), 10):
                # Remaining cash goes to life enjoyment
                lifeExpenditure = cash - healthExpenditure - bankedCash
                
                # Create investment tuple and add to potential states
                potentialStates.append(Investment(healthExpenditure, lifeExpenditure, bankedCash))
                
        # Cache and return the potential states
        self.EnumCache[str(cash)] = potentialStates
        return potentialStates
    
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
        # Generate next state (after degeneration and harvest)
        newState = self.Transition(currentState)
        
        # Check if game is over or state is already cached
        if newState.period > self.numRounds or newState.health <= 0:
            return (newState, 0, 0)
        elif newState in self.cache:
            return self.cache[newState]
        else: 
            # Enumerate all possible states from current state
            stateEnjoyments = self.StateEnum(newState)
            
            # Initialize with placeholder for highest return
            highestReturn = (DPState(0, 0, 0), 0, 0)
            
            # Find state that maximizes total future enjoyment
            for (state, enjoyment) in stateEnjoyments:
                future = self.Solve(state)[1]  # Get future value from optimal decision
                totalValue = enjoyment + future
                if totalValue > highestReturn[1]:
                    highestReturn = (state, totalValue, round(enjoyment, 1))
            
            # Cache the result for this state
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
        
        # Calculate optimal strategies at each point
        for i in strategy[:-1]:
            alternate.append(self.Solve(i[0]))
        
        # Calculate percentage losses
        for i in range(len(strategy[:-2])):
            losses.append(((alternate[i+1][1] + (strategy[i+1][1] - strategy[i+2][1])) - alternate[i][1]) / float(alternate[0][1]))
        
        # Prepare output data
        output = []
        for i in range(len(alternate) - 1):
            output.append([alternate[i], strategy[i+1][0], (alternate[i+1][1] + (strategy[i+1][1] - strategy[i+2][1])), 
                          losses[i], np.cumsum(losses[:i+1])[i], strategy[i], (strategy[i][1] - strategy[i+1][1])])    
        
        # Write analysis to CSV file
        with open(outfile, 'a+', newline='') as f:
            fieldnames = ['ID', 'Lifetime', 'Period', 'Optimal Health', 'Optimal Cash on Hand', 'Remaining Max',
                          'Optimal Earnings This Period', 'Realized Health', 'Realized Cash on Hand', 'Current LE',
                          'Earned This Period', 'Remaining Available', '% Loss', 'Accumulated Loss']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            for row in output:
               writer.writerow({'ID': ID, 'Lifetime': life, 'Period': row[0][0].period, 'Optimal Health': row[0][0].health, 
                               'Optimal Cash on Hand': row[0][0].cash, 'Remaining Max': int(row[0][1]),
                               'Optimal Earnings This Period': row[0][2], 'Realized Health': row[5][0].health, 
                               'Realized Cash on Hand': row[5][0].cash, 'Current LE': row[5][1], 
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
    # Create output file with headers
    with open(outfile, 'w+', newline='') as f:
        fieldnames = ['ID', 'Lifetime', 'Period', 'Optimal Health', 'Optimal Cash on Hand', 'Remaining Max',
                      'Optimal Earnings This Period', 'Realized Health', 'Realized Cash on Hand', 'Current LE',
                      'Earned This Period', 'Remaining Available', '% Loss', 'Accumulated Loss']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
    
    # Process each game in the dataset
    for i in data:
        total = i[-1][2][1]  # Final cumulative life enjoyment
        
        # Extract round data and add terminal state
        pad = [k[2] for k in i] + [[[19, 0, 0], 0]]
        
        # Calculate life enjoyment for each round
        for j in range(len(pad)):
            pad[j][1] = max(pad[-2][1] - pad[j][1], 0)
        
        # Format data for analysis
        pad = [[startState, total]] + pad
        pad = [[DPState(val[0][0], val[0][1], val[0][2]), val[1]] for val in pad]
        
        # Analyze this player's strategy
        HCDP.AnalyzeStrat(pad, i[0][0], i[0][1], outfile)

def main():
    """
    Main function to run the HealthCareDP program with alternative models.
    
    This function:
    1. Reads parameter file
    2. Initializes strategy objects with alternative formulations
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

    # Initialize alternative strategy objects
    regenStrat = RegenerationStrategy(d=10, k=50)  # d=scaling factor, k=half-saturation constant
    enjoymentStrat = LifeEnjoymentStrategy(j=50)   # j=half-saturation constant for enjoyment
    degenStrat = DegenerationStrategy(degen=10)    # Fixed degeneration with special rounds
    harvestStrat = HarvestStrategy(maxHarvest=46.811)
    
    # Set starting state
    startState = params[0] = DPState(params[0][0], params[0][1], params[0][2])

    # Create DP solver
    HCDP = HealthCareDP(startState, params[1], regenStrat, enjoymentStrat, degenStrat, harvestStrat)
          
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
    # BatchRun(readInFile('EighteenRound_inFile.txt', 18)[:10], startState, HCDP, 'EighteenRoundOut_NewDegen.csv')

    # Write output to CSV
    outputfilename = f'output_NewDegen_{fileName[:-4]}.csv'
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