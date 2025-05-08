"""
Healthcare Dynamic Programming Model with Stochastic Health Shocks

This program implements a dynamic programming solution for an experimental healthcare
decision-making game that includes random health shocks. Participants manage health 
and financial resources over multiple rounds to maximize their "life enjoyment" score
while accounting for the possibility of unexpected health declines.

Author: Original by John, documentation added later

Game mechanics:
- Players start with a health score (typically 85)
- Each round, players earn money proportional to their health
- Health decreases each round at an increasing rate
- There is a chance of a random health shock each round
- Players can invest in health regeneration, spend on life enjoyment, or save money
- The objective is to maximize total life enjoyment over all rounds
"""

import csv
import collections
import time
import json
import numpy as np

# Import strategy modules
import strategies.DegenerationStrategy as degen
import strategies.HarvestStrategy as harvest
import strategies.LifeEnjoymentStrategy as le
import strategies.RegenerationStrategy as regen

# Named tuples for state representation
DPState = collections.namedtuple('DPState', 'period, health, cash')
Investment = collections.namedtuple('Investment', 'healthExpenditure, lifeExpenditure, cashRemaining')

class HealthCareDP:
    """
    Main dynamic programming class that finds optimal strategies for the healthcare game
    with stochastic health shocks.
    
    This class implements the core algorithm to explore all possible states and determine
    the optimal decision at each state to maximize expected total life enjoyment,
    considering the possibility of random health shocks.
    """
    
    def __init__(self, state, numRounds, regenStrat, enjoymentStrat, degenStrat, harvestStrat, stochHitChance, stochHitSize):
        """
        Initialize the dynamic programming solver with stochastic components.
        
        Parameters:
        - state: Starting state (DPState tuple)
        - numRounds: Total number of rounds in the game
        - regenStrat: RegenerationStrategy instance
        - enjoymentStrat: LifeEnjoymentStrategy instance
        - degenStrat: DegenerationStrategy instance
        - harvestStrat: HarvestStrategy instance
        - stochHitChance: Probability of a health shock occurring (0.0 to 1.0)
        - stochHitSize: Magnitude of health reduction if a shock occurs
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
        self.stochHitChance = stochHitChance  # Probability of health shock
        self.stochHitSize = stochHitSize  # Magnitude of health shock
    
    def HealthDegeneration(self, currentHealth, currentRound):
        """
        Calculate health degeneration, ensuring health doesn't go below 0.
        
        Parameters:
        - currentHealth: Current health points
        - currentRound: Current round number
        
        Returns:
        - Remaining health after degeneration (never below 0)
        """
        return max(self.degenStrat.HealthDegeneration(currentHealth, currentRound), 0)
    
    def HealthRegained(self, investment):
        """
        Calculate health gained from investment, ensuring it's not negative.
        
        Parameters:
        - investment: Amount invested in health
        
        Returns:
        - Health points gained (never below 0)
        """
        return max(self.regenStrat.HealthRegained(investment), 0)

    def LifeEnjoyment(self, investment, currentHealth):
        """
        Calculate life enjoyment gained from investment.
        
        Parameters:
        - investment: Amount invested in life enjoyment
        - currentHealth: Current health points
        
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
        - state: Current state (DPState or a tuple with state in index 0)
        
        Returns:
        - New DPState tuple after transition
        """
        try:
            # Handle case where state is a tuple with state in index 0
            nextPeriod = state[0].period + 1
            return DPState(nextPeriod,
                           self.degenStrat.HealthDegeneration(state[0].health, nextPeriod),
                           state[0].cash + self.harvestStrat.HarvestAmount(state[0].health))
        except AttributeError:
            # Handle case where state is directly a DPState
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
        Core dynamic programming function to find the optimal strategy with stochastic shocks.
        
        This recursive function:
        1. Transitions to the next state
        2. Calculates a "hit state" representing what happens if a health shock occurs
        3. Checks if the game is over or if the state is already cached
        4. Enumerates all possible investment decisions for both normal and hit states
        5. Recursively finds the value of each resulting state
        6. Returns the weighted average value based on shock probability
        
        Parameters:
        - currentState: Current state (DPState)
        
        Returns:
        - Tuple of (optimal next state, total life enjoyment with shock, total life enjoyment without shock)
        """
        try:
            # Generate next state and potential "hit state" (after health shock)
            newState = self.Transition(currentState)
            hitState = DPState(newState[0], max(newState[1] - self.stochHitSize, 0), newState[2])
        except AttributeError:
            # Alternative handling if currentState format is different
            newState = self.Transition(currentState)
            hitState = DPState(newState[0], max(newState[1] - self.stochHitSize, 0), newState[2])
        
        # Check if game is over
        if newState.period > self.numRounds or newState.health <= 0:
            return (newState, 0, 0)
        # Check if state is already cached
        elif newState in self.cache:
            return self.cache[newState]
        else: 
            # Enumerate all possible states from normal and hit states
            stateEnjoyments = self.StateEnum(newState)
            hitStateEnjoyments = self.StateEnum(hitState)
            
            # Find optimal strategy for normal state
            highestReturn = DPState(0, 0, -1)
            for (state, enjoyment) in stateEnjoyments:
                future = self.Solve(state)[2]  # Expected future value
                totalValue = enjoyment + future
                if totalValue > highestReturn[1]:
                    highestReturn = (state, totalValue, round(enjoyment, 1))
            
            # Find optimal strategy for hit state
            hitStateHighestReturn = DPState(0, 0, -1)
            for (state, enjoyment) in hitStateEnjoyments:
                future = self.Solve(state)[2]  # Expected future value
                totalValue = enjoyment + future
                if totalValue > hitStateHighestReturn[1]:
                    hitStateHighestReturn = (state, totalValue, round(enjoyment, 1))
            
            # Calculate expected value based on probability of shock
            expected_value = (1 - self.stochHitChance) * highestReturn[1] + self.stochHitChance * hitStateHighestReturn[1]
            
            # Cache the results
            self.cache[newState] = (highestReturn, hitStateHighestReturn, expected_value)
        
        return (highestReturn, hitStateHighestReturn, expected_value)
    
    def FindStrat(self, state):
        """
        Generate the optimal path through the state space from the given starting state.
        
        This considers the probability of health shocks when determining optimal actions.
        
        Parameters:
        - state: Starting state (DPState)
        
        Returns:
        - List of states representing the optimal path
        """
        cur = state
        strategy = []
        for i in range(int(self.numRounds)):
            try:
                # Handle different return formats from Solve
                cur = self.Solve(cur)[0][0]
            except:
                cur = self.Solve(cur)
            strategy.append(cur[0][0])
        return strategy
    
    def AnalyzeStrat(self, strategy, ID, life, outfile):
        """
        Analyze how an actual strategy deviates from the optimal strategy.
        
        This function compares actual player decisions against optimal decisions and
        calculates the percentage loss in life enjoyment, considering stochastic effects.
        
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
                writer.writerow({
                    'ID': ID,
                    'Lifetime': life,
                    'Period': row[0][0][0].period,
                    'Optimal Health': row[0][0][0].health,
                    'Optimal Cash on Hand': row[0][0][0].cash, 
                    'Remaining Max': int(row[0][1]),
                    'Optimal Earnings This Period': row[0][2],
                    'Realized Health': row[5][0].health,
                    'Realized Cash on Hand': row[5][0].cash,
                    'Current LE': row[5][1],
                    'Earned This Period': row[6],
                    'Remaining Available': int(row[2]),
                    '% Loss': '%.3f' % row[3],
                    'Accumulated Loss': '%.3f' % row[4]
                })

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
        f.close()
    
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
    Main function to run the Stochastic HealthCareDP program.
    
    This function:
    1. Reads parameter file
    2. Initializes strategy objects
    3. Sets up stochastic shock parameters
    4. Creates the DP solver
    5. Finds the optimal strategy
    6. Optionally runs batch analysis on player data
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

    # Set stochastic parameters
    stochHitChance = 0.2  # 20% chance of health shock each round
    stochHitSize = 50     # Lose 50 health points if shock occurs

    # Initialize strategy objects
    regenStrat = regen.RegenerationStrategy(params[2], params[3], params[4])
    enjoymentStrat = le.LifeEnjoymentStrategy(params[5], params[6], params[7], params[8])
    
    # Create different strategies for 9-round and 18-round games
    degenStrat18 = degen.DegenerationStrategy(7.625, 0.25)
    degenStrat9 = degen.DegenerationStrategy(15, 1)
    harvestStrat18 = harvest.HarvestStrategy(46.811 / 0.93622)
    harvestStrat9 = harvest.HarvestStrategy(93.622)
    
    # Set starting state
    startState = params[0] = DPState(params[0][0], params[0][1], params[0][2])

    # Create stochastic DP solver (using 18-round parameters by default)
    HCDP = HealthCareDP(startState, params[1], regenStrat, enjoymentStrat, 
                        degenStrat18, harvestStrat18, stochHitChance, stochHitSize)
          
    # Find optimal strategy and measure execution time
    start = time.time()
    output = []
    for val in [startState] + HCDP.FindStrat(startState):
        o = HCDP.Solve(val)
        output.append(o)
        print(o)
        
    end = time.time()
    print(f"Execution time: {end - start} seconds")
    
    # batch analysis on player data
    BatchRun(readInFile('EighteenRound_inFile.txt', 18)[:20], startState, HCDP, 'EighteenRoundOut.csv')

    # Write output to CSV
    outputfilename = f'analysis\\output_{fileName[:-4]}.csv'
    with open(outputfilename, 'w+', newline='') as f:
        fieldnames = ['Round', 'Health', 'CashonHand', 'LERemaining', 'LEEarned']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in output:
            if row[0][0].period < 19:  # Only include actual game rounds
                writer.writerow({
                    'Round': row[0][0].period,
                    'Health': row[0][0].health, 
                    'CashonHand': row[0][0].cash,
                    'LERemaining': int(row[1]),
                    'LEEarned': int(row[2])
                })

if __name__ == "__main__":
    main()