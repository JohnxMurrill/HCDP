# Healthcare Dynamic Programming Model

## Overview

This project implements a dynamic programming solution for analyzing decision-making in an experimental healthcare game. In this game, participants manage their health and financial resources over multiple rounds (9 or 18) to maximize their "life enjoyment" score while dealing with health degeneration and resource constraints.

The model was developed to analyze optimal strategies and compare them with actual player behaviors from experimental data, providing insights into healthcare decision-making patterns.

## Project Structure

### Core Components

1. **Dynamic Programming Models**
   - `HealthcareDP.py` - Standard model for optimal strategy calculation
   - `HealthcareDP_Stoch.py` - Model with stochastic health shocks

2. **Strategy Modules**
   - `RegenerationStrategy.py` - Controls health gains from investments
   - `DegenerationStrategy.py` - Controls health decline over rounds
   - `LifeEnjoymentStrategy.py` - Calculates score from enjoyment investments
   - `HarvestStrategy.py` - Determines money earned each round

3. **Data Processing**
   - `DataCleaner.py` - Processes raw experimental data into analysis-ready formats

4. **Configuration Files**
   - `test.txt` - Parameters for 9-round games
   - `test18.txt` - Parameters for 18-round games

5. **Input Data Files**
   - `NineRound_inFile.txt` - Processed experimental data for 9-round games
   - `EighteenRound_inFile.txt` - Processed experimental data for 18-round games

6. **Documentation**
   - `README.md` - This file
   - `docs/` - Additional documentation (see Documentation section)

## Game Mechanics

1. **Health**: Players start with a health score (typically 85 out of 100)
2. **Money/Harvest**: Each round, players earn money proportional to their current health
3. **Degeneration**: Health decreases each round at an increasing rate
4. **Decisions**: Players must choose how to:
   - Invest in health regeneration
   - Spend on "life enjoyment" (the score)
   - Save money for future rounds
5. **Objective**: Maximize total life enjoyment over all rounds

## Usage

### Processing Experimental Data

To convert raw experimental data into analysis-ready format:

```
python DataCleaner.py
```

This reads from CSV files containing experimental data and outputs:
- `NineRound_inFile.txt` - For 9-round games
- `EighteenRound_inFile.txt` - For 18-round games

### Running the Standard DP Model

```
python HealthcareDP.py
```

When prompted, enter the parameter filename (e.g., `test.txt` or `test18.txt`).

### Running the Stochastic DP Model

```
python HealthcareDP_Stoch.py
```

When prompted, enter the parameter filename. This model includes random health shocks to test more robust strategies.

### Parameter File Format

Create a text file with the following format:

```
Start State
numRounds
gamma
sigma
r
alpha
beta
mu
c

[1, 85, 0]  # Starting state: [period, health, cash]
18          # Number of rounds
100         # gamma - Maximum health gain parameter
0.01        # sigma - Steepness parameter for health gain
250         # r - Half-point parameter for health gain
0.01        # alpha - Steepness parameter for enjoyment
0.7         # beta - Health coefficient for enjoyment
0.2         # mu - Base enjoyment independent of health
50          # c - Scaling factor for enjoyment
```

## Mathematical Models

### Health Regeneration

Health gained from investment follows a sigmoid function:

```
HealthRegained = gamma * ((1 - exp(-sigma * investment)) / (1 + exp(-sigma * (investment - r))))
```

### Life Enjoyment

Life enjoyment (score) gained from investment:

```
LifeEnjoyment = c * (beta * (health/100) + mu) * (1 - exp(-alpha * investment))
```

### Health Degeneration

Health lost each round:

```
HealthDegeneration = intercept + (slope * round)
```

If the round exceeds a horizon value, the degeneration doubles:

```
HealthDegeneration = intercept + (2 * slope * round)
```

### Money Earned (Harvest)

Money earned each round:

```
HarvestAmount = maxHarvest * (health/100)
```

## Documentation

Additional documentation is available in the `docs/` directory:

- `User_Guide.md` - Detailed instructions for using the models
- `Version_Comparison.md` - History and differences between code versions

## Requirements

- Python 2.7
- Required packages: numpy, math, csv, json

## Research Applications

This code was developed as part of my graduate studies to implement a dynamic programming solution to the experimental game model used in the following publications:

- Bejarano, H., Kaplan, H., Rassenti, S. (2014). Effects of Retirement and Lifetime Earnings Profile on Health Investment. ESI Working Papers, 14-21. https://digitalcommons.chapman.edu/esi_working_papers/5/

- Bejarano, H., Kaplan, H., Rassenti, S. (2015). Dynamic optimization and conformity in health behavior and life enjoyment over the life cycle. Frontiers in Neuroscience, 9, 137. https://www.frontiersin.org/journals/behavioral-neuroscience/articles/10.3389/fnbeh.2015.00137/full

I am not a credited author on these publications, I developed this code to model the experimental games described therein as part of my graduate coursework/research assistantship.