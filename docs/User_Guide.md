# Healthcare Dynamic Programming Model - User Guide

This guide provides detailed instructions for using the Healthcare Dynamic Programming model to analyze optimal strategies in the healthcare decision-making game and compare them with actual player behaviors.

## Project Workflow

The complete workflow for this project involves several steps:

1. **Collect Experimental Data**:
   - Run experiments with human participants
   - Record decisions and outcomes in CSV format

2. **Process Raw Data**:
   - Use DataCleaner.py to convert raw data to analysis format
   - Filter for specific experimental conditions
   - Generate NineRound_inFile.txt and EighteenRound_inFile.txt

3. **Calculate Optimal Strategies**:
   - Use HealthcareDP.py or HealthcareDP_Stoch.py with parameter files
   - Generate optimal strategies via dynamic programming

4. **Compare Actual vs. Optimal**:
   - Analyze how actual player strategies deviate from optimal
   - Calculate economic losses from suboptimal decisions

## Model Components

The Healthcare DP model consists of five key components:

1. **Main Program (HealthcareDP.py/HealthcareDP_Stoch.py)**: Implements the dynamic programming algorithm
2. **RegenerationStrategy**: Controls health gains from investments
3. **DegenerationStrategy**: Controls health decline over rounds
4. **LifeEnjoymentStrategy**: Calculates score from enjoyment investments
5. **HarvestStrategy**: Determines money earned each round

## Data Processing

### Using DataCleaner.py

```bash
python DataCleaner.py
```

This script:
1. Reads raw experimental data from CSV files
2. Filters for specific experimental conditions
3. Reformats data into the structure needed for analysis
4. Outputs NineRound_inFile.txt and EighteenRound_inFile.txt

### Customizing Data Processing

To modify the filtering conditions, edit the following section in DataCleaner.py:

```python
shortRound = [
    formattedOutput[j:j+9] for j in xrange(0, len(formattedOutput), 9) 
    if formattedOutput[j][7] == 9        # 9-round games
    if formattedOutput[j][3] == 1        # flat = 1
    if formattedOutput[j][4] == 0        # social.life = 0
    if formattedOutput[j][5] == 0        # social.health = 0
    if formattedOutput[j][6] == 0        # retirement = 0
]
```

## Running the Models

### Standard Model

```bash
python HealthcareDP.py
```

When prompted, enter the name of your parameter file (which should be in the same directory).

### Stochastic Model

```bash
python HealthcareDP_Stoch.py
```

The stochastic model includes random health shocks to test more robust strategies. It calculates expected values based on the probability and magnitude of health shocks.

To customize the shock parameters, modify these lines in HealthcareDP_Stoch.py:

```python
stochHitChance = 0.2  # 20% chance of health shock each round
stochHitSize = 50     # Lose 50 health points if shock occurs
```

## Parameter Files

### Creating Parameter Files

Create a text file with the following format:

```
# Parameters
# Add any header information here

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

### Parameter Explanations

#### Regeneration Parameters

- **gamma**: Maximum possible health gain (typically 100)
- **sigma**: Controls how quickly the health gain reaches its maximum (0.01 is standard)
- **r**: Investment amount at which 50% of maximum health gain is achieved (250 is standard)

Higher values of sigma make the curve steeper, while higher values of r shift the curve to the right.

#### Life Enjoyment Parameters

- **alpha**: Controls how quickly enjoyment reaches its maximum (0.01 is standard)
- **beta**: Weight given to the health component (0.7 is standard)
- **mu**: Base enjoyment independent of health (0.2 is standard)
- **c**: Overall scaling factor (50 is standard)

#### Degeneration Parameters

- **intercept**: Base health loss per round (e.g., 7.625 for 18-round games, 15 for 9-round games)
- **slope**: Additional health loss per round (e.g., 0.25 for 18-round games, 1 for 9-round games)
- **horizon**: Round number that triggers increased degeneration (optional)

#### Harvest Parameters

- **maxHarvest**: Maximum money earned at 100 health (e.g., 46.811 for 18-round games, 93.622 for 9-round games)

## Understanding Output

The models produce several types of output:

### Console Output

Shows each optimal state and the expected total life enjoyment:

```
(DPState(period=1, health=85, cash=222), 1234.5, 222.0)
```

This represents:
- The state (period, health, cash)
- Total expected life enjoyment from this point forward
- Life enjoyment earned in this period

### CSV Output

When using the standard output, a CSV file is created with the following columns:

- **Round**: The round number
- **Health**: Optimal health level after investments
- **CashonHand**: Optimal cash remaining after investments
- **LERemaining**: Expected remaining life enjoyment
- **LEEarned**: Life enjoyment earned in this round

### Analysis Output

When using BatchRun to analyze player strategies, a more detailed CSV is produced:

- Player ID and game ID
- Period (round number)
- Optimal health and cash on hand
- Remaining maximum possible life enjoyment
- Optimal earnings for the period
- Realized health and cash on hand (player's actual choices)
- Current life enjoyment
- Earned this period
- Remaining available life enjoyment
- Percentage loss (compared to optimal)
- Accumulated loss

## Analyzing Player Strategies

To compare player strategies with the optimal strategy:

1. Ensure your processed data files (NineRound_inFile.txt, EighteenRound_inFile.txt) are in the same directory
2. In the main function, uncomment and modify the BatchRun line:

```python
BatchRun(readInFile('EighteenRound_inFile.txt', 18)[:20], startState, HCDP, 'EighteenRoundOut.csv')
```

3. Adjust the slice `[:20]` to analyze more or fewer games
4. Run the program as usual

## Customizing the Models

### Different Game Lengths

The model supports games of different lengths:
- For 9-round games: `degenStrat9 = DegenerationStrategy(15, 1, 9)` and `harvestStrat9 = HarvestStrategy(93.622)`
- For 18-round games: `degenStrat18 = DegenerationStrategy(7.625, 0.25, 18)` and `harvestStrat18 = HarvestStrategy(46.811)`

### Modifying Strategy Components

You can modify the mathematical models by editing the strategy classes:

1. **DegenerationStrategy.py**: Change how health declines over time
2. **RegenerationStrategy.py**: Modify how investments translate to health gains
3. **LifeEnjoymentStrategy.py**: Adjust how investments create enjoyment score
4. **HarvestStrategy.py**: Change how money is earned based on health

## Advanced Usage

### Performance Optimization

The model uses caching to avoid redundant calculations. Additional optimizations:

1. **Reduce state space**: Modify `InvestmentEnum()` to consider fewer possible investments
2. **Pre-calculate values**: Use lookup tables for frequently calculated values

### Debugging Tips

1. **Print state transitions**: Add print statements in the `Transition()` method
2. **Inspect cache growth**: Monitor the size of the `cache` dictionary
3. **Verify strategy correctness**: Check that the optimal path follows expected patterns

## Example Workflow

1. Process experimental data:
   ```
   python DataCleaner.py
   ```

2. Calculate optimal strategies for 18-round games:
   ```
   python HealthcareDP.py
   ```
   Enter: `test18.txt`

3. Compare optimal vs. actual player strategies:
   - Uncomment BatchRun in the main function
   - Re-run the program
   - Analyze the output CSV file

4. Test robustness with stochastic model:
   ```
   python HealthcareDP_Stoch.py
   ```
   Enter: `test18.txt`

5. Compare stochastic vs. standard strategies to assess the impact of health shocks

## Troubleshooting

### Common Issues

1. **File Not Found Errors**:
   - Make sure all files are in the current directory
   - Check file names for typos

2. **Parameter Errors**:
   - Verify parameter file format
   - Ensure all required parameters are present

3. **Memory Issues**:
   - For very large state spaces, you may need more memory
   - Consider reducing the precision of cash values or limiting enumeration

4. **Data Format Issues**:
   - If analyzing new experimental data, verify the CSV format matches expectations
   - Adjust DataCleaner.py if necessary