# Healthcare Dynamic Programming - Version Comparison

This document outlines the evolution of the Healthcare Dynamic Programming model through its different versions, highlighting key features, improvements, and differences between versions.

## Overview of Versions

| Version | Filename | Key Features |
|---------|----------|-------------|
| 1.0 | HealthcareDP_1.0.py | Initial implementation, all functionality in one class |
| 1.1 | HealthcareDP_1.1.py | Refactored with named tuples for state representation |
| 1.2 | HealthcareDP_1.2.py | Strategy classes extracted as separate components |
| 3.0 | HealthcareDP_3.0.py | Enhanced degeneration model with horizon parameter |
| Stoch | HealthcareDP_Stoch.py | Added stochastic health shocks |
| NewDegen | HealthcareDP_NewDegen.py | Alternative degeneration and regeneration models |

## Detailed Version Comparison

### Version 1.0 (HealthcareDP_1.0.py)

**Structure:**
- All functionality contained within a single `HealthCareDP` class
- States represented as basic lists `[period, health, cash]`
- Direct implementation of mathematical functions within the class

**Key Features:**
- Basic dynamic programming implementation
- Round-down function for state space reduction
- Simple analysis of strategies

**Limitations:**
- Less modular code structure
- Limited flexibility for parameter variations
- No separate strategy components

### Version 1.1 (HealthcareDP_1.1.py)

**Improvements:**
- Introduction of named tuples (`DPState` and `Investment`) for better state representation
- Enhanced strategy analysis with CSV output
- Improved batch processing capability

**Key Changes:**
- More structured state representation
- Better organization of the dynamic programming algorithm
- Enhanced output formats

### Version 1.2 (HealthcareDP_1.2.py)

**Improvements:**
- Strategy classes extracted as separate components:
  - `RegenerationStrategy`
  - `LifeEnjoymentStrategy`
  - `DegenerationStrategy`
  - `HarvestStrategy`
- Improved modularity and flexibility
- Enhanced batch processing for player data analysis

**Key Changes:**
- More modular design with strategy pattern
- Better separation of concerns
- Improved reusability of components

### Version 3.0 (HealthcareDP_3.0.py)

**Improvements:**
- Enhanced `DegenerationStrategy` with horizon parameter
- Two-phase degeneration model: normal degeneration before horizon, accelerated after
- More flexible parameter handling

**Key Changes:**
- Adjusted health degeneration model
- Better handling of different game lengths
- Improved strategy analysis capabilities

### Stochastic Version (HealthcareDP_Stoch.py)

**New Features:**
- Added stochastic health shocks with configurable probability and magnitude
- Dual strategy handling: standard and post-shock
- Modified `Solve` function to account for expected values under uncertainty

**Key Changes:**
- Incorporates randomness into the model
- Calculates expected value strategies
- More complex state evaluation using weighted outcomes
- Handles both normal and "hit state" (post-shock) scenarios
- Uses external strategy modules 

### NewDegen Version (HealthcareDP_NewDegen.py)

**Alternative Models:**
- Different regeneration formula:
  ```python
  regain = int((100-health)*((investment-self.d*(1-(health/100)))/(investment+self.k)))
  ```
- Different enjoyment formula:
  ```python
  enjoy = currentHealth*(investment/(investment+self.j))
  ```
- Modified degeneration with special rounds:
  ```python
  if currentRound == 3 or currentRound == 9:
      return max(currentHealth - (self.degen+50),0)
  else:
      return max(currentHealth - self.degen,0)
  ```

**Key Changes:**
- Alternative mathematical models
- Different investment enumeration approach
- Special health shock rounds

## Key Algorithmic Differences

### State Representation

- **1.0**: Simple lists `[period, health, cash]`
- **1.1+**: Named tuples `DPState(period, health, cash)`

### Investment Enumeration

- **1.0**: Round-down technique for state space reduction
  ```python
  potentialStates.append([i, cash-i-j, round_down(j, 10)])
  ```
- **1.2+**: Efficiency improvement by only considering unique health gains
  ```python
  if not hr == prev:
      prev = hr
      for lifeExpenditure in range(...):
  ```
- **NewDegen**: Different enumeration with fixed cash increments
  ```python
  for bankedCash in range(0, min(111, cash-healthExpenditure+1), 10):
  ```

### Degeneration Models

- **1.0/1.1/1.2**: Simple linear model
  ```python
  return currentHealth - int(round(self.intercept + (self.slope * currentRound)))
  ```
- **3.0**: Two-phase model with horizon
  ```python
  if currentRound <= self.horizon:
      return max(currentHealth - int(round(self.intercept + (self.slope * currentRound))), 0)
  else:
      return max(currentHealth - int(round(self.intercept + (2 * self.slope * currentRound))), 0)
  ```
- **NewDegen**: Special rounds model
  ```python
  if currentRound == 3 or currentRound == 9:
      return max(currentHealth - (self.degen+50), 0)
  else:
      return max(currentHealth - self.degen, 0)
  ```

### Solving Algorithm

- **1.0/1.1/1.2/3.0**: Standard dynamic programming with caching
- **Stoch**: Extended to handle stochastic events
  ```python
  self.cache[newState] = (highestReturn, hitStateHighestReturn, 
                          (1-self.stochHitChance)*highestReturn[1] + 
                          self.stochHitChance*hitStateHighestReturn[1])
  ```

## Data Processing Evolution

As the model evolved, so did the data processing pipeline:

1. **Early Versions**: Manual data preparation
2. **Later Versions**: Implemented DataCleaner.py to automate processing
3. **Final Workflow**: Complete pipeline from raw CSV to analysis

## Recommended Versions

For different use cases:

- **Standard Analysis**: Version 3.0 (HealthcareDP_3.0.py)
  - Best balance of modular code structure
  - Realistic health degeneration model
  - Comprehensive strategy analysis

- **Shocks Analysis**: Stochastic Version (HealthcareDP_Stoch.py)
  - Includes random health shocks
  - Models unexpected health events
  - Produces more conservative strategies

- **Alternative Model**: NewDegen Version (HealthcareDP_NewDegen.py)
  - Different mathematical formulations
  - Special rounds with larger health shocks
  - Useful for comparative analysis
