# -*- coding: utf-8 -*-
"""
DataCleaner.py

This script processes raw experimental data from healthcare decision-making experiments
and converts it into the formatted JSON files used by the Dynamic Programming model.

The script reads CSV data from experimental sessions, filters for specific experimental
conditions, and outputs separate files for 9-round and 18-round games.
"""

import itertools
import ast
import json


def getFieldnames(csvFile):
    """
    Read the first row of a CSV file to extract column headers.
    
    Parameters:
    - csvFile: Path to the CSV file
    
    Returns:
    - Tuple of field names from the CSV header
    """
    with open(csvFile) as csvfile:
        firstRow = csvfile.readlines(1)
        fieldnames = tuple(firstRow[0].strip('\n').split(","))
    return fieldnames


def writeCursor(csvFile, fieldnames):
    """
    Convert CSV rows into an array of dictionaries with proper data types.
    
    This function reads each row from the CSV, converts values to appropriate
    data types, and creates a dictionary with selected fields from the experiment.
    
    Parameters:
    - csvFile: Path to the CSV file
    - fieldnames: Tuple of field names from the CSV header
    
    Returns:
    - List of dictionaries, each representing one row of experimental data
    """
    cursor = []  # Placeholder for the dictionaries/documents
    with open(csvFile) as csvFile:
        # Skip the header row and process all remaining rows
        for row in itertools.islice(csvFile, 1, None):
            values = list(row.strip('\n').split("\t"))
            # Convert string values to appropriate data types
            for i, value in enumerate(values):
                nValue = ast.literal_eval(value)
                values[i] = nValue
                wholeLine = dict(zip(fieldnames, values[0]))
            
            # Extract relevant fields for our analysis
            cursor.append({k: wholeLine[k] for k in (
                '"newuniqueid"', '"life"', '"period"', '"health"', 
                '"enjoymentbalance"', '"accountbalance"', '"healthinvestment"',
                '"enjoymentinvestment"', '"flat"', '"social.life"', '"social.health"',
                '"retirement"', '"periods"', '"amountharvested"')})
    return cursor


def constructLifetime(data):
    """
    Restructure the data into the format needed for Dynamic Programming analysis.
    
    This function transforms the dictionaries into a nested structure containing:
    [player_id, game_id, [[round, health, cash], cumulative_enjoyment], 
     flat_condition, social_life, social_health, retirement, periods, harvest_amount]
    
    Parameters:
    - data: List of dictionaries from writeCursor
    
    Returns:
    - Restructured list for analysis
    """
    life = []
    for i in data:
        # Calculate remaining cash after investments
        remaining_cash = i['"accountbalance"'] - i['"healthinvestment"'] - i['"enjoymentinvestment"']
        
        # Create the structured entry
        life.append([
            i['"newuniqueid"'],                     # Player ID
            i['"life"'],                            # Game ID
            [[i['"period"'], i['"health"'], remaining_cash], i['"enjoymentbalance"']],  # Round data + enjoyment
            i['"flat"'],                            # Flat condition
            i['"social.life"'],                     # Social life comparison
            i['"social.health"'],                   # Social health comparison
            i['"retirement"'],                      # Retirement condition
            i['"periods"'],                         # Total periods in game
            i['"amountharvested"']                  # Amount harvested this round
        ])
    return life


def writeout(data, name):
    """
    Write processed data to a JSON file.
    
    Parameters:
    - data: List of processed data to write
    - name: Base name for the output file
    """
    outfile = '{}_inFile.txt'.format(name)
    with open(outfile, 'w') as outfile:
        json.dump(data, outfile)


def main():
    """
    Main function to process experimental data and generate input files.
    
    This function:
    1. Loads and processes experimental data
    2. Sorts data by player, game, and round
    3. Filters for specific experimental conditions
    4. Separates 9-round and 18-round games
    5. Generates output files for each game type
    """
    # Input file with raw experimental data
    input_file = 'experimentaldata_Session1-47_2016-05-09.csv'
    
    # Process the data
    fieldnames = getFieldnames(input_file)
    formattedOutput = constructLifetime(writeCursor(input_file, fieldnames))
    
    # Sort by player ID, game ID, and round
    formattedOutput.sort(key=lambda x: (x[0], x[1], x[2]))
    
    # Filter for 9-round games with specific conditions
    # (flat=1, social.life=0, social.health=0, retirement=0)
    shortRound = [
        formattedOutput[j:j+9] for j in range(0, len(formattedOutput), 9) 
        if formattedOutput[j][7] == 9        # 9-round games
        if formattedOutput[j][3] == 1        # flat = 1
        if formattedOutput[j][4] == 0        # social.life = 0
        if formattedOutput[j][5] == 0        # social.health = 0
        if formattedOutput[j][6] == 0        # retirement = 0
    ]
    
    # Filter for 18-round games with specific conditions
    longRound = [
        formattedOutput[j:j+18] for j in range(0, len(formattedOutput), 18) 
        if formattedOutput[j][7] == 18       # 18-round games
        if formattedOutput[j][3] == 1        # flat = 1
        if formattedOutput[j][4] == 0        # social.life = 0
        if formattedOutput[j][5] == 0        # social.health = 0
        if formattedOutput[j][6] == 0        # retirement = 0
    ]
    
    # Combine all filtered data
    groupedOutput = shortRound + longRound
    
    # Extract core data for output files (player ID, game ID, round data with enjoyment)
    shortRoundOut = [x[j][:3] for x in shortRound for j in range(len(x))]
    longRoundOut = [x[j][:3] for x in longRound for j in range(len(x))]
    
    # Generate output files
    writeout(shortRoundOut, 'NineRound')
    writeout(longRoundOut, 'EighteenRound')
    
    # Print summary information
    print("Total filtered games:", len(groupedOutput))
    print("Sample game data:", groupedOutput[0])


if __name__ == "__main__":
    main()