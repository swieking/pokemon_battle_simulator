# Pokemon Battle Simulator

This repository contains a Python-based Pokémon Battle Simulator. The simulator is capable of running a series of battles between different Pokemon, assessing their strengths and weaknesses against one another. It includes mechanisms for simulating attack moves, effects, and accuracy, as well as generating a new generation of Pokémon by breeding the top performers.

## How it Works

The simulator uses a dataset of Pokemon and their stats, along with a dataset of Pokemon moves and their corresponding effects. It simulates battles between two Pokemon, calculating the effects of each move and their corresponding accuracy. The battles are based on the traditional Pokemon battling system, with the addition of randomized effects. 

After a series of battles, the top-performing Pokemon are selected to "breed" and generate a new generation of Pokemon. This mimics the concept of evolution, with the strongest traits being passed down to the next generation.

## Components

The main components of this simulator are a set of Python functions that simulate the Pokemon battles, apply effects, breed Pokemon, and generate new generations of Pokemon. 

### `pokemon_battle_simulator.py`
This script executes the entire Pokemon Battle simulation. It imports and utilizes the other functions, and allows users to set parameters for the simulation.

## Usage

To run the Pokemon Battle Simulator, simply execute the `pokemon_battle_simulator.py` script. This script takes three parameters: 

1. `num_generations`: The number of generations to simulate. The simulator will run battles and breed new Pokemon for the specified number of generations.

2. `num_battles`: The number of battles that each Pokemon will fight in each generation.

3. `effects_activator`: A boolean that determines whether move effects will be applied during the battles.

Example usage: 

```shell
python pokemon_battle_simulator.py --num_generations 50 --num_battles 30 --effects_activator False
```

If you just run `pokemon_battle_simulator.py`, default is `num_battles` = 20, `num_battles` = 10 and `num_battles` = False

## Data input

The datasets used in this project consist of a list of Pokemon form the first Generation with their respective stats, the effectivnis of the different types and a list of moves with their corresponding effects and accuracy. The data of Pokemon is stored in CSV format and the effectivnis/moves data in json

## Data output


This Pokemon Battle Simulator generates multiple outputs, providing a detailed account of the simulated battles and generations. The results are stored in two main locations: ./output_data/plots and ./output_data/generations.

### Plots

In this directory, three plots are generated, which give a comprehensive overview of the Pokemon universe as it evolves through the generations:

    move_frequencies.png: This plot displays the frequencies of different Pokemon moves. The x-axis represents the moves, and the y-axis represents the frequency of each move. This can give you an idea of which moves are most commonly used among the Pokemon in the simulation.

    stat_values.png: This plot shows the values of different stats (like HP, Attack, Defense, and Speed) for the Pokemon. The x-axis represents the different stats, and the y-axis represents the value for each stat. This gives a snapshot of the average capabilities of the Pokemon across generations.

    type_frequencies.png: This plot presents the frequencies of the different types of Pokemon. The x-axis represents the types, and the y-axis represents the frequency of each type. This helps to understand which types are most common in the simulation.

### Generations

In this directory, you can find the data for each generation of Pokemon that is simulated. For each generation, a CSV file is generated which includes the details of each Pokemon, such as their name, type, stats, and moves.

These outputs collectively help in analyzing the results of the simulated Pokemon battles and the evolution of Pokemon across generations. They provide valuable insights into the trends and patterns in the Pokemon universe as it evolves over time.




