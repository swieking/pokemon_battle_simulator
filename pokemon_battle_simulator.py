"""
Pokemon Battle Simulation
Author: Sören Wieking
Date: August 5, 2023

This Python script provides a simulation for Pokemon battles based on the rules and mechanics of the popular 
video game franchise Pokemon. The purpose of this script is to simulate multiple generations of Pokemon 
battles, with the results of each generation influencing the next.

Each Pokemon is represented by a dictionary containing its name, various statistics HP, Attack, Defense, 
Speed, Types, its Moves, and other data. The battles between Pokemon are simulated in a turn-based 
system, with each Pokemon using one of its moves to attack the other each turn. The moves have a chance to hit 
or miss, and can have additional effects such as inflicting status conditions.

The Pokemon's statistics, moves, and types have an impact on the outcome of the battles, and the results are 
used to influence the next generation of Pokemon. This simulation allows for observing how certain statistics, 
moves, and types might become more or less common over multiple generations.

This script also includes functionality for visualizing the results of the simulation, showing how various 
factors evolve over the generations.
"""
import pandas as pd
import random 
import numpy as np
import argparse
import matplotlib.pyplot as plt
import json
import os


def main():

    parser = argparse.ArgumentParser(description='This script runs the Pokémon evolution simulation for a certain number of generations.')
    parser.add_argument('-g', '--num_generations', type=int, default=20,
                        help='The number of generations to simulate. Default is 20.')
    parser.add_argument('-e', '--effects_activator', action='store_true',
                        help='activate Pokemon attack effects')
    parser.add_argument('-b', '--num_battles',  type=int, default=10,
                        help='number of battles each Pokemon make for each generation. Default is 10.')
    
    args = parser.parse_args()
    num_generations = args.num_generations
    effects_activator = args.effects_activator
    num_battles = args.num_battles

    df = generate_pokemons()


    for i in range(num_generations):
        #battle and breed new generation
        df = battle(df, effects_activator, num_battles)
        df = df.sort_values(by='wins', ascending=False)
        
        #safe generation
        if not os.path.exists('./output_data/generations'):
            os.makedirs('./output_data/generations')
        df.to_csv(f'./output_data/generations/generation{i+1}.csv') 
        df = generate_new_generation(df)

        #print generation
        print(f'\n{"="*50}')
        print(f'{" " * 10}Generation {i} top 20 Pokemon:')
        print(f'{"-"*50}')
        print(df.head(20).to_string(index=False, formatters={col: "{:0.0f}".format for col in df.select_dtypes(include=np.number).columns}))
        print(f'{"="*50}\n')

        #reset score
        df['battles'] = 0
        df['wins'] = 0

    visualize(num_generations)


def generate_pokemons():
    """
    Generate a dataframe of Pokemon from a CSV file.

    This function reads Pokemon data from a CSV file, 
    selects specific columns, renames them to lowercase, 
    adds move data to each Pokemon, and initializes 
    battle and and win counts to 0. 

    Returns
    -------
    df : pandas.DataFrame
        A DataFrame containing the Pokemon data.
    """
    df = pd.read_csv('./input_data/FirstGenPokemon.csv')
    df = df.rename(columns=lambda x: x.strip())
    df = df.drop(['Types','Number','Height(m)','Weight(kg)','Male_Pct','Female_Pct','Evolutions','Legendary','Capt_Rate', 'Exp_Points', 'Exp_Speed', 'Base_Total', 'Special','Normal_Dmg','Fire_Dmg','Water_Dmg','Eletric_Dmg','Grass_Dmg','Ice_Dmg','Fight_Dmg','Poison_Dmg','Ground_Dmg','Flying_Dmg','Psychic_Dmg','Bug_Dmg','Rock_Dmg','Ghost_Dmg','Dragon_Dmg'], axis=1)
    df.columns = df.columns.str.lower()
    df = append_moves(df)
    df['battles'] = 0
    df['wins'] = 0
    return(df)
        


def generate_new_generation(df):
    """
    Generates a new generation of Pokemon based on the best individuals of the previous generation.

    Parameters:
        df (pandas.DataFrame): A DataFrame representing the Pokemon of the previous generation, ranked by fitness.

    Returns:
        pandas.DataFrame: A DataFrame representing the new generation of Pokemon.

    This function operates as follows:
    - The top 100 Pokemon from the previous generation are kept.
    - The top 30 of these Pokemon each breed with five other random Pokemon from this group to produce new offspring.
    - The offspring are appended to the DataFrame.
    - The function ensures that a Pokemon does not breed with itself.
    - The breeding process is carried out by the 'breed' function.
    """

    #keep 100 best pokemon from last generation
    df = df.iloc[:100]
    best30 =  df.iloc[:30]

    next_generation = []

    #each of the best pokemon mates with ten other random pokemon to produce a new pokemon.
    for i in range(len(best30)):
        for _ in range(5):
            #choose a random partner for mating
            partner_index = random.randint(0, len(best30) - 1)
            #ensure the pokemon does not mate with itself
            while partner_index == i:
                partner_index = random.randint(0, len(best30) - 1)
            #breed the two pokemon to create a new one
            new_pokemon = breed(best30.iloc[i].to_dict(), best30.iloc[partner_index].to_dict())
            next_generation.append(new_pokemon)

    #append new pokemons
    next_generation_df = pd.DataFrame(next_generation)
    df = pd.concat([df, next_generation_df], ignore_index=True)

    return df

    

def breed(pokemon1, pokemon2):
    """
    Breeds two Pokemon to create a new one.

    Parameters:
        pokemon1 (dict): A dictionary representing the attributes of the first parent Pokemon.
        pokemon2 (dict): A dictionary representing the attributes of the second parent Pokemon.

    Returns:
        dict: A dictionary representing the attributes of the new Pokemon.
    """

    new_pokemon = {}
    #combine the names of the two parent Pokemon
    pokemon1_name_parts = [pokemon1['name'][:len(pokemon1['name'])//2], pokemon1['name'][len(pokemon1['name'])//2:]]
    pokemon2_name_parts = [pokemon2['name'][:len(pokemon2['name'])//2], pokemon2['name'][len(pokemon2['name'])//2:]]
    name = random.choice(pokemon1_name_parts) + random.choice(pokemon2_name_parts)
    name = name.lower() 
    name = name[0].upper() + name[1:]
    new_pokemon['name'] = name
    '''HP, ATK, DEF and speed will be somewhere between the values of the parents,
    with a higher probability close to the average.'''
    for stat in ['hp', 'attack', 'defense', 'speed']:
        stat_value = random.triangular(pokemon1[stat], pokemon2[stat], (pokemon1[stat] + pokemon2[stat]) / 2)
        new_pokemon[stat] = int(stat_value)

    #the types are randomly inherited from parents
    new_pokemon['type1'] = random.choice([pokemon1['type1'], pokemon2['type1']])
    new_pokemon['type2'] = random.choice([pokemon1['type2'], pokemon2['type2']])
    if new_pokemon['type1'] == new_pokemon['type2']:
        new_pokemon['type2'] = None

    #moves can only be inherited if the corresponding type has also been inherited 
    possible_moves = [move for move in move_list if move['type'] in [new_pokemon['type1'], new_pokemon['type2'], 'normal']]

    #choose 4 random possible moves
    chosen_moves = random.sample(possible_moves, min(4, len(possible_moves)))

    #add the selected moves to the new pokemon
    for i, move in enumerate(chosen_moves):
        new_pokemon[f'move{i + 1}'] = move['name']



    return new_pokemon



def battle(df, effects_activator, num_battles):
    """
    Conducts a series of battles between Pokemon in the provided DataFrame.

    Parameters:
        df (pandas.DataFrame): A DataFrame representing the current generation of Pokemon.
        effects_activator (bool): A boolean flag that, if true, allows moves to have additional effects in battles.
        num_battles (int): The number of battles each Pokemon should participate in.

    Returns:
        pandas.DataFrame: The updated DataFrame, with wins and battles updated for each Pokemon.

    The function performs the following operations:
    - The Pokemon participate in a best-of-seven series of battles, with the winner being the first to win four battles.
    - Two Pokemon are chosen randomly to battle each other.
    - A Pokemon cannot fight more than the defined number of battles.
    - The battles are conducted by the 'attack' function.
    - The score (wins and battles) is updated in the DataFrame after each battle.
    """

    #iterate until all pokemon have had their 5 battles
    b = 0
    while df['battles'].min() < num_battles:
        b += 1
        #select two random pokemon who have not yet had their 5 battles
        available = df[df['battles'] < num_battles].index
        if len(available) < 2:  #if only one pokemon is available, end the loop
            break

        pokemon1, pokemon2 = np.random.choice(available, 2, replace=False)
        pokemon1_dict = df.loc[pokemon1].to_dict()
        pokemon2_dict = df.loc[pokemon2].to_dict()
        wins1 = 0
        wins2 = 0
        turn = 0
        k = 0
        #best of seven battle
        while wins1 < 4 and wins2 < 4:
            k += 1
            while pokemon1_dict['hp'] > 0 and pokemon2_dict['hp'] > 0:
                if turn == 0:
                    turn = random.randint(1,2)
                if turn == 1:
                    pokemon1_dict ,pokemon2_dict = attack(pokemon1_dict, pokemon2_dict, effects_activator)
                    turn = 2
                else:
                    pokemon2_dict ,pokemon1_dict = attack(pokemon2_dict, pokemon1_dict, effects_activator)
                    turn = 1
            if pokemon1_dict['hp'] <= 0:
                wins2 += 1 
                turn = 1
            else: 
                wins1 += 1
                turn = 2
            
            #reset hp
            pokemon1_dict = df.loc[pokemon1].to_dict()
            pokemon2_dict = df.loc[pokemon2].to_dict()

        #update scores
        df.at[pokemon1, 'battles'] +=1
        df.at[pokemon2, 'battles'] +=1
        if wins1 == 4:
            df.at[pokemon1, 'wins'] += 1
        else:
            df.at[pokemon2, 'wins'] += 1
    return df


def attack(pokemon1, pokemon2, effects_activator):
    """
    Simulates a turn in a Pokemon battle where 'pokemon1' attacks 'pokemon2'.

    This function takes into account several factors including: the attacker's moves, the chance of 
    the attack hitting, the attack's power, the type effectiveness of the move, and potential additional 
    effects of the move. 

    Parameters
    ----------
    pokemon1 : dict
        A dictionary representing the attacking Pokemon. 
    pokemon2 : dict
        A dictionary representing the defending Pokemon. 
    effects_activator : bool
        A boolean that indicates whether or not additional move effects should be activated during the attack.

    Returns
    -------
    tuple
        A tuple containing the updated 'pokemon1' and 'pokemon2' dictionaries after the attack has been executed.
    """

    confused_paralyzed_miss_multiplier = 1

    #process already active effects 
    if 'eff_identifier' in pokemon2:
        #if the attacking or defending pokemon is poisoned, it additionally loses 20 hp.
        if pokemon2['eff_identifier'][0] == 'poisoned' and pokemon2['eff_identifier'][1] > 0:
            pokemon2['eff_identifier'][1] -= 1
            pokemon2['hp'] -= 20
    if 'eff_identifier' in pokemon1:
        #if the attacking or defending pokemon is poisoned, it additionally loses 20 hp.
        if pokemon1['eff_identifier'][0] == 'poisoned' and pokemon1['eff_identifier'][1] > 0:
            pokemon1['eff_identifier'][1] -= 1
            pokemon1['hp'] -= 20
        #if the attacking pokemon is sleeping or frozen, it misses a turn
        elif pokemon1['eff_identifier'][0] == 'sleeping' and pokemon1['eff_identifier'][1] > 0 or pokemon1['eff_identifier'][0] == 'frozen' and pokemon1['eff_identifier'][1] > 0:
            pokemon1 ['eff_identifier'][1] -= 1
            return pokemon1, pokemon2
        #if the attacking pokemon is paralyzed, its 1.5 times more likely to miss its attack
        elif pokemon1['eff_identifier'][0] == 'paralyzed' and pokemon1['eff_identifier'][1] > 0:
            pokemon1['eff_identifier'][1] -= 1
            confused_paralyzed_miss_multiplier = 1.5
        #if the attacking pokemon is confused, its twice as likely to miss its attack
        elif pokemon1['eff_identifier'][0] == 'confused' and pokemon1['eff_identifier'][1] > 0:
            pokemon1['eff_identifier'][1] -= 1
            confused_paralyzed_miss_multiplier = 2        

    #chose random move
    random_move = random.choice(['move1','move2','move3','move4'])
    for m in move_list:
        if m["name"] == pokemon1[random_move]:
            random_move = m
            break
    
    attack_hit_rate = calculate_miss_rate(random_move['acc'], pokemon2["speed"], confused_paralyzed_miss_multiplier)

    if not random.random() < attack_hit_rate: #missed
        return pokemon1, pokemon2
    else: #hit
        #calculate attack effectivness multiplier
        attack_effectiveness = type_effectiveness[random_move['type']][pokemon2['type1']]
        if pokemon2['type2'] != 'None' and pokemon2['type2'] != None:
            attack_effectiveness = attack_effectiveness * type_effectiveness[random_move['type']][pokemon2['type2']]

        #calculate attack power and subtract from hp
        attack_power = calculate_attack_power(pokemon1['attack'], pokemon2['defense'], random_move['power'], attack_effectiveness)        
        pokemon2['hp'] -= attack_power

        if random_move['effect'] and effects_activator == True:
            pokemon1, pokemon2 = apply_effect(random_move['effect'], pokemon1, pokemon2, attack_power)

    return pokemon1, pokemon2

def calculate_attack_power(atk,defense,power,atk_eff ):
    """
    Calculate the power of an attack considering the attack power, the attacker's attack level, 
    the defender's defense level and the attack effectiveness.

    Parameters:
    atk (int or float): The attack level of the attacking Pokemon.
    defense (int or float): The defense level of the defending Pokemon.
    power (int or float): The inherent power of the move being used.
    atk_eff (float): The effectiveness of the attack based on the type of the move and the type of the defending Pokemon. 
                      This is a multiplier that should be in the range [0, 1.5].
    
    Returns:
    attack_power (float): The final calculated power of the attack in
    """

    # Calculate the ratio of attack to defense
    attack_defense_ratio = atk / defense
    # Apply a sigmoid function to compress the ratio between 0 and 1
    attack_defense_factor = 1 / (1 + np.exp(-attack_defense_ratio))
    # Adjust this factor to ensure it is 0.5 when attack equals defense
    attack_defense_factor = 2 * (attack_defense_factor - 0.5)
    # Calculate the final power
    attack_power = power * attack_defense_factor * atk_eff
    return attack_power


def calculate_miss_rate(acc, speed, confusion_multiplier):
    """
    Function to calculate the hit rate of an attack considering accuracy, speed of the pokemons, and a confusion/paralysis multiplier.

    Parameters:
    acc (int): Accuracy of the attack (1-100).
    speed (int): Speed of the attacking Pokemon.
    confusion_multiplier (float): Multiplier to account for confused or paralyzed status (1.5 or 2).

    Returns:
    float: Adjusted accuracy which now represents the miss rate of the attack.
    """

    #calculate linear speed effect
    m = (0.70 - 1) / (120 - 30)
    b = 1 - m * 30
    speed_multiplier = m * speed + b
    #adjust accuracy for speed and confusion/paralysis
    acc = acc / 100 
    hit_rate = acc * speed_multiplier / confusion_multiplier
    return hit_rate

def apply_effect(effect, pokemon1, pokemon2, attack_power):
    """
    This function applies the specified effect of a move used by the attacking Pokemon on the attacked Pokemon during a battle. 
    Effects can involve direct damage, health recovery, stat modification, damage over time.

    Parameters
    ----------
    effect : str
        The specific effect of the move being used. It can be one of several pre-defined strings each 
        corresponding to a unique effect.
    pokemon1 : dict
        A dictionary representing the Pokemon using the move.
    pokemon2 : dict
        A dictionary representing the Pokemon the move is being used on. 
    attack_power : float
        The power of the move being used

    Returns
    -------
    tuple
        A tuple containing the modified 'pokemon1' and 'pokemon2' dictionaries after the effect has been applied.
    """
    
    may = random.random()
    
    if effect == 'one-hit-ko, if it hits':
        pokemon2['hp'] = 0 
    elif effect == 'user recovers half the hp inflicted on opponent':
        pokemon1['hp'] += attack_power / 2
    elif effect == 'raises users attack and speed':
        pokemon1['speed'] += 10
        pokemon1['attack'] += 5
    elif effect == 'may burn opponent':
        pokemon2['hp'] += 20
    elif effect == 'hits 2-5 times in one turn':
        hits = random.randint(1,4)
        pokemon2['hp'] -= attack_power * hits
    elif effect == "may lowers opponent's speed":
        if may > 0.5:
            pokemon2['speed'] -= 15
    elif effect == "may lowers opponent's defense":
        if may > 0.5:
            pokemon2['defense'] -= 15
    elif effect == 'puts opponent to sleep':
        pokemon2['eff_identifier'] = ['sleeping',2]
    elif effect == 'may paralyzes opponent':
        if may > 0.6:
            pokemon2['eff_identifier'] = ['paralyzed',2]
    elif effect == 'may confuses opponent':
        if may > 0.6:
            pokemon2['eff_identifier'] = ['confused',1]
    elif effect == 'may poison the opponent':
        if may > 0.5:
            pokemon2['eff_identifier'] = ['poisoned',3]
    elif effect == 'may freeze opponent':
        if may > 0.2:
            pokemon2['eff_identifier'] = ['frozen',1]
    elif effect == "may lowers opponent's attack":
        if may > 0.7:
            pokemon2['attack'] -= 20

    return pokemon1, pokemon2 


def append_moves(df):
    """
    This function adds move data to each Pokemon in a DataFrame. Each Pokemon 
    is assigned two moves that match its types and two normal type moves.

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame of Pokémon. Each Pokémon should have 'type1' and 'type2' fields.

    Returns
    -------
    df : pandas.DataFrame
        The same DataFrame, but with added 'move1', 'move2', 'move3', and 'move4' fields.
    """
    df['move1'] = None
    df['move2'] = None
    df['move3'] = None
    df['move4'] = None

    for i in df.index:
        #get the types of the Pokémon
        types = [df.at[i, 'type1'], df.at[i,'type2']]
        #get type specific moves
        type_moves = [move for move in move_list if move['type'] in types]
        #assign 2 type specific moves and 2 normal moves
        pomoves = random.sample(type_moves, 2)
        pomoves += (random.sample([move for move in move_list if move['type'] == 'normal' ],2))
        for j, m in enumerate(pomoves):
            df.at[i,f'move{j+1}'] = m['name']

    return df

def visualize(num_generations):
    type_counts = {}
    move_counts = {}
    avg_hp = []
    avg_atk = []
    avg_def = []
    avg_speed = []

    for i in range(1, num_generations + 1):  
        df = pd.read_csv(f'./output_data/generations/generation{i}.csv')

        # Update type counts
        for type, count in df['type1'].value_counts().items():
            if type not in type_counts:
                type_counts[type] = [0] * num_generations
            type_counts[type][i-1] = count

        # Update move counts
        for move, count in df['move1'].value_counts().items():
            if move not in move_counts:
                move_counts[move] = [0] * num_generations
            move_counts[move][i-1] = count

        # Calculate average stats
        avg_hp.append(df['hp'].mean())
        avg_atk.append(df['attack'].mean())
        avg_def.append(df['defense'].mean())
        avg_speed.append(df['speed'].mean())

    # Create directory if it doesn't exist
    if not os.path.exists('./output_data/plots'):
        os.makedirs('./output_data/plots')

    # Create plots
    # Plot for the type frequencies over generations
    plt.figure(figsize=(20, 10))
    #handle colors for the big amount of lines 
    cm = plt.get_cmap('gist_rainbow')
    NUM_COLORS = len(type_counts)
    plt.gca().set_prop_cycle(color=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
    for type, counts in type_counts.items():
        plt.plot(range(1, num_generations + 1), counts, label=type)   
    plt.title('Type Frequencies over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Frequency')
    plt.xticks(range(1, num_generations + 1)) 
    plt.legend(loc='best')
    plt.show()
    plt.savefig('./output_data/plots/type_frequencies.png')
    plt.close()

    #plot for the move frequencies over generations
    plt.figure(figsize=(20, 10))
    # Create a new dictionary where the keys are the move names and the values are the final counts.
    final_counts = {move: counts[-1] for move, counts in move_counts.items()}
    # Sort this dictionary based on the highest lest value, to better distinguish the colors of the lines
    sorted_moves = sorted(final_counts.items(), key=lambda item: item[1], reverse=True)
    # Use the sorted moves to plot and create the legend.
    for move, _ in sorted_moves:
        counts = move_counts[move]
        plt.plot(range(1, num_generations + 1), counts, label=move)
    plt.title('Move Frequencies over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Frequency')
    plt.xticks(range(1, num_generations + 1)) 
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') 
    plt.show()
    plt.savefig('./output_data/plots/move_frequencies.png', bbox_inches='tight')
    plt.close()

    #plot for the average stat values over generations
    plt.figure(figsize=(20, 10))
    plt.plot(range(1, num_generations + 1), avg_hp, label='HP')
    plt.plot(range(1, num_generations + 1), avg_atk, label='ATK')
    plt.plot(range(1, num_generations + 1), avg_def, label='DEF')
    plt.plot(range(1, num_generations + 1), avg_speed, label='SPEED')
    plt.title('Average Stat Values over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Value')
    plt.xticks(range(1, num_generations + 1)) 
    plt.legend(loc='best')
    plt.show()
    plt.savefig('./output_data/plots/stat_values.png')
    plt.close()
   
if __name__ == "__main__": 
    with open('./input_data/move_list.json', 'r') as f:
        move_list = json.load(f)
    with open('./input_data/type_effectiveness.json', 'r') as f:
        type_effectiveness = json.load(f)
    main()

    