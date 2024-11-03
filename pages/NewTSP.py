import matplotlib.pyplot as plt
from itertools import permutations
from random import shuffle
import random
import numpy as np
import seaborn as sns
import streamlit as st

# Title and instructions
st.title("Traveling Salesman Problem using Genetic Algorithm")
st.write("Input city coordinates and parameters for the genetic algorithm to find the shortest path.")

# User inputs for city names and coordinates
num_cities = st.number_input("Enter number of cities:", min_value=2, max_value=20, value=5)
cities_names = []
x = []
y = []

for i in range(num_cities):
    city_name = st.text_input(f"Enter name for city {i+1}:", key=f"city_{i}")
    city_x = st.number_input(f"Enter x coordinate for {city_name}:", key=f"x_{i}")
    city_y = st.number_input(f"Enter y coordinate for {city_name}:", key=f"y_{i}")
    
    if city_name:
        cities_names.append(city_name)
        x.append(city_x)
        y.append(city_y)

city_coords = dict(zip(cities_names, zip(x, y)))

# Genetic algorithm parameters
n_population = st.number_input("Population Size:", min_value=50, max_value=500, value=250)
crossover_per = st.slider("Crossover Rate:", min_value=0.1, max_value=1.0, value=0.8)
mutation_per = st.slider("Mutation Rate:", min_value=0.01, max_value=1.0, value=0.2)
n_generations = st.number_input("Number of Generations:", min_value=10, max_value=500, value=200)

# Pastel color palette
colors = sns.color_palette("pastel", len(cities_names))

# Initial map of cities
fig, ax = plt.subplots()
for i, (city, (city_x, city_y)) in enumerate(city_coords.items()):
    color = colors[i]
    ax.scatter(city_x, city_y, c=[color], s=1200, zorder=2)
    ax.annotate(city, (city_x, city_y), fontsize=12, ha='center', va='bottom', xytext=(0, -30),
                textcoords='offset points')

fig.set_size_inches(8, 6)
st.pyplot(fig)

# Genetic Algorithm Functions
def initial_population(cities_list, n_population=250):
    population_perms = []
    possible_perms = list(permutations(cities_list))
    random_ids = random.sample(range(len(possible_perms)), n_population)
    for i in random_ids:
        population_perms.append(list(possible_perms[i]))
    return population_perms

def dist_two_cities(city_1, city_2):
    city_1_coords = city_coords[city_1]
    city_2_coords = city_coords[city_2]
    return np.sqrt(np.sum((np.array(city_1_coords) - np.array(city_2_coords))**2))

def total_dist_individual(individual):
    total_dist = 0
    for i in range(len(individual)):
        if i == len(individual) - 1:
            total_dist += dist_two_cities(individual[i], individual[0])
        else:
            total_dist += dist_two_cities(individual[i], individual[i+1])
    return total_dist

def fitness_prob(population):
    total_dist_all_individuals = [total_dist_individual(ind) for ind in population]
    max_population_cost = max(total_dist_all_individuals)
    population_fitness = max_population_cost - np.array(total_dist_all_individuals)
    population_fitness_probs = population_fitness / population_fitness.sum()
    return population_fitness_probs

def roulette_wheel(population, fitness_probs):
    cumsum_probs = fitness_probs.cumsum()
    selection_index = np.searchsorted(cumsum_probs, random.random())
    return population[selection_index]

def crossover(parent_1, parent_2):
    cut = random.randint(1, len(cities_names) - 1)
    offspring_1 = parent_1[:cut] + [city for city in parent_2 if city not in parent_1[:cut]]
    offspring_2 = parent_2[:cut] + [city for city in parent_1 if city not in parent_2[:cut]]
    return offspring_1, offspring_2

def mutation(offspring):
    idx1, idx2 = random.sample(range(len(cities_names)), 2)
    offspring[idx1], offspring[idx2] = offspring[idx2], offspring[idx1]
    return offspring

def run_ga(cities_names, n_population, n_generations, crossover_per, mutation_per):
    population = initial_population(cities_names, n_population)
    best_individual = min(population, key=total_dist_individual)

    for _ in range(n_generations):
        fitness_probs = fitness_prob(population)
        new_population = []

        for __ in range(n_population // 2):
            parent_1 = roulette_wheel(population, fitness_probs)
            parent_2 = roulette_wheel(population, fitness_probs)
            offspring_1, offspring_2 = crossover(parent_1, parent_2)

            if random.random() < mutation_per:
                offspring_1 = mutation(offspring_1)
            if random.random() < mutation_per:
                offspring_2 = mutation(offspring_2)

            new_population.extend([offspring_1, offspring_2])

        population = sorted(new_population, key=total_dist_individual)[:n_population]
        best_individual = min(population, key=total_dist_individual)

    return best_individual, total_dist_individual(best_individual)

if st.button("Run Genetic Algorithm"):
    best_route, min_distance = run_ga(cities_names, n_population, n_generations, crossover_per, mutation_per)

    # Display Results
    st.write("**Shortest Path:**", " → ".join(best_route))
    st.write("**Minimum Distance:**", round(min_distance, 2))

    # Plot the best route
    x_shortest = [city_coords[city][0] for city in best_route] + [city_coords[best_route[0]][0]]
    y_shortest = [city_coords[city][1] for city in best_route] + [city_coords[best_route[0]][1]]

    fig, ax = plt.subplots()
    ax.plot(x_shortest, y_shortest, '--go', label='Best Route', linewidth=2.5)
    plt.legend()

    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            ax.plot([x[i], x[j]], [y[i], y[j]], 'k-', alpha=0.09, linewidth=1)

    plt.title("TSP Best Route Using Genetic Algorithm", fontsize=20)
    fig.set_size_inches(8, 6)
    st.pyplot(fig)
