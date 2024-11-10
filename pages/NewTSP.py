import matplotlib.pyplot as plt
from itertools import permutations, combinations
from random import shuffle
import random
import numpy as np
import statistics
import pandas as pd
import seaborn as sns
import streamlit as st

# Streamlit form to take user inputs
st.title("Traveling Salesman Problem")

# Input fields
x_input = st.text_input("Enter x coordinates separated by commas (e.g., 0,3,6,7,15,10,16,5,8,1.5):")
y_input = st.text_input("Enter y coordinates separated by commas (e.g., 1,2,1,4.5,-1,2.5,11,6,9,12):")
cities_names_input = st.text_input("Enter city names separated by commas (e.g., Gliwice, Cairo, Rome, Krakow, etc.):")

# Parse inputs
if x_input and y_input and cities_names_input:
    x = list(map(float, x_input.split(',')))
    y = list(map(float, y_input.split(',')))
    cities_names = cities_names_input.split(',')

    if len(x) == len(y) == len(cities_names):
        # Core code (same as provided, with slight modifications to use user input)
        city_coords = dict(zip(cities_names, zip(x, y)))
        n_population = 250
        crossover_per = 0.8
        mutation_per = 0.2
        n_generations = 200

        # Pastel Palette
        colors = sns.color_palette("pastel", len(cities_names))

        # City Icons
        city_icons = {name: icon for name, icon in zip(cities_names, ["♕", "♖", "♗", "♘", "♙", "♔", "♚", "♛", "♜", "♝"][:len(cities_names)])}

        # Plot cities
        fig, ax = plt.subplots()
        ax.grid(False)  # Grid

        for i, (city, (city_x, city_y)) in enumerate(city_coords.items()):
            color = colors[i]
            icon = city_icons[city]
            ax.scatter(city_x, city_y, c=[color], s=1200, zorder=2)
            ax.annotate(icon, (city_x, city_y), fontsize=40, ha='center', va='center', zorder=3)
            ax.annotate(city, (city_x, city_y), fontsize=12, ha='center', va='bottom', xytext=(0, -30),
                        textcoords='offset points')

            # Connect cities with opaque lines
            for j, (other_city, (other_x, other_y)) in enumerate(city_coords.items()):
                if i != j:
                    ax.plot([city_x, other_x], [city_y, other_y], color='gray', linestyle='-', linewidth=1, alpha=0.1)

        fig.set_size_inches(16, 12)
        st.pyplot(fig)

        # (Function definitions remain the same, include all functions here)

        # Run the Genetic Algorithm
        best_mixed_offspring = run_ga(cities_names, n_population, n_generations, crossover_per, mutation_per)

        total_dist_all_individuals = [total_dist_individual(individual) for individual in best_mixed_offspring]
        index_minimum = np.argmin(total_dist_all_individuals)
        minimum_distance = min(total_dist_all_individuals)
        st.write(f"Minimum Distance: {minimum_distance}")

        # Shortest path visualization
        shortest_path = best_mixed_offspring[index_minimum]
        st.write(f"Shortest Path: {shortest_path}")

        # Coordinates for the shortest path
        x_shortest, y_shortest = zip(*[city_coords[city] for city in shortest_path])
        x_shortest += (x_shortest[0],)
        y_shortest += (y_shortest[0],)

        fig, ax = plt.subplots()
        ax.plot(x_shortest, y_shortest, '--go', label='Best Route', linewidth=2.5)
        plt.legend()

        for i in range(len(x)):
            for j in range(i + 1, len(x)):
                ax.plot([x[i], x[j]], [y[i], y[j]], 'k-', alpha=0.09, linewidth=1)

        plt.title(label="TSP Best Route Using GA", fontsize=25, color="k")
        str_params = f'\n{n_generations} Generations\n{n_population} Population Size\n{crossover_per} Crossover\n{mutation_per} Mutation'
        plt.suptitle(f"Total Distance Travelled: {round(minimum_distance, 3)}{str_params}", fontsize=18, y=1.047)

        for i, txt in enumerate(shortest_path):
            ax.annotate(f"{i+1}- {txt}", (x_shortest[i], y_shortest[i]), fontsize=20)

        fig.set_size_inches(16, 12)
        st.pyplot(fig)

    else:
        st.error("Ensure that x, y coordinates, and city names have the same length.")
else:
    st.info("Please provide the x, y coordinates and city names to proceed.")

best_mixed_offspring = run_ga(cities_names, n_population, n_generations, crossover_per, mutation_per)

total_dist_all_individuals = []
for i in range(0, n_population):
    total_dist_all_individuals.append(total_dist_individual(best_mixed_offspring[i]))

index_minimum = np.argmin(total_dist_all_individuals)

minimum_distance = min(total_dist_all_individuals)
st.write(minimum_distance)

#shortest path
# shortest_path = offspring_list[index_minimum]
shortest_path = best_mixed_offspring[index_minimum]
st.write(shortest_path)

x_shortest = []
y_shortest = []
for city in shortest_path:
    x_value, y_value = city_coords[city]
    x_shortest.append(x_value)
    y_shortest.append(y_value)

x_shortest.append(x_shortest[0])
y_shortest.append(y_shortest[0])

fig, ax = plt.subplots()
ax.plot(x_shortest, y_shortest, '--go', label='Best Route', linewidth=2.5)
plt.legend()

for i in range(len(x)):
    for j in range(i + 1, len(x)):
        ax.plot([x[i], x[j]], [y[i], y[j]], 'k-', alpha=0.09, linewidth=1)

plt.title(label="TSP Best Route Using GA",
          fontsize=25,
          color="k")

str_params = '\n'+str(n_generations)+' Generations\n'+str(n_population)+' Population Size\n'+str(crossover_per)+' Crossover\n'+str(mutation_per)+' Mutation'
plt.suptitle("Total Distance Travelled: "+
             str(round(minimum_distance, 3)) +
             str_params, fontsize=18, y = 1.047)

for i, txt in enumerate(shortest_path):
    ax.annotate(str(i+1)+ "- " + txt, (x_shortest[i], y_shortest[i]), fontsize= 20)

fig.set_size_inches(16, 12)
# plt.grid(color='k', linestyle='dotted')
st.pyplot(fig)
