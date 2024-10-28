import streamlit as st
import random

# Set up page
st.set_page_config(page_title="Genetic Algorithm")
st.header("Genetic Algorithm", divider="gray")

# Define constants and Streamlit inputs
POP_SIZE = 500  # Number of Chromosomes in our population
TARGET = st.text_input("Enter your target string:", "Adam")  # Goal
MUT_RATE = st.number_input("Enter your mutation rate (0-1):", min_value=0.0, max_value=1.0, value=0.2)  # Mutation rate
GENES = ' abcdefghijklmnopqrstuvwxyzQWERTYUIOPLKJHGFDSAZXCVBNM'  # Gene pool

# Functions for genetic algorithm operations
def initialize_pop(target):
    population = []
    for _ in range(POP_SIZE):
        population.append([random.choice(GENES) for _ in range(len(target))])
    return population

def fitness_cal(target, chromosome):
    return sum(1 for t, c in zip(target, chromosome) if t != c)

def selection(population):
    return sorted(population, key=lambda x: x[1])[:POP_SIZE // 2]

def crossover(selected, chromo_len):
    offspring = []
    for _ in range(POP_SIZE):
        parent1, parent2 = random.choice(selected)[0], random.choice(selected)[0]
        crossover_point = random.randint(1, chromo_len - 1)
        child = parent1[:crossover_point] + parent2[crossover_point:]
        offspring.append(child)
    return offspring

def mutate(offspring, mutation_rate):
    for i in range(len(offspring)):
        offspring[i] = [gene if random.random() > mutation_rate else random.choice(GENES) for gene in offspring[i]]
    return offspring

def main():
    if st.button("Run Genetic Algorithm"):
        population = initialize_pop(TARGET)
        generation = 1
        found = False

        while not found:
            # Calculate fitness for each chromosome
            population = [[chromo, fitness_cal(TARGET, chromo)] for chromo in population]
            population = sorted(population, key=lambda x: x[1])

            # Check if the target was reached
            if population[0][1] == 0:
                st.success(f"Target found! Generation: {generation}, String: {''.join(population[0][0])}")
                found = True
                break

            # Select, crossover, and mutate to form the new generation
            selected = selection(population)
            offspring = crossover(selected, len(TARGET))
            population = mutate(offspring, MUT_RATE)
            
            # Display intermediate generation info
            st.write(f"Generation {generation}, Best Fit: {''.join(population[0][0])}, Fitness: {population[0][1]}")
            generation += 1

# Run the genetic algorithm
main()
