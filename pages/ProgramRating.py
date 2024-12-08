import streamlit as st
import csv
import requests
import random
from io import StringIO

# Function to read the CSV file and convert it to the desired format
def read_csv_to_dict(file_content):
    program_ratings = {}
    reader = csv.reader(file_content)
    header = next(reader)  # Skip the header
    for row in reader:
        program = row[0]
        ratings = [float(x) for x in row[1:]]  # Convert the ratings to floats
        program_ratings[program] = ratings
    return program_ratings

# Streamlit interface
st.title("Optimal Program Scheduling with Genetic Algorithm")

# Input GitHub URL
github_url = st.text_input("Enter the raw GitHub URL for the CSV file:", 
                           "https://raw.githubusercontent.com/your_username/your_repo/main/program_rating.csv")

# Load the CSV file from GitHub
if github_url:
    try:
        response = requests.get(github_url)
        response.raise_for_status()  # Raise an error for bad responses
        file_content = StringIO(response.text)
        ratings = read_csv_to_dict(file_content)

        # Input parameters
        GEN = st.number_input("Generations", min_value=1, max_value=1000, value=100)
        POP = st.number_input("Population Size", min_value=1, max_value=500, value=50)
        CO_R = st.slider("Crossover Rate", min_value=0.0, max_value=0.95, value=0.8, step=0.01)
        MUT_R = st.slider("Mutation Rate", min_value=0.01, max_value=0.05, value=0.02, step=0.01)
        EL_S = st.number_input("Elitism Size", min_value=1, max_value=50, value=2)

        # Prepare the data
        all_programs = list(ratings.keys())
        all_time_slots = list(range(6, 24))

        # Fitness function
        def fitness_function(schedule):
            total_rating = 0
            for time_slot, program in enumerate(schedule):
                total_rating += ratings[program][time_slot]
            return total_rating

        # Crossover
        def crossover(schedule1, schedule2):
            crossover_point = random.randint(1, len(schedule1) - 2)
            child1 = schedule1[:crossover_point] + schedule2[crossover_point:]
            child2 = schedule2[:crossover_point] + schedule1[crossover_point:]
            return child1, child2

        # Mutation
        def mutate(schedule):
            mutation_point = random.randint(0, len(schedule) - 1)
            new_program = random.choice(all_programs)
            schedule[mutation_point] = new_program
            return schedule

        # Genetic Algorithm
        def genetic_algorithm(initial_schedule, generations, population_size, crossover_rate, mutation_rate, elitism_size):
            population = [initial_schedule]

            for _ in range(population_size - 1):
                random_schedule = initial_schedule.copy()
                random.shuffle(random_schedule)
                population.append(random_schedule)

            for generation in range(generations):
                new_population = []

                # Elitism
                population.sort(key=lambda schedule: fitness_function(schedule), reverse=True)
                new_population.extend(population[:elitism_size])

                while len(new_population) < population_size:
                    parent1, parent2 = random.choices(population, k=2)
                    if random.random() < crossover_rate:
                        child1, child2 = crossover(parent1, parent2)
                    else:
                        child1, child2 = parent1.copy(), parent2.copy()

                    if random.random() < mutation_rate:
                        child1 = mutate(child1)
                    if random.random() < mutation_rate:
                        child2 = mutate(child2)

                    new_population.extend([child1, child2])

                population = new_population

            return population[0]

        # Brute force initial schedule
        initial_schedule = all_programs[:len(all_time_slots)]
        rem_t_slots = len(all_time_slots) - len(initial_schedule)

        # Genetic algorithm
        genetic_schedule = genetic_algorithm(initial_schedule, GEN, POP, CO_R, MUT_R, EL_S)
        final_schedule = initial_schedule + genetic_schedule[:rem_t_slots]

        # Display results
        st.subheader("Optimal Schedule")
        for time_slot, program in enumerate(final_schedule):
            st.write(f"Time Slot {all_time_slots[time_slot]:02d}:00 - Program {program}")

        st.write("Total Ratings:", fitness_function(final_schedule))

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching the file: {e}")
