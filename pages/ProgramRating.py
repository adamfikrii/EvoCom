import streamlit as st
import csv
import random
import pandas as pd

# Function to read the CSV file and convert it to the desired format
def read_csv_to_dict(file_path):
    program_ratings = {}
    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        for row in reader:
            program = row[0]
            ratings = [float(x) for x in row[1:]]  # Convert the ratings to floats
            program_ratings[program] = ratings
    return program_ratings

# Path to the CSV file
file_path = 'pages/program_ratings.csv'

# Get the data in the required format
program_ratings_dict = read_csv_to_dict(file_path)
ratings = program_ratings_dict

GEN = 100
POP = 50
CO_R = 0.8
MUT_R = 0.02
EL_S = 2

all_programs = list(ratings.keys())
all_time_slots = list(range(6, 24))  # Time slots

# Defining fitness function
def fitness_function(schedule):
    total_rating = sum(ratings[program][time_slot] for time_slot, program in enumerate(schedule))
    return total_rating

# Initialize all schedules
def initialize_pop(programs, time_slots):
    if not programs:
        return [[]]
    all_schedules = []
    for i in range(len(programs)):
        for schedule in initialize_pop(programs[:i] + programs[i + 1:], time_slots):
            all_schedules.append([programs[i]] + schedule)
    return all_schedules

def finding_best_schedule(all_schedules):
    return max(all_schedules, key=fitness_function)

all_possible_schedules = initialize_pop(all_programs, all_time_slots)

# Genetic Algorithm Functions
def crossover(schedule1, schedule2):
    point = random.randint(1, len(schedule1) - 2)
    return schedule1[:point] + schedule2[point:], schedule2[:point] + schedule1[point:]

def mutate(schedule):
    point = random.randint(0, len(schedule) - 1)
    schedule[point] = random.choice(all_programs)
    return schedule

def genetic_algorithm(initial_schedule, generations=GEN, population_size=POP, crossover_rate=CO_R, mutation_rate=MUT_R, elitism_size=EL_S):
    population = [initial_schedule] + [random.sample(initial_schedule, len(initial_schedule)) for _ in range(population_size - 1)]
    for generation in range(generations):
        population.sort(key=fitness_function, reverse=True)
        new_population = population[:elitism_size]
        while len(new_population) < population_size:
            if random.random() < crossover_rate:
                parent1, parent2 = random.choices(population, k=2)
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = population[0][:], population[1][:]
            if random.random() < mutation_rate:
                mutate(child1)
            if random.random() < mutation_rate:
                mutate(child2)
            new_population.extend([child1, child2])
        population = new_population
    return population[0]

# Streamlit UI with Center Alignment
st.markdown(
    """
    <style>
    .centered {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title
st.markdown('<div class="centered"><h1>Genetic Algorithm - Optimal Program Schedule</h1></div>', unsafe_allow_html=True)

# Sliders and Button (Centered)
st.markdown('<div class="centered">', unsafe_allow_html=True)
CO_R = st.slider("Crossover Rate (CO_R)", 0.0, 0.95, 0.8, 0.01)
MUT_R = st.slider("Mutation Rate (MUT_R)", 0.01, 0.05, 0.02, 0.01)
calculate_button = st.button("Calculate Optimal Schedule")
st.markdown('</div>', unsafe_allow_html=True)

# Results (Run Calculation)
if calculate_button:
    initial_best_schedule = finding_best_schedule(all_possible_schedules)
    rem_t_slots = len(all_time_slots) - len(initial_best_schedule)
    genetic_schedule = genetic_algorithm(initial_best_schedule, crossover_rate=CO_R, mutation_rate=MUT_R)
    final_schedule = initial_best_schedule + genetic_schedule[:rem_t_slots]

    # Display Results (Centered)
    st.markdown('<div class="centered">', unsafe_allow_html=True)
    st.write("### Final Optimal Schedule:")
    schedule_table = pd.DataFrame({
        "No.": [i + 1 for i in range(len(final_schedule))],
        "Time Slot": [f"{all_time_slots[time_slot]:02d}:00" for time_slot in range(len(final_schedule))],
        "Program": final_schedule
    })
    st.table(schedule_table)
    st.write("### Total Ratings:", fitness_function(final_schedule))
    st.markdown('</div>', unsafe_allow_html=True)
