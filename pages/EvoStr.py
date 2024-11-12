import numpy as np
import streamlit as st
from numpy import exp, sqrt, cos, e, pi, argsort, randn, rand, asarray, seed
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Ackley function for plotting
def objective(x, y):
    return -20.0 * exp(-0.2 * sqrt(0.5 * (x**2 + y**2))) - exp(0.5 * (cos(2 * pi * x) + cos(2 * pi * y))) + e + 20

# Streamlit interface for visualization
st.title("Ackley Function Visualization")

# Input range sliders
r_min = st.slider("Select minimum range for x and y", -30.0, 0.0, -15.0)
r_max = st.slider("Select maximum range for x and y", 0.0, 30.0, 15.0)

# Generate mesh for plot
xaxis = np.arange(r_min, r_max, 0.1)
yaxis = np.arange(r_min, r_max, 0.1)
x, y = np.meshgrid(xaxis, yaxis)
results = objective(x, y)

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, results, cmap='jet')
st.pyplot(fig)

# Define the Ackley objective function for optimization
def objective(v):
    x, y = v
    return -20.0 * exp(-0.2 * sqrt(0.5 * (x**2 + y**2))) - exp(0.5 * (cos(2 * pi * x) + cos(2 * pi * y))) + e + 20

# Bounds check
def in_bounds(point, bounds):
    for d in range(len(bounds)):
        if point[d] < bounds[d, 0] or point[d] > bounds[d, 1]:
            return False
    return True

# Evolution strategy (μ, λ)
def es_comma(objective, bounds, n_iter, step_size, mu, lam):
    best, best_eval = None, 1e+10
    n_children = int(lam / mu)
    population = []

    # Initial population
    for _ in range(lam):
        candidate = None
        while candidate is None or not in_bounds(candidate, bounds):
            candidate = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
        population.append(candidate)

    # Perform search
    for epoch in range(n_iter):
        scores = [objective(c) for c in population]
        ranks = argsort(argsort(scores))
        selected = [i for i, _ in enumerate(ranks) if ranks[i] < mu]
        
        children = []
        for i in selected:
            if scores[i] < best_eval:
                best, best_eval = population[i], scores[i]
                st.write(f"{epoch}, Best: f({best}) = {best_eval:.5f}")

            for _ in range(n_children):
                child = None
                while child is None or not in_bounds(child, bounds):
                    child = population[i] + randn(len(bounds)) * step_size
                children.append(child)

        population = children

    return best, best_eval

# Streamlit interface for (μ, λ) strategy
st.title("Evolution Strategy Optimization of the Ackley Function")
st.write("Using a (μ, λ) evolution strategy to minimize the Ackley function.")
seed_value = st.number_input("Random Seed", value=1, step=1)
seed(seed_value)
n_iter = st.slider("Number of Iterations", min_value=100, max_value=5000, value=5000)
step_size = st.slider("Step Size", min_value=0.01, max_value=1.0, value=0.15)
mu = st.slider("Number of Parents (μ)", min_value=1, max_value=50, value=20)
lam = st.slider("Number of Children (λ)", min_value=10, max_value=200, value=100)
bounds = np.array([[-5.0, 5.0], [-5.0, 5.0]])

if st.button("Run Optimization (μ, λ)"):
    best, score = es_comma(objective, bounds, n_iter, step_size, mu, lam)
    st.write("Optimization Complete!")
    st.write(f"Best Solution: f({best}) = {score}")

# Evolution strategy (μ + λ)
def es_plus(objective, bounds, n_iter, step_size, mu, lam):
    best, best_eval = None, 1e+10
    n_children = int(lam / mu)
    population = []

    for _ in range(lam):
        candidate = None
        while candidate is None or not in_bounds(candidate, bounds):
            candidate = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
        population.append(candidate)

    for epoch in range(n_iter):
        scores = [objective(c) for c in population]
        ranks = argsort(argsort(scores))
        selected = [i for i, _ in enumerate(ranks) if ranks[i] < mu]
        
        children = []
        for i in selected:
            if scores[i] < best_eval:
                best, best_eval = population[i], scores[i]
                st.write(f"{epoch}, Best: f({best}) = {best_eval:.5f}")
            
            children.append(population[i])
            for _ in range(n_children):
                child = None
                while child is None or not in_bounds(child, bounds):
                    child = population[i] + randn(len(bounds)) * step_size
                children.append(child)

        population = children

    return best, best_eval

# Streamlit interface for (μ + λ) strategy
if st.button("Run Optimization (μ + λ)"):
    best, score = es_plus(objective, bounds, n_iter, step_size, mu, lam)
    st.write("Optimization Complete!")
    st.write(f"Best Solution: f({best}) = {score}")
