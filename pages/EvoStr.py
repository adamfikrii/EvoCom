import numpy as np
import streamlit as st
from numpy import exp, sqrt, cos, e, pi
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Ackley function
def objective(x, y):
    return -20.0 * exp(-0.2 * sqrt(0.5 * (x**2 + y**2))) - exp(0.5 * (cos(2 * pi * x) + cos(2 * pi * y))) + e + 20

# Streamlit interface
st.title("Ackley Function Visualization")

# Input range sliders
r_min = st.slider("Select minimum range for x and y", -30.0, 0.0, -15.0)
r_max = st.slider("Select maximum range for x and y", 0.0, 30.0, 15.0)

# Sample input range uniformly at 0.1 increments
xaxis = np.arange(r_min, r_max, 0.1)
yaxis = np.arange(r_min, r_max, 0.1)

# Create a mesh from the axis
x, y = np.meshgrid(xaxis, yaxis)

# Compute Ackley function values
results = objective(x, y)

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, results, cmap='jet')

# Display the plot in Streamlit
st.pyplot(fig)

import numpy as np
import streamlit as st
from numpy import exp, sqrt, cos, e, pi, argsort, randn, rand, asarray, seed

# Ackley objective function
def objective(v):
    x, y = v
    return -20.0 * exp(-0.2 * sqrt(0.5 * (x**2 + y**2))) - exp(0.5 * (cos(2 * pi * x) + cos(2 * pi * y))) + e + 20

# Check if a point is within bounds
def in_bounds(point, bounds):
    for d in range(len(bounds)):
        if point[d] < bounds[d, 0] or point[d] > bounds[d, 1]:
            return False
    return True

# Evolution strategy (mu, lambda) algorithm
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
        # Evaluate fitness for the population
        scores = [objective(c) for c in population]
        # Rank scores in ascending order
        ranks = argsort(argsort(scores))
        # Select top mu ranked solutions
        selected = [i for i, _ in enumerate(ranks) if ranks[i] < mu]
        
        # Create children from parents
        children = []
        for i in selected:
            # Check if this parent is the best solution
            if scores[i] < best_eval:
                best, best_eval = population[i], scores[i]
                st.write(f"{epoch}, Best: f({best}) = {best_eval:.5f}")

            # Generate children
            for _ in range(n_children):
                child = None
                while child is None or not in_bounds(child, bounds):
                    child = population[i] + randn(len(bounds)) * step_size
                children.append(child)

        # Replace population with children
        population = children

    return best, best_eval

# Streamlit UI
st.title("Evolution Strategy Optimization of the Ackley Function")
st.write("Using a (μ, λ) evolution strategy to minimize the Ackley function.")

# Input parameters
seed_value = st.number_input("Random Seed", value=1, step=1)
seed(seed_value)

n_iter = st.slider("Number of Iterations", min_value=100, max_value=5000, value=5000)
step_size = st.slider("Step Size", min_value=0.01, max_value=1.0, value=0.15)
mu = st.slider("Number of Parents (μ)", min_value=1, max_value=50, value=20)
lam = st.slider("Number of Children (λ)", min_value=10, max_value=200, value=100)
bounds = np.array([[-5.0, 5.0], [-5.0, 5.0]])

# Run the optimization
if st.button("Run Optimization"):
    best, score = es_comma(objective, bounds, n_iter, step_size, mu, lam)
    st.write("Optimization Complete!")
    st.write(f"Best Solution: f({best}) = {score}")

# evolution strategy (mu + lambda) of the ackley objective function
from numpy import asarray
from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi
from numpy import argsort
from numpy.random import randn
from numpy.random import rand
from numpy.random import seed

# objective function
def objective(v):
	x, y = v
	return -20.0 * exp(-0.2 * sqrt(0.5 * (x**2 + y**2))) - exp(0.5 * (cos(2 * pi * x) + cos(2 * pi * y))) + e + 20

# check if a point is within the bounds of the search
def in_bounds(point, bounds):
	# enumerate all dimensions of the point
	for d in range(len(bounds)):
		# check if out of bounds for this dimension
		if point[d] < bounds[d, 0] or point[d] > bounds[d, 1]:
			return False
	return True

# evolution strategy (mu + lambda) algorithm
def es_plus(objective, bounds, n_iter, step_size, mu, lam):
	best, best_eval = None, 1e+10
	# calculate the number of children per parent
	n_children = int(lam / mu)
	# initial population
	population = list()
	for _ in range(lam):
		candidate = None
		while candidate is None or not in_bounds(candidate, bounds):
			candidate = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
		population.append(candidate)
	# perform the search
	for epoch in range(n_iter):
		# evaluate fitness for the population
		scores = [objective(c) for c in population]
		# rank scores in ascending order
		ranks = argsort(argsort(scores))
		# select the indexes for the top mu ranked solutions
		selected = [i for i,_ in enumerate(ranks) if ranks[i] < mu]
		# create children from parents
		children = list()
		for i in selected:
			# check if this parent is the best solution ever seen
			if scores[i] < best_eval:
				best, best_eval = population[i], scores[i]
				print('%d, Best: f(%s) = %.5f' % (epoch, best, best_eval))
			# keep the parent
			children.append(population[i])
			# create children for parent
			for _ in range(n_children):
				child = None
				while child is None or not in_bounds(child, bounds):
					child = population[i] + randn(len(bounds)) * step_size
				children.append(child)
		# replace population with children
		population = children
	return [best, best_eval]

# seed the pseudorandom number generator
seed(1)
# define range for input
bounds = asarray([[-5.0, 5.0], [-5.0, 5.0]])
# define the total iterations
n_iter = 5000
# define the maximum step size
step_size = 0.15
# number of parents selected
mu = 20
# the number of children generated by parents
lam = 100
# perform the evolution strategy (mu + lambda) search
best, score = es_plus(objective, bounds, n_iter, step_size, mu, lam)
print('Done!')
print('f(%s) = %f' % (best, score))
