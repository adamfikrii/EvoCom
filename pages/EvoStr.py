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
