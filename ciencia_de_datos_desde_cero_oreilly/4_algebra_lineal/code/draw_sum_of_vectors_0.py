import matplotlib.pyplot as plt
import numpy as np

# Define the vectors
vector1 = np.array([1, 2])
vector2 = np.array([2, 1])

# The sum of the 2 vectors
resultant = vector1 + vector2

# Plot the vectors
# plt.figure() creates a new figure (a blank canvas) for plotting.
# figsize=(8, 8) sets the size of the canvas to be 8x8 inches.
plt.figure(figsize=(8, 8))

# plt.quiver is used to plot arrows.
# The first two 0 values represent the starting point of the vector (the origin (0,0)
# angles='xy' ensures that the arrow is drawn based on x and y coordinates.
# scale_units='xy' and scale=1 make sure that the arrow length matches the actual length of the vector.
# color='r' collor of the vector
# label='Vector 1' is the label used in the legend.


# Representation of the first vector
plt.quiver(
    0,
    0,
    vector1[0],
    vector1[1],
    angles="xy",
    scale_units="xy",
    scale=1,
    color="r",
    label="Vector 1",
)


# Representation of the second vector
plt.quiver(
    0,
    0,
    vector2[0],
    vector2[1],
    angles="xy",
    scale_units="xy",
    scale=1,
    color="b",
    label="Vector 2",
)

# Representation of the sum of the 2 previous vectors
plt.quiver(
    0,
    0,
    resultant[0],
    resultant[1],
    angles="xy",
    scale_units="xy",
    scale=1,
    color="g",
    label="Resultant (Sum)",
)

#  Set Up the Plot Appearance
# Set the grid, limits, and labels

# set the limits of the x and y axes, allowing enough space to show all the vectors.
plt.xlim(-1, 4)
plt.ylim(-1, 4)

# adds a grid to the plot, making it easier to visualize coordinates.
plt.grid()

# plt.axhline() and plt.axvline() draw horizontal and vertical lines
# at 𝑦 = 0 and x=0 to represent the x- and y-axes.
plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)

# plt.gca().set_aspect('equal') makes sure that one unit on the x-axis is the same length as one unit on the y-axis, so the vectors' lengths are accurate.
plt.gca().set_aspect("equal", adjustable="box")

# plt.legend() displays the labels of the arrows (from label in plt.quiver).
plt.legend()

# Set the title and axis labels
plt.title("Vector Sum")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

# Show the plot
plt.show()
print(".")
print("end")
