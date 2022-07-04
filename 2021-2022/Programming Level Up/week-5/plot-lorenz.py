import matplotlib.pyplot as plt
import numpy
import pickle

with open("data.pkl", "rb") as pkl_file:
    xs, ys, zs = pickle.load(pkl_file)

# Plot
ax = plt.figure().add_subplot(projection='3d')

ax.plot(xs, ys, zs, lw=0.5)
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor")

plt.show()
