import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3

# Input file name
fname = "nbody.dat"

print("loading input file ", fname," ...")

# Opens input file
f = open(fname)

# Reads header info (nBodies, nIters)
first_line = f.readline()
line = np.fromstring(first_line, sep=' ')
nBodies = int(line[0])
timesteps = int(line[1])

# Adjusts marker size based on number of bodies in problem
marker = 1.0
if (nBodies > 1000):
    marker = 0.5
if (nBodies > 10000 ):
    marker = 0.1
if( nBodies > 100000 ):
    marker = 0.02

# Allocations array to hold a timestep
arr = np.empty(dtype=float, shape=(nBodies,3))

# Reads initial conditions
for i in range(nBodies):
    line = f.readline()
    body = np.fromstring(line, sep=' ')
    arr[i,:] = body

# Create a 3D plot and initialize it with initial particle states
fig, ax = plt.subplots()
ax = p3.Axes3D(fig)

# Build Plot
points, = ax.plot3D(arr[:,0], arr[:,1], arr[:,2], 'wo', markersize=marker)
ax.set_ylim(-2.0, 2.0)
ax.set_xlim(-2.0, 2.0)
ax.set_zlim3d(-2.0, 2.0)
ax.set_facecolor('xkcd:black')
plt.axis('off')
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

# Function that will be called for each frame of the animation
def update(data):
    update.t += 1
    print("Processing Time Step ", update.t)
    # Reads a set of bodies into an array
    arr = np.empty(dtype=float, shape=(nBodies,3))
    for i in range(nBodies):
        line = f.readline()
        body = np.fromstring(line, sep=' ')
        arr[i,:] = body

    points.set_xdata(arr[:,0])
    points.set_ydata(arr[:,1])
    points.set_3d_properties(arr[:,2]) # No set_zdata, se we use this

    return points,

update.t = -1

# Generate the animation
ani = animation.FuncAnimation(fig, update, timesteps-2)

# Save .mp4 of the animation
# Bitrate may need to be increased for higher particle counts
ani.save('nbody_simulation.mp4', fps=60, bitrate=500000, extra_args=["-s", "1280x720"])
#plt.show()
