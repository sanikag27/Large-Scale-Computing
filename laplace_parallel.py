from mpi4py import MPI
import time
import numpy as np
import math
import matplotlib.pyplot as plt

ROWS, COLUMNS = 1000, 1000  # Size of the grid
MAX_TEMP_ERROR = 0.01

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Number of rows each process is responsible for
rows_per = ROWS // size 

# Initialize temperature arrays
temperature = np.zeros((rows_per + 2, COLUMNS + 2))  # Including ghost rows
temperature_last = np.zeros((rows_per + 2, COLUMNS + 2))

# Calculate the start and end rows each process works on
# start_row = rank * rows_per + 1
# end_row = start_row + rows_per - 1

start_time = time.time()

def initialize_temperature(temp, rank):
    # Initialize the interior with 0
    temp[:, :] = 0

    # Right boundary condition (sinusoidal)
    for i in range(1, rows_per + 2):
        globali = i + rank * rows_per
        # Adjust the indexing to account for the ghost row offset
        temp[i, COLUMNS + 1] = 100 * math.sin(((3.14159 / 2) / ROWS) * globali)

    # Bottom boundary condition for the last process only
    if rank == size - 1:
        for i in range(COLUMNS + 2):
            temp[rows_per + 1, i] = 100 * math.sin(((3.14159 / 2) / COLUMNS) * i)

initialize_temperature(temperature_last, rank)

# Maximum number of iterations
max_iterations = None
if rank == 0:
    max_iterations = int(input("Enter Maximum Iterations: "))

max_iterations = comm.bcast(max_iterations, root=0)

dt = 100  # Initial error (change in temperature)
iteration = 1

# Main computation loop
while dt > MAX_TEMP_ERROR and iteration < max_iterations:

    # Send/Receive boundary rows to/from neighboring processes
    if rank > 0: # (upper-neighbor)
        comm.Send(temperature_last[1, :], dest=rank - 1)
        #sends the second row (index 1) of the temperature_last array to the process above (dest=rank - 1)
        comm.Recv(temperature_last[0, :], source=rank - 1)
        #receives the ghost row from the process above into the first row (index 0) of the temperature_last array.

    if rank < size - 1: # (lower-neighbor)
        comm.Send(temperature_last[rows_per, :], dest=rank + 1)
        # sends the last real row (index rows_per) of temperature_last to the process below (dest=rank + 1).
        comm.Recv(temperature_last[rows_per + 1, :], source=rank + 1)
        # receives the ghost row from the process below into the last row (index rows_per + 1) of the temperature_last array.
    comm.Barrier()


    # Compute new temperature values (excluding ghost rows)
    for i in range(1, rows_per + 1):
        for j in range(1, COLUMNS + 1):
            temperature[i, j] = 0.25 * (
                temperature_last[i + 1, j] +
                temperature_last[i - 1, j] +
                temperature_last[i, j + 1] +
                temperature_last[i, j - 1]
            )


    dt_indi = 0
    for i in range(1, rows_per+1):
        for j in range(1, COLUMNS + 1):
            dt_indi = max(dt_indi, temperature[i,j] - temperature_last[i,j])
            temperature_last[ i , j ] = temperature [ i , j ]

    dt = comm.allreduce(dt_indi, op=MPI.MAX)
    if rank == 0:
        print(f"iteration: {iteration}", flush = True)
    iteration += 1

# Final step: Gather data from all processes
local_temp = temperature_last[1:rows_per + 1, 1:COLUMNS + 1].flatten()  # Flatten valid rows (excluding ghost rows)

if rank == 0:
    # Prepare space for the full grid
    full_grid = np.zeros((ROWS, COLUMNS))
else:
    full_grid = None

comm.Gather(local_temp, full_grid, root=0)

# Gather the local results from all processes into the full grid on the root process


# Reshape and plot the final temperature distribution on the root process
if rank == 0:
    full_grid = np.reshape(full_grid, (ROWS, COLUMNS))  # Reshape the flattened array
    plt.imshow(full_grid)
    plt.colorbar()
    plt.savefig("newest.png")
    plt.show()

# Finalize MPI
MPI.Finalize()

end_time = time.time()
if rank == 0:
    print(f"Time taken: {end_time - start_time} seconds")