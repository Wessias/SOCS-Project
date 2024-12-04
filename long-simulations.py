# %% [markdown]
# # Left to do Friday
# - Stop simulation when less than 2 particles are left
# - Use one door and adjust width of door (10 simulations per door, large radius)
# - Use one door and adjust door sight (10 simulations per door, large radius)
# - Use one door and one size and adjust the number of particles (10 simulations per door, large radius)
# - Number of doors (on different positions), play around with other things.
# - Plot the graphs in some nice way


# %%
from main import run_simulation, door
import numpy as np
import matplotlib.pyplot as plt

n_itterations = 100000
# %%

def vary_door_size(size_list, sim_per_size):
    n_particles = 100
    particle_size = np.random.uniform(0.3,0.45, n_particles)
    board_size = 50  # [m]
    particle_vision = 3 # [m]
    vision = 1000
    delta_t = 0.1
    
    time_list = []

    for size in size_list:
        doors = [
            door(np.array([0, -board_size/2]), size, vision, "horizontal")
        ]


        for i in range(sim_per_size):
            time_size = 0
            positions, v, time, _ = run_simulation(n_particles, particle_size, board_size, particle_vision, n_itterations, delta_t, doors)
            time_size += time

        time_list.append(time_size/sim_per_size)

    return time_list

# %%
min_size = 4
max_size = 10
n_sizes = 5
size_list = np.linspace(min_size, max_size, n_sizes)

sim_per_size = 2

time_list = vary_door_size(size_list, sim_per_size)

# %% Plotting
print(time_list)
plt.plot(size_list, time_list)
# %%
