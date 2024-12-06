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
import pandas as pd

n_itterations = 100000
# %%

def vary_door_size(size_list, sim_per_size):
    n_particles = 100
    board_size = 50  # [m]
    particle_vision = 3 # [m]
    vision = 1000
    delta_t = 0.1
    particle_size = np.random.uniform(0.3,0.45, n_particles)
    time_list = []

    for index, size in enumerate(size_list):
        print("Progress: ", index, '/', len(size_list))
        doors = [
            door(np.array([0, -board_size/2]), size, vision, "horizontal")
        ]

        for i in range(sim_per_size):
            time_size = 0
            positions, v, time, _ = run_simulation(n_particles, particle_size, board_size, particle_vision, n_itterations, delta_t, doors)
            time_size += time

        time_list.append(time_size/sim_per_size)

    return time_list

# %% Simulation varying door size
min_size = 0.5
max_size = 10
n_sizes = 10
size_list = np.linspace(min_size, max_size, n_sizes)

sim_per_size = 4

time_list = vary_door_size(size_list, sim_per_size)

# %%
np.savetxt("varying_door_size.csv", np.array([time_list, size_list]), delimiter=",")

# %% Plotting varying door size

data = np.genfromtxt('varying_door_size.csv', delimiter=',')
print(data)
time_list = data[0]
size_list = data[1]

print(time_list)
plt.plot(size_list, time_list)
plt.xlabel("Door size")
plt.ylabel("Time to escape")
plt.title("Time to escape varying door size")
plt.show()
# %%

def vary_door_sight(size_list, sim_per_size):
    n_particles = 100
    board_size = 50  # [m]
    particle_vision = 3 # [m]
    delta_t = 0.1
    particle_size = np.random.uniform(0.3,0.45, n_particles) #For independent runs probably should just implement this in init func

   
    
    time_list = []

    for size_radius in size_list:
        print("Size ", round(size_radius,2))
        doors = [
            door(np.array([ -board_size/2, 0]), 1, size_radius, "vertical")
        ]


        for i in range(sim_per_size):
            time_size = 0
            positions, v, time, escape_times = run_simulation(n_particles, particle_size, board_size, particle_vision, n_itterations, delta_t, doors)
            time_size += time

        time_list.append(time_size/sim_per_size)

    return time_list

# %%

min_size = 5
max_size = 30
n_sizes = 10
size_list = np.linspace(min_size, max_size, n_sizes)

sim_per_size = 4

time_list_varying_sight = vary_door_sight(size_list, sim_per_size)

np.savetxt("varying_door_sight.csv", np.array([time_list, size_list]), delimiter=",")

# %%
data = np.genfromtxt('varying_door_sight.csv', delimiter=',')
print(data)
time_list = data[0]
size_list = data[1]


print(time_list_varying_sight)
print(size_list)
plt.plot(size_list, time_list_varying_sight)
plt.xlabel("Door sight")
plt.ylabel("Time to escape")
plt.title("Time to escape varying door sight")
plt.show()


# %%


def vary_num_particles(size_list, sim_per_size):
    board_size = 50  # [m]
    particle_vision = 3 # [m]
    delta_t = 0.1
    door_sight = 1000
    door_width = 1
    
   
    
    time_list = []

    for num_part in size_list:
        print("Particle amount: ", num_part)
        doors = [
            door(np.array([ -board_size/2, 0]), door_width, door_sight, "vertical")
        ]


        n_particles = num_part
        for i in range(sim_per_size):
            particle_size = np.random.uniform(0.3,0.45, n_particles) 

            time_size = 0
            positions, v, time, escape_times = run_simulation(n_particles, particle_size, board_size, particle_vision, n_itterations, delta_t, doors)
            time_size += time

        time_list.append(time_size/sim_per_size)

    return time_list
# %%
min_size = 10
max_size = 300
jump_size = 50
size_list = np.arange(min_size, max_size, jump_size)


sim_per_size = 1

time_list_varying_num_particles = vary_num_particles(size_list, sim_per_size)

np.savetxt("varying_number_of_particles.csv", np.array([time_list, size_list]), delimiter=",")


# %%

data = np.genfromtxt('varying_number_of_particles.csv', delimiter=',')
print(data)
time_list = data[0]
size_list = data[1]


print(time_list_varying_num_particles)
print(size_list)
plt.plot(size_list, time_list_varying_num_particles)
plt.xlabel("Number of particles")
plt.ylabel("Time to escape")
plt.title("Time to escape varying number of particles")
plt.show()
