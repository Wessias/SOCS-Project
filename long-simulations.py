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

eta = 0.1
kappa = 2.4e5 # [kg/(ms)]
n_particles = 100
A = 2000 # [N]
B = 0.08 # [m]
k = 1.2e5 # [kg/s^2]
m = 60 # [kg] average weight of a Taylor Swift fan (teenage girl in the range [13,18])
relaxation_time = 0.5 # [s]
v0 = 3.5 # [m/s] achieveable speed by most Taylor Swift fans in a high-octane situation

particle_size = np.random.uniform(0.3,0.45, n_particles)
board_size = 100  # [m]
particle_vision = 3 # [m]
n_itterations = 100000
delta_t = 0.1

doors = [
    door(np.array([board_size/2, 0]), 2, 15, "vertical"),
    door(np.array([-board_size/2, 0]), 1, 15, "vertical"),
    door(np.array([0,board_size/2]), 4, 10, "horizontal")
    ]

positions, v, total_time, escape_times = run_simulation(n_particles, particle_size, 
                                                        board_size, particle_vision,
                                                          n_itterations, delta_t, doors)



# %%
