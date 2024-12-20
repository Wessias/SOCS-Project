# %% [markdown]
#Såg att i lecture notes 8 finns ett tillämpnings exempel: *Collective dynamics of pedestrians in a corridor*
#som verkar matcha vårt arbete ganska väl, se ->  https://www.wellesu.com/10.1103/PhysRevE.102.022307
#
#Specifikt känns det som att model (c) (Social Force Model + Vicsek Model) är exakt vad vi vill ha.
# SFM använder sig av en "desire force" + "social force" (agents vill inte vara för nära varandra) + "wall force" (inte gå in i väg) 
# och Vicsek ger oss att agents vill matcha individer i närheten //Viggo


# # Nice get commands to know
# git pull

# git add .
# git commit -m "message"
# git push

# git reset --hard HEAD~

# %% [markdown]
# # Left to do Wednesday
# ## Priority High
# - Remove particles when they reach the door - Klar
# - Global parameters for scalars used in multiple places - Klar
# - Remove the wall where the door is - klar
# - Ability to use more doors - Klar
# - Variable door vision - Klar
# - Randomize particle size - Klar

# # Left to do Friday
# - Stop simulation when less than 2 particles are left
# - Use one door and adjust width of door (10 simulations per door, large radius)
# - Use one door and adjust door sight (10 simulations per door, large radius)
# - Use one door and one size and adjust the number of particles (10 simulations per door, large radius)
# - Number of doors (on different positions), play around with other things.
# - Plot the graphs in some nice way

# Pastear in lite kod snuttar som jag tror kan vara användbara för oss
# %% Interaction function for Vicsek model
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.distance import cdist
import time

wall_color = '#197598'
door_color = '#9AC27E'
particle_color = '#354926'
arrow_color = '#9AC27E'
face_color = '#C1CEB2'
sight_color = '#197598'

eta = 0.1
kappa = 2.4e5 # [kg/(ms)]
n_particles = 100
A = 2000 # [N]
B = 0.08 # [m]
k = 1.2e5 # [kg/s^2]
m = 60 # [kg] average weight of a Taylor Swift fan (teenage girl in the range [13,18])
relaxation_time = 0.5 # [s]
v0 = 3.5 # [m/s] achieveable speed by most Taylor Swift fans in a high-octane situation

class door:
    def __init__(self, position, size, vision, orientation):
        self.position = position
        self.vision = vision
        self.size = size
        self.orientation = orientation # "horizontal" or "vertical"

def next_v_vecsek_model(position, v, Rf, L, delta_t):    
    N = position.shape[0]
    x = position[:, 0]
    y = position[:, 1]

    theta_next = np.zeros(N)

    dist2 = cdist(position, position, 'sqeuclidean')
    neighbor_mask = dist2 <= Rf ** 2

    theta = np.arctan2(v[:, 1], v[:, 0])

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    av_sin_theta = neighbor_mask @ sin_theta / neighbor_mask.sum(axis=1)
    av_cos_theta = neighbor_mask @ cos_theta / neighbor_mask.sum(axis=1)

    theta_avg = np.arctan2(av_sin_theta, av_cos_theta)
    theta_delta = eta * np.random.normal(0, 1, N) * delta_t
    
    theta_next = theta_avg + theta_delta

    v[:, 0] = v0 * np.cos(theta_next)
    v[:, 1] = v0 * np.sin(theta_next)

    # for j in range(N):
    #     # No need for replicas in our case
    #     dist2 = (x - x[j]) ** 2 + (y - y[j]) ** 2 
    #     nn = np.where(dist2 <= Rf ** 2)[0]
        
    #     # The list of nearest neighbours is set.
    #     nn = nn.astype(int)
    #     theta = np.arctan2(v[nn, 1], v[nn, 0])
        
    #     # Circular average.
    #     av_sin_theta = np.mean(np.sin(theta))
    #     av_cos_theta = np.mean(np.cos(theta))

    #     theta_avg = np.arctan2(av_sin_theta, av_cos_theta)
    #     theta_delta = eta * np.random.normal(0, 1) * delta_t
    #     speed = np.sqrt(v[j, 0] ** 2 + v[j, 1] ** 2)

    #     theta_next[j] = theta_avg + theta_delta

    # v[:, 0] = v0 * np.cos(theta_next)
    # v[:, 1] = v0 * np.sin(theta_next)
          
    return v

# reflection on boundry shouldn't be needed due to forces by the walls
def reflecting_boundary_conditions(positions, L):
    mask_lower = positions < -L/2
    mask_upper = positions > L/2

    positions[mask_lower] = -L/2 + (-L/2 - positions[mask_lower])
    positions[mask_upper] = L/2 - (positions[mask_upper] - L/2)


def desire_force(doors, speed_desire, position, v, relaxation_time):
    # add posibility to go toward a line (2, 2) array, use the shortest distance

    N = position.shape[0]
    f_desire = np.zeros((N, 2))

    for i in range(N):
        # Calculate distances to each door
        distances = [np.linalg.norm(door.position - position[i]) for door in doors]
        # Find the nearest door
        nearest_door_index = np.argmin(distances)
        nearest_door = doors[nearest_door_index]
        nearest_door_position = nearest_door.position
        nearest_distance = distances[nearest_door_index]

        if nearest_distance <= nearest_door.vision:
            # Compute the desired force
            direction = nearest_door_position - position[i]
            direction = direction / np.linalg.norm(direction)
            v_desire = direction * speed_desire
            f_desire[i] = m * ((v_desire - v[i]) / relaxation_time)
        else:
            # Outside door's vision radius, no desire force
            f_desire[i] = 0

    return  f_desire

def social_force(position, particle_size, A, B):
    #Maybe only calculate for neighbours?
    N = position.shape[0]
    
    f_social = np.zeros((N, 2))
    for i in range(N):
        distance = position - position[i]
        force_norm = np.linalg.norm(distance, axis=1)

        nn = (force_norm > 0) # & (force_norm < particle_size)

        force_direction = distance[nn] / force_norm[nn][:, None]

        force_size = -A * np.exp((particle_size[i] - force_norm[nn]) / B)

        nn_forces = (force_size[:, None] * force_direction)

        f_social[i] = np.sum(nn_forces, axis=0)
    
    return f_social
        
def granular_force(position, v, particle_size, k, kappa):
    N = position.shape[0]

    f_granular = np.zeros((N, 2))
    for i in range(N):
        distance = position - position[i]
        force_norm = np.linalg.norm(distance, axis=1)
        
        nn = (force_norm > 0) & (force_norm < particle_size[i])

        kompressive_force_magnitude = k * (particle_size[i] - force_norm[nn])
        force_direction = distance[nn] / force_norm[nn][:, None]
        kompressive_force = kompressive_force_magnitude[:, None] * force_direction

        friction_force_magnitude = kappa * np.sum((v[nn] - v[i]), axis=1)
        orthogonal_direction = np.array([force_direction[:, 1], -force_direction[:, 0]]).T
        friction_force = friction_force_magnitude[:, None] * orthogonal_direction

        nn_forces = kompressive_force + friction_force

        f_granular[i] = np.sum(nn_forces, axis=0)

    return -f_granular

def wall_social_force(position, particle_size, L, A, B):
    # if the door is on the border, the wall force shouldn't exist
    N = position.shape[0]
    
    f_social_wall = np.zeros((N, 2))
    for i in range(N):
        particle_pos = position[i]

        is_near_door = False
        for door in doors:
            if door.orientation == "horizontal":
                left_edge = door.position[0] - door.size / 2
                right_edge = door.position[0] + door.size / 2

                # If horizontally near a door, skip vertical wall force
                if left_edge <= particle_pos[0] <= right_edge and abs(particle_pos[1] - door.position[1]) <= particle_size[i]:
                    is_near_door = True
                    break

            elif door.orientation == "vertical":
                bottom_edge = door.position[1] - door.size / 2
                top_edge = door.position[1] + door.size / 2

                # If vertically near a door, skip horizontal wall force
                if bottom_edge <= particle_pos[1] <= top_edge and abs(particle_pos[0] - door.position[0]) <= particle_size[i]:
                    is_near_door = True
                    break




        if not is_near_door:
            distance_to_wall = L/2 - np.abs(position[i])
        
            force_magnitude = A * np.exp((particle_size[i]/2 - distance_to_wall) / B)
            direction = -np.sign(position[i])
        
            f_social_wall[i] = force_magnitude * direction

    return f_social_wall


def granular_wall_force(position, v, particle_size, L, k):
    # doesn't work as intended
    max_radius = max(particle_size/2)

    distance_to_wall = L/2 - np.abs(position)

    kompressive_force_magnitude = k * (max_radius - distance_to_wall)
    direction = -np.sign(position)

    kompressive_force = kompressive_force_magnitude * direction

    friction_force_magnitude = kappa * np.sum(v, axis=1)
    orthogonal_direction = - np.array([direction[:, 1], -direction[:, 0]]).T
    friction_force = friction_force_magnitude[:, None] * orthogonal_direction

    f_granular_wall = (kompressive_force + friction_force) if np.any(distance_to_wall < max_radius) else 0

    return f_granular_wall

def next_v_force_model(position, v, Rf, L, particle_size, delta_t, doors):
    # m = 1, delta_t = 1
    # must fineturn the parameters
    df = desire_force(doors, v0, position, v, relaxation_time)
    sf = social_force(position, particle_size, A, B)
    gf = granular_force(position, v, particle_size, k, kappa)
    wf = wall_social_force(position, particle_size, L, A, B)
    gwf = granular_wall_force(position, v, particle_size, L, k) # something wack is happening

    v = v + (1/m) * (df + sf + gf + wf) * delta_t # + gwf

    return v

def update_v(position, v, particle_vision, board_size, particle_size, delta_t, doors):
    # m = 1

    v_vicsek_model = next_v_vecsek_model(position, v, particle_vision, board_size, delta_t)
    v_force_model = next_v_force_model(position, v, particle_size, board_size, particle_size, delta_t, doors)

    v_combined = v0 * (v_vicsek_model + v_force_model) / np.linalg.norm(v_vicsek_model + v_force_model)

    return v_combined


def index_of_particles_close_to_doors(doors, positions, particle_size):
    # Check if particle is near a door (disable wall force in these regions)
    N = positions.shape[0]
    idx_of_close = []

    for i in range(N):
        is_near_door = False
        for door in doors:
            if door.orientation == 'horizontal':
                # Horizontal door: span along x-axis
                left_edge = door.position[0] - door.size / 2
                right_edge = door.position[0] + door.size / 2

                # Check if particle is horizontally within the door and close vertically
                if left_edge <= positions[i, 0] <= right_edge and abs(positions[i, 1] - door.position[1]) <= 1.5*particle_size[i]:
                    is_near_door = True
                    break

            elif door.orientation == 'vertical':
                # Vertical door: span along y-axis
                bottom_edge = door.position[1] - door.size / 2
                top_edge = door.position[1] + door.size / 2

                # Check if particle is vertically within the door and close horizontally
                if bottom_edge <= positions[i, 1] <= top_edge and abs(positions[i, 0] - door.position[0]) <= 1.5*particle_size[i]:
                    is_near_door = True
                    break

        if is_near_door:
            idx_of_close.append(i)
    return idx_of_close 
    
#    %% Particle simulation

def init_particles(N, L, v_max=1):
    v = (np.random.rand(N, 2) - 0.5) * v_max
    #position = np.random.uniform(-L/4, L/4, (N, 2))

    positions = np.empty((0, 2))  # Initialize an empty array for positions

    max_size = max(particle_size)
    tries = 0
    while positions.shape[0] < N:
        # Generate a candidate position
        candidate = np.random.uniform(-L / 3.5, L / 3.5, (1, 2))  # Shape (1, 2)

        # Compute distances to all existing positions
        if positions.shape[0] == 0 or np.all(np.linalg.norm(positions - candidate, axis=1) >= max_size + 0.5):
            positions = np.vstack([positions, candidate])  # Add the candidate if it's valid
        
        if tries > 10000000:
            print("Reset positions")
            positions = np.empty((0, 2))
            tries = 0  
        tries += 1
    
    #Just used this cause for a smaller arena it was chaos.
    #x = np.linspace(-L/5, L/5, int(np.sqrt(N)))
    #y = np.linspace(-L/10, L/10, int(np.sqrt(N)))

    # Create a 2D grid of x and y values
    #x_grid, y_grid = np.meshgrid(x, y)

    # Combine the grid into an array of points
    #position = np.vstack([x_grid.ravel(), y_grid.ravel()]).T

    return positions, v

def run_simulation(n_particles, particle_size, board_size, particle_vision, n_itterations, delta_t, doors):
    # if particles tuch the door, they should be removed
    escape_times = []
    position, v = init_particles(n_particles, board_size)

    for i in range(n_itterations):
        if len(position) <= 2:
            return position, v, round(i*delta_t,2), escape_times

        v = update_v(position, v, particle_vision, board_size, particle_size, delta_t, doors)
        position = position + v * delta_t
        
        particles_to_remove = index_of_particles_close_to_doors(doors, position, particle_size)

        if len(particles_to_remove) != 0:
            position = np.delete(position, particles_to_remove, axis=0)
            v = np.delete(v, particles_to_remove, axis=0)
            escape_times.append(round(delta_t*i,2))
        
        # reflecting_boundary_conditions(position, board_size)

    return position, v, round(i*delta_t,2), escape_times

# %% Simulation animation

def run_simulation_animation(n_particles, particle_size, board_size, particle_vision, n_itterations, delta_t, doors):
    
    position, v = init_particles(n_particles, board_size)
    elapsed_time = 0

    def reset_simulation():
        nonlocal position, v, particle_size, elapsed_time
        position, v = init_particles(n_particles, board_size)
        particle_size = np.random.uniform(0.3, 0.45, n_particles)
        elapsed_time = 0  # Reset elapsed time

    reset_simulation()

    fig, ax = plt.subplots()
    scatter = ax.scatter(position[:, 0], position[:, 1], s=particle_size*100)
    quiver = ax.quiver(position[:, 0], position[:, 1], v[:, 0], v[:, 1], color=arrow_color)
    ax.set_xlim(-board_size/2-0.1, board_size/2+0.1)
    ax.set_ylim(-board_size/2-0.1, board_size/2+0.15)
    fig.set_facecolor(face_color)
    ax.axis("off")
    #ax.set_xlabel('x')
    #ax.set_ylabel('y')
    ax.set_title('Particle simulation')
    plt.plot([-board_size/2, -board_size/2, board_size/2, board_size/2, -board_size/2],
     [-board_size/2, board_size/2, board_size/2, -board_size/2, -board_size/2], 
     color=wall_color, label='Wall')
    
    for door in doors:
        if door.orientation == "vertical":
            plt.plot([door.position[0],door.position[0], door.position[0] ], [door.position[1], door.position[1]+door.size, door.position[1]-door.size], color=door_color)
        else:
            plt.plot([door.position[0],door.position[0]+door.size, door.position[0]-door.size ], [door.position[1], door.position[1], door.position[1]], color=door_color)


     # Draw vision circles for the doors
    vision_circles = []
    for door in doors:
        circle = plt.Circle(door.position, door.vision, color=sight_color, linestyle='dotted', fill=False, linewidth=1.5)
        ax.add_artist(circle)
        vision_circles.append(circle)

    elapsed_time = -2*delta_t  # Track elapsed time

    def update_frame(frame):
        nonlocal position, v, scatter, quiver, particle_size, elapsed_time
        if len(position) == 0:
            # ax.set_title(
            # f'Evacuation simulation, frame: {frame} ({frame * delta_t:.2f} s)\n'
            # f'People remaining: {len(position)}', color="white")
            return

        
        elapsed_time += delta_t  # Update elapsed time
        v = update_v(position, v, particle_vision, board_size, particle_size, delta_t, doors)
        position = position + v * delta_t
        particles_to_remove = index_of_particles_close_to_doors(doors, position, particle_size)

        if len(particles_to_remove) != 0:
            position = np.delete(position, particles_to_remove, axis=0)
            v = np.delete(v, particles_to_remove, axis=0)
            particle_size = np.delete(particle_size, particles_to_remove)
        # reflecting_boundary_conditions(position, board_size)
        if frame == 2:
            #Break between runs
            time.sleep(4)

        if frame % 1 == 0: # Update plot every 1 frames
            scatter.remove()
            quiver.remove()
            if len(position) == 0:
                frame = 1
                reset_simulation()

            scatter = ax.scatter(position[:, 0], position[:, 1], color=particle_color, s=particle_size**3*500)
            quiver = ax.quiver(position[:, 0], position[:, 1], v[:, 0], v[:, 1], color=arrow_color, scale=40)

            ax.set_title(
             f'Evacuation simulation\n'
             f'People remaining: {len(position)}, Time Passed: {elapsed_time:.2f} s '
         )
        return scatter, quiver, *vision_circles

    def on_key(event):
        if event.key == 'r':  # Reset animation on 'r' key press
            reset_simulation()

    
    ani = animation.FuncAnimation(fig, update_frame, frames=n_itterations*2**52, interval=1, blit=False)
    fig.canvas.mpl_connect('key_press_event', on_key)  # Connect key press event
    plt.show()
# %% Simulation parameters, values from paper we used
# The variables that are comments I have put as global up top for now.
n_particles = 200
#particle_size = 0.4 # [m] average shoulder width of a teenage girl in the range [13-18]
particle_size = np.random.uniform(0.3,0.45, n_particles)
board_size = 50  # [m]
particle_vision = 3 # [m]
n_itterations = 100000
delta_t = 0.1
#eta = 0.1
#kappa = 2.4e5 # [kg/(ms)]
#A = 2000 # [N]
#B = 0.08 # [m]
#k = 1.2e5 # [kg/s^2]
#m = 60 # [kg]
#relaxation_time = 0.5 # [s]
#v0 

# (door_position, door_width, door_sight, door_orientation)
doors = [
    door(np.array([ -board_size/2, 0]), 1, 20, "vertical"),
    door(np.array([ 20, -board_size/2]), 3, 20, "horizontal")
    ] #Two doors
#door_possition = np.array([[-particle_size, particle_size], [-board_size/2, -board_size/2]])



# %% Simulation animation, need to run entire script to see animation
run_simulation_animation(n_particles, particle_size, board_size, particle_vision, n_itterations, delta_t, doors)

# %% Simulation plot
#positions, v, total_time, escape_times = run_simulation(n_particles, particle_size, board_size, particle_vision, n_itterations, delta_t, doors)

#if len(positions) != 0:
#    plt.scatter(positions[:, 0], positions[:, 1], label='Final position')
#    plt.quiver(positions[:, 0], positions[:, 1], v[:, 0], v[:, 1], color='red')
#    plt.show()

#print(len(positions))



 # %% [markdown]
# (SFM model) velocity(t+dt) = velocity(t) + (1/m)(desire_force + social_force + granular_force + wall_force + granular_wall_force) 
# $$ v^{\textbf{SFM}}(t + \Delta t) = \frac{1}{m}( \bm{F_{D} + F_{S} + F_{G} + F_{W} + F_{GW}})$$

# %%
