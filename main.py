# %% [markdown]
#Såg att i lecture notes 8 finns ett tillämpnings exempel: *Collective dynamics of pedestrians in a corridor*
#som verkar matcha vårt arbete ganska väl, se ->  https://www.wellesu.com/10.1103/PhysRevE.102.022307
#
#Specifikt känns det som att model (c) (Social Force Model + Vicsek Model) är exakt vad vi vill ha.
# SFM använder sig av en "desire force" + "social force" (agents vill inte vara för nära varandra) + "wall force" (inte gå in i väg) 
# och Vicsek ger oss att agents vill matcha individer i närheten //Viggo


# Pastear in lite kod snuttar som jag tror kan vara användbara för oss
# %% Interaction function for Vicsek model
import numpy as np
import matplotlib.pyplot as plt

def next_v_vecsek_model(position, v, Rf, L, eta, delta_t):
    """
    Function to calculate the orientation at the next time step.
    
    Parameters
    ==========
    x, y : Positions.
    theta : Orientations.
    Rf : Flocking radius.
    L : Dimension of the squared arena.
    eta : Noise strength.
    """
    
    N = position.shape[0]
    x = position[:, 0]
    y = position[:, 1]

    theta_next = np.zeros(N)

    for j in range(N):
        # No need for replicas in our case
        dist2 = (x - x[j]) ** 2 + (y - y[j]) ** 2 
        nn = np.where(dist2 <= Rf ** 2)[0]
        
        # The list of nearest neighbours is set.
        nn = nn.astype(int)
        theta = np.arctan2(v[nn, 1], v[nn, 0])
        
        # Circular average.
        av_sin_theta = np.mean(np.sin(theta))
        av_cos_theta = np.mean(np.cos(theta))

        theta_avg = np.arctan2(av_sin_theta, av_cos_theta)
        theta_delta = eta * np.random.normal(0, 1) * delta_t
        speed = np.sqrt(v[j, 0] ** 2 + v[j, 1] ** 2)

        theta_next[j] = theta_avg + theta_delta

    v[:, 0] = speed * np.cos(theta_next)
    v[:, 1] = speed * np.sin(theta_next)
          
    return v

def next_v_force_model(position, v, Rf, L, eta, particle_radius, delta_t):
    df = desire_force([0, -50], 1, position, v, 1)
    sf = social_force(position, particle_radius, 1, 1)
    gf = granular_force()
    wf = wall_force()
    gwf = granular_wall_force()

    v = v + (df + sf) # + gf + wf + gwf)*delta_t

    return v


# %% Find neighbours function

def list_neighbours(x, y, N_particles, cutoff_radius):
    '''Prepare a neigbours list for each particle.'''
    neighbours = []
    neighbour_number = []
    for j in range(N_particles):
        distances = np.sqrt((x - x[j]) ** 2 + (y - y[j]) ** 2)
        neighbor_indices = np.where(distances <= cutoff_radius)
        neighbours.append(neighbor_indices)
        neighbour_number.append(len(neighbor_indices))
    return neighbours, neighbour_number


# %% Kod vi kan anpassa om vi vill använda reflekterande väggar 
# Reflecting boundary conditions.

def reflecting_boundary_conditions(positions, L):
    positions[positions < -L/2] = L + positions[positions < -L/2]
    positions[positions > L/2] = L - positions[positions > L/2]

    # if we want to use some other dimension
    # x[x < L/2] = L - x[x < L/2]
    # x[x > L/2] = L - x[x > L/2]
    # y[y < L/2] = L - y[y < L/2]
    # y[y > L/2] = L - y[y > L/2]

    # for j in range(N_particles):
    #     if nx[j] < x_min:
    #         nx[j] = x_min + (x_min - nx[j])
    #         nvx[j] = - nvx[j]

    #     if nx[j] > x_max:
    #         nx[j] = x_max - (nx[j] - x_max)
    #         nvx[j] = - nvx[j]

    #     if ny[j] < y_min:
    #         ny[j] = y_min + (y_min - ny[j])
    #         nvy[j] = - nvy[j]
                
    #     if ny[j] > y_max:
    #         ny[j] = y_max - (ny[j] - y_max)
    #         nvy[j] = - nvy[j]

# %% Funktioner vi vill implementera?

def desire_force(target_position, speed_desire, position, v, relaxation_time):
    direction = target_position - position
    direction = direction / np.linalg.norm(direction)
    v_desire = direction * speed_desire# * np.linalg.norm(v)
    f_desire = (v_desire - v) / relaxation_time

    return  f_desire

def social_force(position, particle_radius, A, B):
    N = position.shape[0]
    
    f_social = np.zeros((N, 2))
    for i in range(N):
        distance = position - position[i]
        force_norm = np.linalg.norm(distance, axis=1)

        nn = (force_norm < particle_radius) & (force_norm > 0)

        force_direction = distance[nn] / force_norm[nn][:, None]

        force_size = A * np.exp((force_norm[nn] - particle_radius) / B)

        nn_forces = (force_size[:, None] * force_direction)

        f_social[i] = np.sum(nn_forces, axis=0)
        
    return f_social

def granular_force():
    return

def wall_force():
    return

def granular_wall_force():
    return

def update_v(position, v, particle_vision, board_size, particle_radius, eta, delta_t):
    # m = 1

    v_vicsek_model = next_v_vecsek_model(position, v, particle_vision, board_size, eta, delta_t)
    v_force_model = next_v_force_model(position, v, particle_radius, board_size, eta, particle_size, delta_t)

    v_combined = (v_vicsek_model + v_force_model) / np.linalg.norm(v_vicsek_model + v_force_model)

    return v_combined

# %%

def init_particles(N, L, v_max=1):
    v = (np.random.rand(N, 2) - 0.5) * v_max
    position = np.random.uniform(-L/2, L/2, (N, 2))

    return position, v

def run_simulation(n_particles, particle_size, board_size, particle_vision, n_itterations, eta, delta_t):
    position, v = init_particles(n_particles, board_size)
    plt.scatter(position[:, 0], position[:, 1], label='Initial position')
    plt.quiver(position[:, 0], position[:, 1], v[:, 0], v[:, 1], color='blue')

    for i in range(n_itterations):
        if i % 1000 == 0:
            print(f'current itteration: {i}')
        v = update_v(position, v, particle_vision, board_size, particle_size, eta, delta_t)
        position = position + v * delta_t
        reflecting_boundary_conditions(position, board_size)

    return position, v
    

# do we want to use the vicsek particle speed?

n_particles = 100
particle_size = 1
board_size = 100 * particle_size
particle_vision = 10
n_itterations = 5000
eta = 0.1
delta_t = 0.1

positions, v = run_simulation(n_particles, particle_size, board_size, particle_vision, n_itterations, eta, delta_t)

plt.scatter(positions[:, 0], positions[:, 1], label='Final position')
plt.quiver(positions[:, 0], positions[:, 1], v[:, 0], v[:, 1], color='red')
plt.xlim(-board_size/2, board_size/2)
plt.ylim(-board_size/2, board_size/2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Particle simulation')
plt.show()



 # %% [markdown]
# (SFM model) velocity(t+dt) = velocity(t) + (1/m)(desire_force + social_force + granular_force + wall_force + granular_wall_force) 
# $$ v^{\textbf{SFM}}(t + \Delta t) = \frac{1}{m}( \bm{F_{D} + F_{S} + F_{G} + F_{W} + F_{GW}})$$
# %%
