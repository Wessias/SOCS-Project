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

def interaction(x, y, theta, Rf, L):
    """
    Function to calculate the orientation at the next time step.
    
    Parameters
    ==========
    x, y : Positions.
    theta : Orientations.
    Rf : Flocking radius.
    L : Dimension of the squared arena.
    s : Discrete steps.
    """
    
    N = np.size(x)

    theta_next = np.zeros(N)
    
    

    for j in range(N):
        # No need for replicas in our case
        dist2 = (x - x[j]) ** 2 + (y - y[j]) ** 2 
        nn = np.where(dist2 <= Rf ** 2)[0]
        
        # The list of nearest neighbours is set.
        nn = nn.astype(int)
        
        # Circular average.
        av_sin_theta = np.mean(np.sin(theta[nn]))
        av_cos_theta = np.mean(np.cos(theta[nn]))
        
        theta_next[j] = np.arctan2(av_sin_theta, av_cos_theta)
                   
    return theta_next


# %% Find neighbours function

.
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
for j in range(N_particles):
    if nx[j] < x_min:
        nx[j] = x_min + (x_min - nx[j])
        nvx[j] = - nvx[j]

    if nx[j] > x_max:
        nx[j] = x_max - (nx[j] - x_max)
        nvx[j] = - nvx[j]

    if ny[j] < y_min:
        ny[j] = y_min + (y_min - ny[j])
        nvy[j] = - nvy[j]
            
    if ny[j] > y_max:
        ny[j] = y_max - (ny[j] - y_max)
        nvy[j] = - nvy[j]

# %% Funktioner vi vill implementera?

def desire_force():
    return

def social_force():
    return

def granular_force():
    return

def wall_force():
    return

def granular_wall_force():
    return

# %% [markdown]
# (SFM model) velocity(t+dt) = velocity(t) + (1/m)(desire_force + social_force + granular_force + wall_force + granular_wall_force) 
# $$ v^{\textbf{SFM}}(t + \Delta t) = \frac{1}{m}( \bm{F_{D} + F_{S} + F_{G} + F_{W} + F_{GW}})$$
# %%
