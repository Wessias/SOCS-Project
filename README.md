# Evacuation Dynamics Simulation

![Simulation GIF](simulation_run.gif)

This project models the dynamics of pedestrian evacuation using a hybrid approach that combines the **Social Force Model** and the **Vicsek Model**. It was developed as part of a university course on Simulation of Complex Systems and explores how different factors affect crowd behavior during emergency evacuations.

## Project Description

The simulation captures realistic behaviors of individuals in a crowded environment—such as a concert venue or sports arena—as they attempt to exit through one or more doors. It investigates how evacuation time and crowd flow are influenced by:

- **Door size and positioning**
- **Visibility (line-of-sight to exits)**
- **Crowd density**
- **Individual movement tendencies (alignment and personal space)**

Each agent is affected by both social forces (e.g., avoiding collisions and walls) and behavioral alignment with neighbors, simulating emergent group behavior under stress. 

## Key Features

- 🧠 **Behavioral modeling** using physics-inspired social force interactions and local velocity alignment  
- 🎞️ **Real-time animation** of the crowd evacuation process  
- 🧪 **Experimental flexibility** with adjustable parameters for scenario testing  
- 📊 **Data collection** of escape times, density effects, and flow patterns  

## Insights Gained

- Increasing door width improves evacuation only up to a certain threshold.
- Partial visibility (some agents seeing the exit) leads to nearly optimal evacuation—full visibility isn't required.
- Dense crowds increase evacuation times and unpredictability, emphasizing the importance of design and planning in real-world venues.

## Why This Project Matters

This project demonstrates how agent-based simulations can inform safer architectural designs and emergency response strategies. It’s an integration of modeling, simulation, and visualization—illustrating how complex systems can be better understood through computation.

## Technologies Used

- **Python**
- **NumPy**, **SciPy** — numerical computations
- **Matplotlib** — visual animation of agent movement

