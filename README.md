# Simulating the Solar System: Orbital Mechanics and Spaceflight Planning

An interactive solar system simulator developed as a Final Year Project at Nanyang Technological University.

The project models the Solar System using Newtonian N-body gravitation and provides an integrated spaceflight planning module for visualizing and executing interplanetary transfers.

## Features

* N-body simulation of all eight planets
* Earth-Moon system simulation
* Procedurally generated asteroid belt
* Leapfrog numerical integration for long-term orbital stability
* Multiple viewing modes (top-down, Earth-Moon, side view)
* Interactive zoom, pan and time controls
* Custom celestial body creation
* Hohmann transfer planning
* Bi-elliptic transfer planning
* Launch window and phase angle visualization
* Real-time delta-v calculations

## Technologies Used

* Python 3.13
* NumPy
* Pygame
* Matplotlib

## Running the Project

Install dependencies:

```bash
pip install numpy pygame matplotlib
```

Run the simulation:

```bash
python main.py
```

## Controls

| Key          | Function                              |
| ------------ | ------------------------------------- |
| 1 / 2 / 3    | Switch view modes                     |
| Q / W        | Adjust simulation speed               |
| Mouse Wheel  | Zoom                                  |
| Click + Drag | Pan                                   |
| Shift + Drag | Create custom body                    |
| S            | Spawn spacecraft                      |
| B            | Toggle Hohmann / Bi-Elliptic transfer |
| Space        | Execute transfer burn                 |
| N            | Reset spacecraft                      |

## Validation

The simulation was validated against accepted planetary orbital periods and demonstrated:

* Orbital period accuracy within 0.64% of accepted values
* Energy drift below 0.009% over 100 simulated years
* Stable long-term orbital behaviour using the Leapfrog integrator

## Final Year Project

School of Electrical & Electronic Engineering
Nanyang Technological University

Author: Gian Gabriel Santos Feliciano

## License

Specify your preferred license here.
