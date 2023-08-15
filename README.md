# HPC-fluid-mechanics-with-python
_The code base for the course HPC-Fluid-Mechanics-with-Python at Uni-Freiburg._

## Introduction
This repository consists of the code base to use MPI to simulate Sliding Lid induced cavity by parallelization. 

## Code breakdown

### Folder structure

The folder **HPCsubmission** has the code base to run the simulation code in parallel.

<pre>
```
HPCsubmission/
│
├── simulations/    [Code to simulate different flow conditions.]
│   ├── couette_flow.py
│   ├── poiseuille_flow.py
│   ├── shear_wave_decay.py
│   ├── sliding_lid.py
│   └── sliding_lid_parallel.py
|
├── utils/    [Code to for utility functions and classes.]
│   ├── boundaries.py    [Code to realize boundary conditions - rigid and moving wall on all four boundaries.]
│   ├── constants.py    [Constants used across the module.]
│   ├── fluid_mechanics.py    [Code for all basic operations of fluid simulation - streaming, collision, equilibrium, setting pressure gradient, calculating density and velocity and others.]
│   ├── parallelization.py    [Code to handle parallelization using MPI.]
│   ├── plots.py    [Code to plot graphs.]
│   └── utils.py    [Code for other utility functions.]
|
├── utils/    [Code to simulate different flow conditions.]
│   ├── couette_flow.py
|
├── scripts/    [Contains .sh scripts that handle different execution scenarios.]
│   ├── bwUni_run_scale_test.sh    [(bwUni) Runs Sliding Lid on different grid sizes on different processor counts.]
│   ├── bwUni_run_sliding_lid_Re.sh    [(bwUni) Runs Sliding Lid on a range of Re by varying initial velocity or omega.]
│   ├── bwUni_run_sliding_lid.sh    [(bwUni) Runs Sliding Lid on single configuration.]
│   └── run_simulations.sh    [Run all files in /simulations for one configuration and generate all inference plots where applicable.]
│
├── plots/    [All plots generated during simulation.]
│
└── reports/    [All reports generated during simulation.]
```
</pre>

The folder **Milestones-dev** has code structured along the development phases of the module.

The folder **Report** has the metadata of the final report.

### Running simulations

Clone the git repository in the required environment. 
Create a virtual environment and install requirements.txt. 
Use HPCsubmission/ as the pwd.

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd HPCsubmission
```

Each file in the `simulations/` has an argument parser that can be used to configure the different parameters of the simulation.
The `scripts/run_simulations.sh` file has a sample run configured for each simulation, including the configurations of multiple runs where the performance/output is compared against multiple parameter values (For example, Omega comparison in Shear Wave, etc).

To run a `.sh` file,
```
chmod +x name_of_file.sh
./name_of_file.sh [-parameters]
```
#### Parallelization - On the bwUniCluster

The `scripts/bwUni_*.sh` files can be run directly on the bwiUniCluster. 
Each file has the simulation configuration used to generate the plots in the report. 
They can be edited to alter the parameters of the run.

#### Parameters for the simulations

| Argument | Description | Sliding Lid (Parallel) | Sliding Lid | Poiseuille Flow | Couette Flow | Shear Wave Decay |
|-----|-----|-----|-----|-----|-----|-----|
| -nx | Grid size along X-axis, default 300 | &#x2713; | &#x2713;| &#x2713;| &#x2713;| &#x2713;|
| -ny | Grid size along Y-axis, default 300 | &#x2713; | &#x2713;| &#x2713;| &#x2713;| &#x2713;|
| -o | Omega, default 1.2 | &#x2713; | &#x2713;| &#x2713;| &#x2713;| &#x2713;|
| -u | Horizontal velocity of the top wall, default 0.01 | &#x2713;| &#x2713;||&#x2713;||
| -re_min | Minimum Reynolds number for comparison run, default 100 | &#x2713;| | | | |
| -re_max | Maximum Reynolds number for comparison run, default 1000 | &#x2713;| | | | |
| -re_count | Number of Reynolds number values to pick between minimum and maximum range, default 10|&#x2713;|| | | |
| -rho_in | Density at inlet, default 1.001 |||&#x2713;|||
| -rho_out | Density at outlet, default 0.999 |||&#x2713;|||
|-u_epsilon| Horizontal velocity factor, default None |||||&#x2713;|
|-rho_o | Initial density, default None | ||||&#x2713;|
|-rho_epsilon| Horizontal density factor, default None |||||&#x2713;|
| -omega_min | Minimum omega for comparison run, default 0.1 |||||&#x2713;|
| -omega_max | Maximum omega for comparison run, default 1.5 |||||&#x2713;|
| -omega_count | Number of omega values to pick between minimum and maximum range, default 15 |||||&#x2713;|
| -run_multiple | Indicate parameter to vary while simulating for multiple Re values. Options - velocity, omega| &#x2713;||
| -nt | Timesteps for simulation, default varies | &#x2713;| &#x2713;|&#x2713;|&#x2713;|&#x2713;|
| -nt_log | Timestep interval to record/plot values, default varies| &#x2713;| &#x2713;|&#x2713;|&#x2713;|&#x2713;|
| -time_log_path | Path to file to log execution time, no default | &#x2713;| &#x2713;||||
| -plot_grid | Number of density plots in one row of the subplotgrid, default 5 | &#x2713;|&#x2713;|&#x2713;|
| -no_plot | Flag on whether to plot (action) |&#x2713;|||||



