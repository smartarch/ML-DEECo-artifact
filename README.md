# ML-DEECo artifact

This is an artifact repository for ACSOS 2022 paper **ML-DEECo: a Machine-Learning-Enabled Framework for Self-organizing Components** by *Michal Töpfer*, *Milad Abdullah*, *Martin Kruliš*, *Tomáš Bureš*, and *Petr Hnětynka*.

This repository features implementation of the **ML-DEECo** framework as well as two examples of simulations implemented using the framework. 

* [`ml_deeco`](https://github.com/smartarch/ML-DEECo/tree/artifact) &ndash; a framework for implementation of simulations of adaptive distributed systems with native support for machine learning
* [`drone_charging_example`](drone_charging_example) &ndash; simulation of a smart farming system for protection of fields against birds using battery-powered flying drones
* [`smart_factory`](smart_factory) &ndash; simulation of a smart factory with adaptive security rules and machine-learning-based estimates of workers' tardiness

### Prerequisites

The ML-DEECo itself requires Python 3.6+ and the following packages installable via `pip`: `numpy`, `seaborn`, `matplotlib`, and `pyyaml`. The typical expected application is to include the ML-DEECo repository as a submodule of your simulation repository (as we did in this artifact).

The artifacts may require additional packages based on which ML library they are using.

