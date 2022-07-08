# ML-DEECo artifact

This is an artifact repository for ACSOS 2022 paper **ML-DEECo: a Machine-Learning-Enabled Framework for Self-organizing Components** by *Michal Töpfer*, *Milad Abdullah*, *Martin Kruliš*, *Tomáš Bureš*, and *Petr Hnětynka*.

This repository features implementation of the **ML-DEECo** framework as well as two examples of simulations implemented using the framework. 

* [`ml_deeco`](https://github.com/smartarch/ML-DEECo/tree/artifact) &ndash; a framework for implementation of simulations of adaptive distributed systems with native support for machine learning
* [`drone_charging_example`](drone_charging_example) &ndash; simulation of a smart farming system for protection of fields against birds using battery-powered flying drones
* [`smart_factory`](smart_factory) &ndash; simulation of a smart factory with adaptive security rules and machine-learning-based estimates of workers' tardiness

### Prerequisites

The ML-DEECo itself requires Python 3.6+ and the following packages installable by

```
pip install numpy seaborn matplotlib pyyaml
```

The typical expected application is to include the ML-DEECo repository as a submodule of your simulation repository (as we did in this artifact).

The artifacts may require additional packages based on which ML library they are using (detailed in their readme files).


### Documentation

The documentation is scattered in several readme files:

* [ML-DEECo readme](https://github.com/smartarch/ML-DEECo/blob/master/README.md) &ndash; describes the installation and usage of the framework itself. Also explains the components and ensembles as well as overall architecture.
* [ML-DEECo simple-example readme](https://github.com/smartarch/ML-DEECo/blob/master/examples/simple_example/README.md) &ndash; explains the simple example which can be used as a bootstrap code to quickly sketch your own simulation
* [Drone charging example readme](drone_charging_example) &ndash; contains the installation and usage guidelines for the smart farming example, expected results, configuration details, and component model overview
* [Factory workers example readme](smart_factory) &ndash; contains the installation and usage guidelines for the factory access control example, expected results, and simulation architecture overview
