# Security rules for a smart factory

In this example, we use the ML-DEECo framework to implement a simulation of a smart factory with adaptive security rules. 

In this document, a simple guide to run the example is presented:

- [Installation](#installation)
- [Usage](#usage)
- [Simulation overview](#simulation-overview)

## Installation

The example requires Python 3 with `numpy`, `tensorflow`, `matplotlib` and `seaborn` libraries installed.

Furthermore, `ml_deeco` shall be installed to run the simulation:

```
pip install ../ml_deeco
```

## Usage

The experiments presented in the paper were run for three iterations:

```
py run.py -v 2 -p
```

* `-v 2` sets a reasonable the verbosity level
* `-p` enables displaying the plots with results

The rest of the parameters of the simulation are defined in the configuration YAML files: [`factory.yaml`](experiments/factory.yaml) (configures the start time of shifts, arrival times of workers, etc.), [`estimators.yaml`](experiments/estimators.yaml) (definition of the ML model and folder for the results). 

## Simulation overview

In this example, we model a simulation of a smart factory with security rules to access doors, etc. The factory has multiple working places, each with a team of workers. The teams of workers work on projects for several customers. The workers from one team are thus allowed only to enter their workplace and cannot enter the other workplaces (to protect the intellectual property of the customers).

In the morning, a shift of workers is assigned to each workplace. These workers are granted permission to enter the factory 30 minutes before their shift starts. Then they have to take a protective headgear from a dispenser inside the factory. Only with the headgear, they are allowed to enter their workplace.

Apart from the workers assigned to the shift, there are also several standby workers in case some assigned workers do not arrive in time. When a worker does not arrive at the factory 16 minutes before their shift starts, they get canceled for the day, and a standby worker is called to replace them. We assume that it will take approximately 30 minutes before the standby arrives. The cancellation of the worker includes revoking the permission to enter the factory and granting that permission to the standby.

The scenario is dynamic as replacing the workers in the shift with standbys requires changing the permissions to enter the factory and the workplaces.

### [Components](components.py)

We consider the following component to represent all entities in the system (including those which are not physical objects, such as shifts of workers):

`Door`, `Dispenser` (of protective headgear), `Factory`, `WorkPlace`, `Shift`, `Worker`

### [Ensembles](ensembles.py)

#### Security-related

The members of the ensemble are given a certain security permission (e.g., to enter the factory).

* `ShiftTeam` - all working workers in one shift (those not cancelled, incl. called standbys)
* `AccessToFactory`
* `AccessToDispenser`
* `AccessToWorkPlace`

#### Late workers replacement

We show a use of ML in this use case by replacing the rigid rule of canceling a worker 16 minutes before the shift starts with a dynamic threshold based on a machine-learning-based estimate.

* `CancelLateWorkers` &ndash; ensemble with ML estimate
* `ReplaceLateWithStandbys` &ndash; possible use of heuristic for matching standbys to shifts
