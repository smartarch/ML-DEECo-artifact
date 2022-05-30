# Security rules for a smart factory

In this example, we use the ML-DEECo framework to implement a simulation of a smart factory with adaptive secutrity rules. 

In this document, a simple guide to run the example is presented:

- [Installation](#installation)
- [Usage](#usage)

## Installation

The example requires Python 3 with `numpy`, `tensorflow`, `matplotlib` and `seaborn` libraries installed.

Furthermore, `ml_deeco` shall be installed to run the simulation (the `--editable` switch can be omitted if one does not plan to change the code of ML-DEECo):

```
pip install --editable ../ml_deeco
```

## Usage

The experiments presented in the paper were run for three iterations:

```
py run.py -o results -b 16 -v 2 -i 3 -p
```

* `-o results` defines the folder to save results to
* `-b 16` sets the baseline for worker cancellation (to 16 minutes before the shift starts) 
* `-v 2` sets a reasonable the verbosity level
* `-i 3` sets the three iterations
* `-p` enables displaying the plots with results

## Simulation overview

### [Components](components.py)

`Door`, `Dispenser` (of protective headgear), `Factory`, `WorkPlace`, `Shift`, `Worker`

### [Ensembles](ensembles.py)

#### Security-related

* `ShiftTeam` - all working workers in one shift (those not cancelled, incl. called standbys)
* `AccessToFactory`
* `AccessToDispenser`
* `AccessToWorkPlace`

#### Late workers replacement

* `CancelLateWorkers` &ndash; ensemble with ML estimate
* `ReplaceLateWithStandbys` &ndash; possible use of heuristic for matching standbys to shifts
