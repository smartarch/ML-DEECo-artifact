# Drone Charging Example

In this broad example, we provide a simulation that runs a system which protects field of crops against flocks of birds using drones. In this document, a complete guide to run the example is presented:

- [Installation](#installation)
- [Usage](#usage)
- [YAML Experiments](#yaml-experiments)
- [Simulation &ndash; Components and Ensembles](#simulation-ndash-components-and-ensembles)

## Installation
To run the example, some libraries must be installed. The installation requires `Python 3` and `pip`. If you have `pip` installed skip to - [Package Installation](#package-installation).

To install `pip` follow the following instructions:

### Install pip on Debian/Ubuntu
```
apt install python3-pip
```
### Install pip on CentOS, RHEL, Fedora
```
yum -y update
yum install python-pip
```
### Install pip on Arch Linux
```
pacman -S python-pip
```
### Install pip on openSUSE
```
zypper install python3-pip
```
### Install pip on Windows
First download https://bootstrap.pypa.io/get-pip.py and copy/save it in a folder. Then run the following command:
```
python <path-to-get-pip.py>/get-pip.py
```
### Package Installation
All the required packages and libraries are stored in [requirements.txt](requirements.txt).
> :warning: the requirements include **Tensorflow**, and the size of its dependencies could reach **1.5 GB**

#### Step 1: install all the packages by running the following command:
```
pip install -r requirements.txt
```
#### Step 2: install ML-DEECo using `pip` (the `--editable` switch can be omitted if one does not plan to change the code of ML-DEECo):
```
pip install --editable ../ml_deeco
```

## Usage
The simulation is configured with a YAML file. A few examples can be found in [experiments](experiments). The results will be stored in `results` folder. For a quick run, simply execute the following command:

```
py run.py experiments/12drones.yaml
```

The above command runs the simulation once and store the results in results folder. To run the simulation multiple times use `-n <NUMBER>`, and to view a chart at the end of run, use `-c`.


```
py run.py experiments/12drones.yaml -n 5 -c
```

Please note that the simulation will not train unless `-t <NUMBER>` is set, and it must be set more than 1. 

To Observe the outcomes during runtime, one can use `-v <NUMBER>` which sets the verboseness between 0-4. 

The following command will run `12drones.yaml` for 4 iterations, where each iteration consists of running the simulation 5 times and then training the estimator. Therefore, it will run and collect the results of total of 20 simulation runs, with verboseness level of 2. The first 5 simulation runs will use no estimation at all.

```
py run.py experiments/12drones.yaml -n 5 -t 4 -v 2 -c
```

The above command will spend some time to finalize and store the results. Should the YAML file not change, the graph will look like the following one:

![12-drone-t3-n5](results/output/12drones_neural_network.png)

As the behavior of the system is influenced by the estimates, the data collected in the second iteration will be different from the first iteration. To prevent feedback loops, the `-d ` switch can be used to accumulate training data from all previous iterations (without it, only the data from the current iteration are used for training). Note that this will increase the time needed to train the estimator, because more examples are used for training. 

Additionally, one might try the experiment with different test split (using `--test_split <RATE>`), different hidden layers (using `--hidden_layers [<NUMBER>,<NUMBER>]`) or a random seed (using `-s <NUMBER>`). To specify a subfolder in results to store all the results use `-o <PATH>`; if the folder does not exist it will be created.

```
py run.py experiments/12drones.yaml -n 5 -t 6 -d -o test_12_drones --hidden_layers 126 126 -c --test_split 0.4 --seed 423
```
> The above command will run the simulation 30 times (6 x 5) with training every 5 runs, accumulating all data, saving 6 models in results/test_12_drones/neural_network folder. The model will have two hidden layers with 126 neurons, splitting 40% data for validation and the random seed to initialize random objects is 4232. The below chart shows the results:
 
![12-drone-t63-n5](results/test_12_drones/12drones_neural_network.png)

It could be observed that with tuning neural network parameters, the outcome varies, and it could be improved. The models are stored in the results/test_12_drones/neural_network as `h5` files. They are portable models that could be used with the same simulation (using `-l <PATH-TO-MODEL>`), but perhaps with different size of flocks of birds (overriding the YAML configuration with `-x <NUMBER>`). Additionally, a visualizer is attached to the simulation, and it can be toggled with `-a`.
> :warning: using *`-a`* with multiple runs will produce GIF animations for all of them, and it might take excessive storage and time.

```
py run.py experiments/12drones.yaml -l results/test_12_drones/neural_network/model_6.h5  -x 20 -a -o vis_12drones
```
The above command will produce animated scenario of the 12drone world. The file is located in vis_12drones/animations.

![12-drone-animated](results/vis_12drones/animations/12drones_1_1.gif)

For further run options, 
<pre>
usage: run.py [-h] [-x BIRDS] [-n NUMBER] [-t TRAIN] [-o OUTPUT] [-v VERBOSE]
              [-a] [-c] [-w {baseline,neural_network}] [-d]
              [--test_split TEST_SPLIT]
              [--hidden_layers HIDDEN_LAYERS [HIDDEN_LAYERS ...]] [-s SEED]
              [-b BASELINE] [-l LOAD] [-e] [--threads THREADS]
              input
         

  -h, --help            show this help message and exit
  -x BIRDS, --birds BIRDS 
                        number of birds, if no set, it loads from yaml file.
  -n NUMBER, --number NUMBER
                        the number of simulation runs per training.
  -t TRAIN, --train TRAIN
                        the number of trainings to be performed.
  -o OUTPUT, --output OUTPUT
                        the output folder
  -v VERBOSE, --verbose VERBOSE
                        the verboseness between 0 and 4.
  -a, --animation       toggles saving the final results as a GIF animation.
  -c, --chart           toggles saving and showing the charts.
  -w {baseline,neural_network}, --waiting_estimation {baseline,neural_network}
                        The estimation model to be used for predicting charger
                        waiting time.
  -d, --accumulate_data
                        False = use only training data from last iteration.
                        True = accumulate training data from all previous
                        iterations.
  --test_split TEST_SPLIT
                        Number of records used for evaluation.
  --hidden_layers HIDDEN_LAYERS [HIDDEN_LAYERS ...]
                        Number of neurons in hidden layers.
  -s SEED, --seed SEED  Random seed.
  -b BASELINE, --baseline BASELINE
                        Constant for baseline.
  -l LOAD, --load LOAD  Load the model from a file.
  -e, --examples        Additional examples.
  --threads THREADS     Number of CPU threads TF can use.

  </pre>  

  ## YAML Experiments
  The experiment world configuration is specified in a YAML input file. To keep the variable domain in a manageable rate, most of the tests were performed on the basis of similar world configurations, but changing number of drones, birds, charger and the capacity of charging rate. The following table summarizes the possible configurations:

|                    Configuration | Description                                                            | Example                      |
|---------------------------------:|:-----------------------------------------------------------------------|:-----------------------------|
|                           drones | The number of drones.                                                  | `10`                         |
|                            birds | The number of birds.                                                   | `85`                         |
|                         chargers | List of chargers points on the map.                                    | `[[17,29],[28,13]]`          |
|                           fields | List of field rectangles (top-left and bottom-right) on the map.       | `[[3,4,21,18],[35,7,48,36]]` |
|                         maxSteps | Time steps that the simulation will be running.                        | `500`                        |
|                         mapWidth | The width of the map.                                                  | `50`                         |
|                        mapHeight | The height of the map.                                                 | `50`                         |
|                      droneRadius | The protecting radius (points) of the drones.                          | `5`                          |
|                       droneSpeed | The speed of drones                                                    | `1`                          |
|                        birdSpeed | The speed of birds.                                                    | `1`                          |
|                     chargingRate | The rate of charging battery by a charger per time step.               | `0.04`                       |
|     totalAvailableChargingEnergy | The total available charging rate for all chargers (set 1 for full).   | `0.08`                       |
|     droneMovingEnergyConsumption | The energy drones spend by moving.                                     | `0.01`                       |
| droneProtectingEnergyConsumption | The energy drones spend by standing.                                   | `0.005`                      |
|            droneBatteryRandomize | If set > 0, the drones will start with different battery at beginning. | `0`                          |
|       droneStartPositionVariance | if set > 0, the drones will start from random places in the map.       | `0`                          |

## Simulation &ndash; Components and Ensembles

The simulation runs a number of stateful and stateless component that perform in each time step:

### Overview

* Components (stateful)
    * Agents (stateful components that can move)
        * [Drone](#Drone)
        * [Bird](#Bird) &ndash; represents a flock of birds
    * [Charger](#Charger)
* Ensembles (stateless)
    * [Drone Charging](#Drone-Charging)
    * [Field Protection](#Field-Protection)
* World configuration
  * World, Environment &ndash; hold the world configuration.
  * [Field](#field) 

#### Utilities

* Run file ([`run.py`](run.py))
* Plots generator ([`utils/plots.py`](utils/plots.py))
* Average Log ([`utils/average_log.py`](utils/average_log.py)) &ndash; logging of simulation progress.
* Visualizer ([`utils/visualizers.py`](utils/visualizers.py)) &ndash; animations generator.

### Field
The field class instances represent the agricultural fields on the map. Each field has a number of crops to be protected. The fields are divided into places (based on the protection radius of drones). To simplify the simulation, the fields are presented as rectangles,: `[x1, y1, x2, y2]`
~~~text
    (x1,y1) .__________
            |          |
            |__________|.(x2,y2). 
~~~

### Drone
The drones protect the fields from birds by moving to the field and scaring the flocks of birds away. In programming perspective, drone components have access to shared `WORLD` and they can find the position to protect. In a real-life scenario, it is assumed that additional sensors will perform the detection of birds, and it can be read from them. The drones have the following states:

#### Drone State
* **IDLE**: default initial state of a drone.
* **PROTECTING**: the drone is protecting a field.
* **MOVING_TO_CHARGING**: the drone is moving towards a charger.
* **CHARGING**: the drone is being charged.
* **TERMINATED**: the battery level of the drone is below 0, and it does not operate anymore (unrecoverable).

### Bird
The birds are the threats to the crops in the fields. They find undamaged crops and eat them in one or multiple visits (2 in our case). They flee to random place of the map (which are not fields) if they see a drone around. The behavior of birds is randomized which influences the results of the simulation, thus one ought to attempt multiple runs and average the results. The birds state goes as the following:

#### Bird State
* **IDLE**: default state of birds, when they are away from fields.
* **ATTACKING**: a state where a bird has targeted a field, and it is attacking it.
* **FLEEING**: a state where a bird is flying away from drones.

### Charger
chargers are the components that provide energy to the drones. The capacity of charger is calculated according to the number of drones and chargers available. The charging rate and saturation (available charging rate) is configured in YAML files. For current existing configuration (assuming energy provided is `0.04`), charging is set as:

| Experiment | Chargers | Calculated Capacity | Maximum Charging Rate |
|-----------:|:--------:|:-------------------:|:---------------------:|
|   8 Drones |    3     |          1          |         0.12          |
|  10 Drones |    3     |          1          |         0.12          |
|  12 Drones |    3     |          1          |         0.12          |
|  16 Drones |    2     |          2          |         0.16          |
|  20 Drones |    3     |          2          |         0.24          |
|  24 Drones |    3     |          2          |         0.24          |


### Field Protection

The field protection ensemble manages the protection of the fields against the birds. There is one instance of the `FieldProtection` ensemble for each field on the map. The closest IDLE drone to each field becomes member of the ensemble and is assigned the task to protect the field. The fields are sorted based on the number of unprotected places they have (which is the total number of places minus the number of protecting drones); therefore, the priority goes as `unprotected_places / places`. In each time step the ensembles are resorted, and re-materialized to find the idle drones to protect the fields. 

### Drone Charging

Our example has three types of ensembles to perform the drone charging. The `DroneChargingPreAssignment` partitions the drones among the chargers, so that each drone is assigned to the closest charger. The `DroneChargingAssignment` selects the drones in need of charging, and the `AcceptedDronesAssignment` groups the drones which were assigned a slot at the charger &ndash; those start moving to the charger and start charging when they get there. Once on the charger, the drone will charge until its battery is full and then its state changes to IDLE. A more detailed description of the ensembles is given later in the text.

The following graph shows the cycle of a drone and how ensembles (colored as light orange) change course of the drone. However, an ensemble does not directly change the state of a drone, it rather sets for example the `target_field` attribute of the drone to command it to protect that field. Another example is that a drone could be in need of charging, but the charger is busy, so the drone will keep its current state (perhaps protecting the field) till the accepting ensemble signals that the charger is free now. For a better performance, the drones will start moving, when they know that by the time they reach the charger, the charger will be free.

![drone-cycle](components/drone-cycle.svg)

The ensemble definitions are the same in both the baseline and machine-learning-based approaches. The key difference between both approaches is the computation of the waiting time, which is used for deciding whether a drone needs charging. In the baseline, the drones do not know how long they will probably wait for a free charging slot after they close enough to the charger. However, in ML-based, the waiting time is predicted and the drones add that waiting time to the time they need to fly to charger. Therefore, even with a sufficient battery, they will move toward the chargers sooner than usual and this helps them to survive. The charging ensembles are:

#### DroneChargingPreAssignment

Finds the closest charger to a drone in each time step. Technically, we have an instance of `DroneChargingPreAssignment` for each charger, so it groups the drones for which this charger is the closest.

#### DroneChargingAssignment

The ensemble groups the drones which need charging (again, we have an instance for each charger, and we only consider the drones already selected by the corresponding `DroneChargingPreAssignment`). The decision whether a drone needs charging is done as follows:

* Baseline: `battery - energy to fly to charger < threshold`
* ML-based: `battery - (energy to fly to charger + waiting time estimate) < threshold`

To estimate the waiting time, we use a neural network (specified and trained using the ML-DEECo framework) with the following input features:

* battery
* drone_state
* charger_distance
* accepted_drones_count
* charger_capacity
* neighbor_drones_average_battery
* neighbor_drones
* potential_drones
* accepted_drones_missing_battery
* charging_drones_count
* charging_drones_missing_battery
* potential_drones_with_lower_battery
* waiting_drones_count
* waiting_drones_with_lower_battery

#### AcceptedDronesAssignment

The last ensemble selects those drones from the members of `DroneChargingAssignment` for which there is a free slot at the charger. More precisely, we accept a drone for charging, if there will be a free slot at the charger at the time the drone arrives there (assuming it starts flying towards the charger now). As soon as a drone is accepted, is starts moving toward the charger and then gets fully charged.
