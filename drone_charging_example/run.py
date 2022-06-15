""" 
This file contains a simple experiment run
"""
from typing import Optional
import os
import argparse
import random
import numpy as np
import math

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU in TF. The models are small, so it is actually faster to use the CPU.
import tensorflow as tf

from world import WORLD, ENVIRONMENT  # This import should be first
from components.drone_state import DroneState
from utils.visualizers import Visualizer
from utils import plots
from utils.average_log import AverageLog

from ml_deeco.estimators import ConstantEstimator, NeuralNetworkEstimator
from ml_deeco.simulation import Experiment #, SIMULATION_GLOBALS
from ml_deeco.utils import setVerboseLevel, verbosePrint, Log


# instead of os, to work on every platform
from pathlib import Path
import yaml

def readYaml (file):
    with open(file, "r") as stream:
        try:
            return yaml.load(stream, Loader=yaml.CLoader)
        except yaml.YAMLError as e:
                raise e
                
def run(args):
    """
    Runs `args.iterations` times _iteration_ of [`args.simulations` times _simulation_ + 1 training].
    """

    # Fix random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Set number of threads
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    yamlObject = loadConfig(args)

    folder, yamlFileName = prepareFoldersForResults(args)
 

    averageLog, totalLog = createLogs()
    visualizer: Optional[Visualizer] = None

    def prepareSimulation(iteration, s):
        """Prepares the _Simulation_ (formerly known as _Run_)."""
        components, ensembles = WORLD.reset()
        if args.animation:
            nonlocal visualizer
            visualizer = Visualizer(WORLD)
            visualizer.drawFields()
        return components, ensembles

    def stepCallback(components, materializedEnsembles, step):
        """Collect statistics after one _Step_ of the _Simulation_."""
        for chargerIndex in range(len(WORLD.chargers)):
            charger = WORLD.chargers[chargerIndex]
            accepted = set(charger.acceptedDrones)
            waiting = set(charger.waitingDrones)
            potential = set(charger.potentialDrones)
            WORLD.chargerLogs[chargerIndex].register([
                len(charger.chargingDrones),
                len(accepted),
                len(waiting - accepted),
                len(potential - waiting - accepted),
            ])

        if args.animation:
            visualizer.drawComponents(step + 1)

    def simulationCallback(components, ensembles, t, i):
        """Collect statistics after each _Simulation_ is done."""
        totalLog.register(collectStatistics(t, i))
        WORLD.chargerLog.export(folder / f"charger_logs/{yamlFileName}_{t + 1}_{i + 1}.csv")

        if args.animation:
            verbosePrint(f"Saving animation...", 3)
            visualizer.createAnimation(folder / f"animations/{yamlFileName}_{t + 1}_{i + 1}.gif")
            verbosePrint(f"Animation saved.", 3)

        if args.plot:
            verbosePrint(f"Saving charger plot...", 3)
            plots.createChargerPlot(
                WORLD.chargerLogs,
                folder / f"charger_logs/{yamlFileName}_{str(t + 1)}_{str(i + 1)}.png",
                f"World: {yamlFileName}\n Run: {i + 1} in training {t + 1}\nCharger Queues")
            verbosePrint(f"Charger plot saved.", 3)

    def iterationCallback(t):
        """Aggregate statistics from all _Simulations_ in one _Iteration_."""

        # calculate the average rate
        averageLog.register(totalLog.average(t * args.simulations, (t + 1) * args.simulations))

        for estimator in experiment.estimators:
            estimator.saveModel(t + 1)

    experiment = Experiment(
        prepareSimulation,
        iterationCallback=iterationCallback,
        simulationCallback=simulationCallback,
        stepCallback=stepCallback,
        iterations=args.iterations,
        simulations=args.simulations,
        steps=ENVIRONMENT.maxSteps,
        baselineEstimator=0,
        configFile=args.config,
        baseFolder=folder)

    # 
    # WORLD.waitingTimeEstimator = experiment.waitingTimeEstimator
    # WORLD.batteryEstimator = experiment.batteryEstimator
    # WORLD.waitingTimeBaseline = 0
    WORLD.experiment = experiment
    WORLD.initEstimators(experiment)
    experiment.run()

    totalLog.export(folder / f"{yamlFileName}.csv")
    averageLog.export(folder / f"{yamlFileName}_average.csv")

    plots.createLogPlot(
        totalLog.records,
        averageLog.records,
        folder / f"{yamlFileName}.png",
        f"World: {yamlFileName}",
        (args.simulations, args.iterations)
    )
    return averageLog


def loadConfig(args):
    # load config from yaml
    yamlObject = readYaml(args.input)

    yamlObject['chargerCapacity'] = findChargerCapacity(yamlObject)
    yamlObject['totalAvailableChargingEnergy'] = min(
        yamlObject['chargerCapacity'] * len(yamlObject['chargers']) * yamlObject['chargingRate'],
        yamlObject['totalAvailableChargingEnergy'])

    ENVIRONMENT.loadConfig(yamlObject)

    return yamlObject


def findChargerCapacity(yamlObject):
    margin = 1.3
    chargers = len(yamlObject['chargers'])
    drones = yamlObject['drones']

    c1 = yamlObject['chargingRate']
    c2 = yamlObject['droneMovingEnergyConsumption']

    return math.ceil(
        (margin * drones * c2) / ((chargers * c1) + (chargers * margin * c2))
    )


def createLogs():
    totalLog = AverageLog([
        'Active Drones',
        'Total Damage',
        'Alive Drone Rate',
        'Damage Rate',
        'Charger Capacity',
        'Train',
        'Run',
    ])
    averageLog = AverageLog([
        'Active Drones',
        'Total Damage',
        'Alive Drone Rate',
        'Damage Rate',
        'Charger Capacity',
        'Train',
        'Average Run',
    ])
    return averageLog, totalLog


def prepareFoldersForResults(args):
    # prepare folder structure for results
    yamlFileName = os.path.splitext(os.path.basename(args.input))[0]

    folder = Path() / 'results' / args.output
    folder.mkdir(parents=True, exist_ok=True)

    animations = Path() / folder / 'animations'
    animations.mkdir(parents=True, exist_ok=True)
  
    chargerLogs = Path() / folder / 'charger_logs'
    chargerLogs.mkdir(parents=True, exist_ok=True)

    return folder, yamlFileName



def collectStatistics(train, iteration):
    MAXDRONES = ENVIRONMENT.droneCount if ENVIRONMENT.droneCount > 0 else 1
    MAXDAMAGE = sum([field.allCrops for field in WORLD.fields])

    return [
        len([drone for drone in WORLD.drones if drone.state != DroneState.TERMINATED]),
        sum([field.damage for field in WORLD.fields]),
        len([drone for drone in WORLD.drones if drone.state != DroneState.TERMINATED]) / MAXDRONES,  # rate
        sum([field.damage for field in WORLD.fields]) / MAXDAMAGE,  # rage
        ENVIRONMENT.chargerCapacity,
        train + 1,
        iteration + 1,
    ]



def validateArguments(args):
    if args.config:
        configPath = Path(args.config)
        if not configPath.exists():
            raise IOError(f"such file {args.config} does not exist")
        simulationConfigObject = readYaml(configPath)
        
    else:
        simulationConfigObject = {}

    # override
    for argument in args.__dict__ :
        if args.__dict__[argument] is not None:
            simulationConfigObject[argument] = args.__dict__[argument]
    
    # set default values
    for argument , default in zip(
        [ 'verbose', 'seeds', 'threads'],
        [0, 42, 4]):
        if argument not in simulationConfigObject:
            simulationConfigObject[argument] = default

    return simulationConfigObject

def main():
    parser = argparse.ArgumentParser(
        # TODO add proper description
        description='')
    parser.add_argument(
        'input', type=str, 
        help='YAML world address to be run.',
    )
    # ML-DEECO Config file and overrides, if NONE, it loads from the --config <file.yaml> file
    parser.add_argument(
        '-c', '--config', type=str, 
        help='the configuration file', 
        required=False, default=None
    )
    parser.add_argument(
        '-i', '--iterations', type=int, 
        help='The number of iterations (trainings) to be performed.', 
        required=False,  default=None
    )
    parser.add_argument(
        '-s', '--simulations', type=int, 
        help='The number of simulation runs per iteration.', 
        required=False, default=None
    )
    # drone example arguments
    parser.add_argument(
        '-v', '--verbose', type=int, 
        help='the verboseness between 0 and 4.', 
        required=False, default=None
    )
    parser.add_argument(
       '--seed', type=int, 
       help='Random seed.', 
       required=False, default=42)

    parser.add_argument(
        '--threads', type=int, 
        help='Number of CPU threads TF can use.', 
        required=False, default=4)
    
    parser.add_argument(
        '-o', '--output', type=str, 
        help='the output folder', 
        required=False, default="output"
    )
    parser.add_argument(
        '-a', '--animation', action='store_true', 
        help='toggles saving the final results as a GIF animation.',
        required=False,  default=False
    )
    parser.add_argument(
        '-p', '--plot', action='store_true', 
        help='toggles saving the plot results.',
        required=False,  default=False
    )
    args = parser.parse_args()
    config = validateArguments(args)

    # TODO optional, to use only one dictionary we combine both into args
    for argument in config:
        args.__dict__[argument] = config[argument]

    setVerboseLevel(args.verbose)
    run(args)


if __name__ == "__main__":
    main()
