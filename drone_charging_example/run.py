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

from ml_deeco.simulation import Experiment, Configuration #, SIMULATION_GLOBALS
from ml_deeco.utils import setVerboseLevel, verbosePrint, Log


# instead of os, to work on every platform
from pathlib import Path

                
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

    # finds the local attributes in the Arguments
    localWorld = createLocalWorld(args)
    folder, outputFileName = prepareFoldersForResults(args)
 
    averageLog, totalLog = createLogs()
    visualizer: Optional[Visualizer] = None

    def prepareSimulation(iteration, s):
        """Prepares the _Simulation_ (formerly known as _Run_)."""
        components, ensembles = WORLD.reset()
        if localWorld['animation']:
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

        if localWorld['animation']:
            visualizer.drawComponents(step + 1)

    def simulationCallback(components, ensembles, t, i):
        """Collect statistics after each _Simulation_ is done."""
        totalLog.register(collectStatistics(t, i))
        WORLD.chargerLog.export(folder / f"charger_logs/{outputFileName}_{t + 1}_{i + 1}.csv")

        if localWorld['animation']:
            verbosePrint(f"Saving animation...", 3)
            visualizer.createAnimation(folder / f"animations/{outputFileName}_{t + 1}_{i + 1}.gif")
            verbosePrint(f"Animation saved.", 3)

        if args.plot:
            verbosePrint(f"Saving charger plot...", 3)
            plots.createChargerPlot(
                WORLD.chargerLogs,
                folder / f"charger_logs/{outputFileName}_{str(t + 1)}_{str(i + 1)}.png",
                f"World: {outputFileName}\n Run: {i + 1} in training {t + 1}\nCharger Queues")
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
        config=args)


    WORLD.experiment = experiment
    WORLD.initEstimators(experiment)
    experiment.run()

    totalLog.export(folder / f"{outputFileName}.csv")
    averageLog.export(folder / f"{outputFileName}_average.csv")

    plots.createLogPlot(
        totalLog.records,
        averageLog.records,
        folder / f"{outputFileName}.png",
        f"World: {outputFileName}",
        (args.simulations, args.iterations)
    )
    return averageLog


def createLocalWorld(args):
    # load config from yaml
    localDict = args.locals

    localDict['chargerCapacity'] = findChargerCapacity(localDict)
    localDict['totalAvailableChargingEnergy'] = min(
        localDict['chargerCapacity'] * len(localDict['chargers']) * localDict['chargingRate'],
        localDict['totalAvailableChargingEnergy'])

    ENVIRONMENT.loadConfig(localDict)

    return localDict


def findChargerCapacity(localDict):
    margin = 1.3
    chargers = len(localDict['chargers'])
    drones = localDict['drones']

    c1 = localDict['chargingRate']
    c2 = localDict['droneMovingEnergyConsumption']

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
    outputFileName = os.path.splitext(os.path.basename(args.name))[0]

    folder = Path() / 'results' / args.output
    folder.mkdir(parents=True, exist_ok=True)
    # after creating the folder, we need to change it
    args.setConfig('output',folder)

    animations = Path() / folder / 'animations'
    animations.mkdir(parents=True, exist_ok=True)
  
    chargerLogs = Path() / folder / 'charger_logs'
    chargerLogs.mkdir(parents=True, exist_ok=True)

    return folder, outputFileName



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
    if args.configs:
        simulationConfigObject = Configuration(args.configs)
        
    else:
        simulationConfigObject = Configuration()

    # override
    for key, value in args.__dict__.items():
        if args.__dict__[key] is not None:
            # determine if the argument should be stored in locals or ml-deeco arguments
            if key in simulationConfigObject.__dict__:
                simulationConfigObject.setConfig(key,value)
            else:
                simulationConfigObject.locals[key] = value

    return simulationConfigObject

def main():
    parser = argparse.ArgumentParser(
        # TODO add proper description
        description='')
    # parser.add_argument(
    #     'input', type=str, 
    #     help='YAML world address to be run.',
    # )
    # ML-DEECO Config file and overrides, if NONE, it loads from the --config <file.yaml> file
    parser.add_argument(
        '-c', '--configs', type=str, nargs='+',
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
        required=False, default=None
    )
    parser.add_argument(
        '-n', '--name', type=str, 
        help='the name of the experiment', 
        required=False, default=None
    )
    parser.add_argument(
        '-a', '--animation', action='store_true', 
        help='toggles saving the final results as a GIF animation.',
        required=False,  default=None
    )
    parser.add_argument(
        '-p', '--plot', action='store_true', 
        help='toggles saving the plot results.',
        required=False,  default=None
    )
    args = parser.parse_args()
    # from this point ARGS is a CONFIGURATION instance, not an ARGPARSE namespace
    args = validateArguments(args)
    setVerboseLevel(args.verbose)
    run(args)


if __name__ == "__main__":
    main()
