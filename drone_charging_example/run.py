""" 
This file contains a simple experiment run
"""
from typing import Optional

from yaml import load
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
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
from ml_deeco.simulation import Simulation, SIMULATION_GLOBALS
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

    yamlObject = loadConfig(args)

    folder, yamlFileName = prepareFoldersForResults(args)

    averageLog, totalLog = createLogs()
    visualizer: Optional[Visualizer] = None

    createEstimators(args, folder)
    WORLD.initEstimators()

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
        WORLD.chargerLog.export(f"{folder}/charger_logs/{yamlFileName}_{t + 1}_{i + 1}.csv")

        if args.animation:
            verbosePrint(f"Saving animation...", 3)
            visualizer.createAnimation(f"{folder}/animations/{yamlFileName}_{t + 1}_{i + 1}.gif")
            verbosePrint(f"Animation saved.", 3)

        if args.chart:
            verbosePrint(f"Saving charger plot...", 3)
            plots.createChargerPlot(
                WORLD.chargerLogs,
                f"{folder}\\charger_logs\\{yamlFileName}_{str(t + 1)}_{str(i + 1)}",
                f"World: {yamlFileName}\n Run: {i + 1} in training {t + 1}\nCharger Queues")
            verbosePrint(f"Charger plot saved.", 3)

    def iterationCallback(t):
        """Aggregate statistics from all _Simulations_ in one _Iteration_."""

        # calculate the average rate
        averageLog.register(totalLog.average(t * args.simulations, (t + 1) * args.simulations))

        for estimator in SIMULATION_GLOBALS.estimators:
            estimator.saveModel(t + 1)

    simulation = Simulation(
        args.iterations,
        args.simulations,
        prepareSimulation,
        iterationCallback=iterationCallback,
        simulationCallback=simulationCallback,
        stepCallback=stepCallback )
        
    simulation.run_experiment(ENVIRONMENT.maxSteps)

    totalLog.export(f"{folder}\\{yamlFileName}.csv")
    averageLog.export(f"{folder}\\{yamlFileName}_average.csv")

    plots.createLogPlot(
        totalLog.records,
        averageLog.records,
        f"{folder}\\{yamlFileName}.png",
        f"World: {yamlFileName}",
        (args.simulations, args.iterations)
    )
    return averageLog


def loadConfig(args):
    # load config from yaml
    yamlFile = open(args.input, 'r')
    yamlObject = load(yamlFile, Loader=Loader)

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
    folder = f"results\\{args.output}"

    if not os.path.exists(f"{folder}\\animations"):
        os.makedirs(f"{folder}\\animations")
    if not os.path.exists(f"{folder}\\charger_logs"):
        os.makedirs(f"{folder}\\charger_logs")
    return folder, yamlFileName


def createEstimators(args, folder):
    # create the estimators
    commonArgs = {
        "accumulateData": args.accumulate_data,
        "saveCharts": args.chart,
        "testSplit": 0.2,
    }
    WORLD.waitingTimeEstimator = NeuralNetworkEstimator(
        [255, 255], # hidden layers
        fit_params={
            "batch_size": 256,
        },
        outputFolder=f"{folder}\\waiting_time",
        name="Waiting Time",
        **commonArgs,
    )
    WORLD.waitingTimeBaseline = 0
    # if args.load != "":
    #     waitingTimeEstimator.loadModel(args.load)
    WORLD.batteryEstimator = NeuralNetworkEstimator(
        [255, 255], # hidden layers
        fit_params={
            "batch_size": 256,
        },
        outputFolder=f"{folder}\\battery",
        name="Battery",
        **commonArgs,
    )


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

class TestClass:
    def __init__ (self, a, b):
        self.A = a
        self.B = b
    


def validateArguments(args):

    if args.config:
        configPath = Path(args.config)
        if not configPath.exists():
            raise IOError(f"such file {args.config} does not exist")
        simulationConfigFile = open(configPath, 'r')
        simulationConfigObject = load(simulationConfigFile, Loader=Loader)
        
    else:
        simulationConfigObject = {}

    # override
    for argument in args.__dict__ :
        if args.__dict__[argument] is not None:
            simulationConfigObject[argument] = args.__dict__[argument]
    
    # set default values
    for argument , default in zip(
        ['output', 'verbose', 'chart', 'animation', 'seeds', 'thread'],
        ['output', 0        , False  , False      , 42     , 4]
    ):
        if argument not in simulationConfigObject:
            simulationConfigObject[argument] = default

    def validateArgument(argument, condition, message):
        if argument not in simulationConfigObject \
           or not condition(simulationConfigObject[argument]):
            raise argparse.ArgumentTypeError(message)
        return True


    notZeroCondition = lambda x: x >= 0
    notNoneCondition = lambda x: x is not None
    noneCondition = lambda x: x is None
    intTypeCondition = lambda x: isinstance(x, int)
    boolTypeCondition = lambda x: isinstance(x, bool)
    stringTypeCondition = lambda x: isinstance(x, str)
    andCondition = lambda a,b: lambda x: a(x) and b(x)
    orCondition = lambda a,b: lambda x: a(x) or b(x)

    # required arguments
    finalCheckup = [ 
        validateArgument(
            x, 
            notNoneCondition, 
            f"{x} must be provided in the configuration file or arguments")
        for x in ['iterations', 'simulations', 'output', 'input']
    ]
    # positive integers
    finalCheckup.extend([
        validateArgument(
            x, 
            orCondition(
                noneCondition, 
                andCondition(notZeroCondition,intTypeCondition)
            ), 
            f"{x} must be a positive integer number > 0.")
        for x in ['iterations', 'simulations', 'seeds','threads']
    ])
    # strings
    finalCheckup.extend([
        validateArgument(
            x, 
            stringTypeCondition,
            f"{x} must be a string")
        for x in ['input', 'output']
    ])

    # special attribute
    finalCheckup.append(
        validateArgument(
            'accumulate_data', 
            orCondition(
                boolTypeCondition, 
                andCondition(notZeroCondition,intTypeCondition)
            ),
            'accumulate_data must be True, False or positive integer'
            )
    )

    if not all(finalCheckup):
        raise argparse.ArgumentTypeError('an error ocurred.')

    return simulationConfigObject

def main():
    parser = argparse.ArgumentParser(
        # TODO add proper description
        description='')
    parser.add_argument(
        'input', type=str, 
        help='YAML world address to be run.',
    )
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
    parser.add_argument(
        '-o', '--output', type=str, 
        help='the output folder', 
        required=False, default=None
    )
    parser.add_argument(
        '-v', '--verbose', type=int, 
        help='the verboseness between 0 and 4.', 
        required=False, default=None
    )
    parser.add_argument(
        '-a', '--animation', action='store_true', 
        help='toggles saving the final results as a GIF animation.',
        required=False,  default=None
    )
    parser.add_argument(
       '--seed', type=int, 
       help='Random seed.', 
       required=False, default=None)

    parser.add_argument(
        '--threads', type=int, 
        help='Number of CPU threads TF can use.', 
        required=False, default=None)

    args = parser.parse_args()
    config = validateArguments(args)

    # TODO optional, to use only one dictionary we combine both into args
    for argument in config:
        args.__dict__[argument] = config[argument]

    setVerboseLevel(args.verbose)
    run(args)


if __name__ == "__main__":
    main()
