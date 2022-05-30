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
from ml_deeco.simulation import run_experiment, SIMULATION_GLOBALS
from ml_deeco.utils import setVerboseLevel, verbosePrint, Log


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

    run_experiment(args.iterations, args.simulations, ENVIRONMENT.maxSteps, prepareSimulation,
                   iterationCallback=iterationCallback, simulationCallback=simulationCallback, stepCallback=stepCallback)

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

    # yamlObject['drones']=drones
    if args.birds > -1:
        yamlObject['birds'] = args.birds
    # yamlObject['maxSteps']=int(args.timesteps)
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
        "testSplit": args.test_split,
    }
    WORLD.waitingTimeEstimator = NeuralNetworkEstimator(
        args.hidden_layers,
        fit_params={
            "batch_size": 256,
        },
        outputFolder=f"{folder}\\waiting_time",
        name="Waiting Time",
        **commonArgs,
    )
    WORLD.waitingTimeBaseline = args.baseline
    # if args.load != "":
    #     waitingTimeEstimator.loadModel(args.load)
    WORLD.batteryEstimator = NeuralNetworkEstimator(
        args.hidden_layers,
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


def main():
    parser = argparse.ArgumentParser(description='Process YAML source file (S) and run the simulation (N) Times with Model M.')
    parser.add_argument('input', type=str, help='YAML address to be run.')

    parser.add_argument('-i', '--iterations', type=int, help='The number of iterations (trainings) to be performed.', required=False, default="1")
    parser.add_argument('-s', '--simulations', type=int, help='The number of simulation runs per iteration.', required=False, default="1")

    parser.add_argument('-o', '--output', type=str, help='the output folder', required=False, default="output")
    parser.add_argument('-v', '--verbose', type=int, help='the verboseness between 0 and 4.', required=False, default="0")
    parser.add_argument('-a', '--animation', action='store_true', default=False,
                        help='toggles saving the final results as a GIF animation.')
    parser.add_argument('-c', '--chart', action='store_true', default=False, help='toggles saving and showing the charts.')

    parser.add_argument('-d', '--accumulate_data', action='store', default=False, const=True, nargs="?", type=int,
                        help='False = use only training data from last iteration.\nTrue = accumulate training data from all previous iterations.\n<number> = accumulate training data from last <number> iterations.')
    parser.add_argument('--test_split', type=float, help='Number of records used for evaluation.', required=False, default=0.2)
    parser.add_argument('--hidden_layers', nargs="+", type=int, default=[256, 256], help='Number of neurons in hidden layers.')
    parser.add_argument('-b', '--baseline', type=int, help='Constant for waiting time baseline.', required=False, default=0)

    parser.add_argument('--seed', type=int, help='Random seed.', required=False, default=42)
    parser.add_argument('--threads', type=int, help='Number of CPU threads TF can use.', required=False, default=4)
    # parser.add_argument('-l', '--load', type=str, help='Load the model from a file.', required=False, default="")  # TODO: split for waiting time and battery

    parser.add_argument('-x', '--birds', type=int, help='number of birds, if no set, it loads from yaml file.', required=False, default=-1)
    args = parser.parse_args()

    setVerboseLevel(args.verbose)

    if args.iterations <= 0:
        raise argparse.ArgumentTypeError(f"Number of iterations must be positive: {args.iterations}")
    if args.simulations <= 0:
        raise argparse.ArgumentTypeError(f"Number of simulations must be positive: {args.simulations}")

    run(args)


if __name__ == "__main__":
    main()
