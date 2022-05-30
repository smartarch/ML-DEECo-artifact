import argparse
import os
import random
from pathlib import Path
from typing import List
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU in TF. The models are small, so it is actually faster to use the CPU.
import tensorflow as tf

from ml_deeco.estimators import NeuralNetworkEstimator
from ml_deeco.simulation import Component, run_experiment, SIMULATION_GLOBALS
from ml_deeco.utils import setVerboseLevel, verbosePrint, Log, setVerbosePrintFile, AverageLog

from configuration import CONFIGURATION, createFactory, setArrivalTime
from components import Shift, Worker
from helpers import DayOfWeek
from plots import plotStandbysAndLateness, plotLateWorkersNN


arrivedAtWorkplaceTimeAvgTimes = []
workerLogs = {}
cancelledWorkersLog: Log
shiftsLog = AverageLog(["iteration", "simulation", "shift", "arrived", "standbys", "avg_work_start_time", "lateness"])


def computeLateness(workers):
    """Mean of square of delay of workers which arrive late."""
    arrivalTimes = np.array([w.arrivedAtWorkplaceTime for w in workers])
    return float(np.mean(np.max(arrivalTimes - CONFIGURATION.shiftStart, 0) ** 2))


def run(args):

    # Fix random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Set number of threads
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # initialize output path
    CONFIGURATION.outputFolder = Path(args.output_folder)
    os.makedirs(CONFIGURATION.outputFolder, exist_ok=True)
    outputFile = open(CONFIGURATION.outputFolder / "output.txt", "w")

    # initialize configuration
    CONFIGURATION.cancellationBaseline = args.baseline
    CONFIGURATION.lateWorkersNN = NeuralNetworkEstimator(
        [32, 64, 32], fit_params={"batch_size": 4096},
        name="late_workers", outputFolder=CONFIGURATION.outputFolder / "late_workers"
    )

    # initialize verbose printing
    setVerboseLevel(args.verbose)
    setVerbosePrintFile(outputFile)

    from ensembles import getEnsembles, CancelLateWorkers

    def prepareSimulation(_i, simulation):
        """Prepares the components and ensembles for the simul """
        global workerLogs, cancelledWorkersLog
        workerLogs = {}
        cancelledWorkersLog = Log(["time", "worker", "bus_arrival", "shift"])

        CONFIGURATION.dayOfWeek = DayOfWeek(simulation)

        components: List[Component] = []
        shifts = []

        factory, workplaces, busStop = createFactory()
        components.append(factory)

        for workplace in workplaces:
            workers = [Worker(workplace, busStop) for _ in range(CONFIGURATION.workersPerShift)]
            for worker in workers:
                setArrivalTime(worker, simulation)
            standbys = [Worker(workplace, busStop) for _ in range(CONFIGURATION.standbysPerShift)]
            shift = Shift(workplace, workers, standbys)
            components += [workplace, shift, *workers, *standbys]
            shifts.append(shift)

            if args.log_workers:
                for worker in workers + standbys:
                    workerLogs[worker] = Log(["x", "y", "state", "isAtFactory", "hasHeadGear"])

        return components, getEnsembles(shifts)

    def stepCallback(components, ensembles, step):
        for cancelled in filter(lambda c: isinstance(c, CancelLateWorkers), ensembles):
            for worker in cancelled.lateWorkers:
                cancelledWorkersLog.register([step, worker, worker.busArrivalTime, cancelled.shift])

        if args.log_workers:
            for worker in filter(lambda c: isinstance(c, Worker), components):
                workerLogs[worker].register([int(worker.location.x), int(worker.location.x), worker.state, worker.isAtFactory, worker.hasHeadGear])

    def simulationCallback(components, _ens, i, s):
        workersLog = Log(["worker", "shift", "state", "isAtFactory", "hasHeadGear", "busArrivalTime", "arrivedAtFactoryTime",
                          "arrivedAtWorkplaceTime"])

        shifts = filter(lambda c: isinstance(c, Shift), components)
        for shift in shifts:
            arrivedWorkers = list(filter(lambda w: w.arrivedAtWorkplaceTime is not None, shift.workers))
            avgArriveTime = sum(map(lambda w: w.arrivedAtWorkplaceTime, arrivedWorkers)) / len(arrivedWorkers)
            arrivedAtWorkplaceTimeAvgTimes.append(avgArriveTime)
            standbysCount = len(shift.calledStandbys)
            verbosePrint(f"{shift}: arrived {len(arrivedWorkers)} workers ({standbysCount} standbys), avg. time = {avgArriveTime:.2f}, lateness = {computeLateness(arrivedWorkers):.0f}", 2)
            shiftsLog.register([i + 1, s + 1, str(shift), len(arrivedWorkers), standbysCount, avgArriveTime, computeLateness(arrivedWorkers)])

            for worker in shift.assigned | shift.standbys:
                workersLog.register([worker, shift, worker.state, worker.isAtFactory, worker.hasHeadGear, worker.busArrivalTime, worker.arrivedAtFactoryTime, worker.arrivedAtWorkplaceTime])
        shiftsLog.registerAvg()

        workersAtFactory = list(filter(lambda c: isinstance(c, Worker) and c.arrivedAtFactoryTime is not None, components))
        if workersAtFactory:
            avgFactoryArrivalTime = sum(map(lambda w: w.arrivedAtFactoryTime, workersAtFactory)) / len(workersAtFactory)
            verbosePrint(f"Average arrival at factory = {avgFactoryArrivalTime:.2f}", 2)

        os.makedirs(CONFIGURATION.outputFolder / f"cancelled_workers/{i + 1}/", exist_ok=True)
        cancelledWorkersLog.export(CONFIGURATION.outputFolder / f"cancelled_workers/{i + 1}/{s + 1}.csv")

        os.makedirs(CONFIGURATION.outputFolder / f"workers/{i + 1}/", exist_ok=True)
        workersLog.export(CONFIGURATION.outputFolder / f"workers/{i + 1}/{s + 1}.csv")

        if args.log_workers:
            os.makedirs(CONFIGURATION.outputFolder / f"all_workers/{i+1}/{s+1}", exist_ok=True)
            for worker in filter(lambda c: isinstance(c, Worker), components):
                workerLogs[worker].export(CONFIGURATION.outputFolder / f"all_workers/{i+1}/{s+1}/{worker}.csv")

    def iterationCallback(i):
        global arrivedAtWorkplaceTimeAvgTimes
        avgTimesAverage = sum(arrivedAtWorkplaceTimeAvgTimes) / len(arrivedAtWorkplaceTimeAvgTimes)
        verbosePrint(f"Average arrival time in the iteration: {avgTimesAverage:.2f}", 1)
        arrivedAtWorkplaceTimeAvgTimes = []

        # save the NN
        CONFIGURATION.lateWorkersNN.saveModel(str(i + 1))
        plotLateWorkersNN(CONFIGURATION.lateWorkersNN, CONFIGURATION.outputFolder / f"nn_{i + 1}.png", f"Iteration {i + 1}", show=args.show_plots)

    run_experiment(args.iterations, 7, CONFIGURATION.steps, prepareSimulation,
                   stepCallback=stepCallback, simulationCallback=simulationCallback, iterationCallback=iterationCallback)
    outputFile.close()
    shiftsLog.export(CONFIGURATION.outputFolder / "shifts.csv")
    shiftsLog.exportAvg(CONFIGURATION.outputFolder / "shifts_avg.csv")
    plotStandbysAndLateness(shiftsLog, args.iterations, 7, CONFIGURATION.outputFolder / "shifts.png", show=args.show_plots)


def main():
    parser = argparse.ArgumentParser(description='Smart factory simulation')
    parser.add_argument('-v', '--verbose', type=int, help='the verboseness between 0 and 4.', required=False, default="0")
    parser.add_argument('-s', '--seed', type=int, help='Random seed.', required=False, default=42)
    parser.add_argument('--threads', type=int, help='Number of CPU threads TF can use.', required=False, default=4)
    parser.add_argument('-o', '--output_folder', type=str, help='Output folder for the logs.', required=True, default='results')
    parser.add_argument('-w', '--log_workers', action='store_true', help='Save logs of all workers.', required=False, default=False)
    parser.add_argument('-b', '--baseline', type=int, help="Cancel missing workers 'baseline' minutes before the shift starts.", required=False, default=16)
    parser.add_argument('-i', '--iterations', type=int, help="Number of iterations to run.", required=False, default=3)
    parser.add_argument('-p', '--show_plots', action='store_true', help='Show plots during the run.', required=False, default=False)
    args = parser.parse_args()

    run(args)


if __name__ == "__main__":
    main()
