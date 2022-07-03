import argparse
import os
import random
from pathlib import Path
from typing import List
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU in TF. The models are small, so it is actually faster to use the CPU.
import tensorflow as tf

from ml_deeco.simulation import Component, Experiment
from ml_deeco.utils import setVerboseLevel, verbosePrint, Log, setVerbosePrintFile, AverageLog
from ml_deeco.estimators import Estimator

from configuration import CONFIGURATION, createFactory, setArrivalTime, Configuration
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


class FactoryExperiment(Experiment):

    lateWorkers: Estimator

    def __init__(self, args):
        super().__init__(CONFIGURATION)
        self.args = args

        # Fix random seeds
        random.seed(args.seed)
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)

        # Set number of threads
        tf.config.threading.set_inter_op_parallelism_threads(args.threads)
        tf.config.threading.set_intra_op_parallelism_threads(args.threads)

        # initialize output path
        os.makedirs(CONFIGURATION.output, exist_ok=True)
        self.outputFile = open(CONFIGURATION.output / "output.txt", "w")

        # initialize verbose printing
        setVerboseLevel(args.verbose)
        setVerbosePrintFile(self.outputFile)

    def prepareSimulation(self, _i, simulation):
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

            if self.args.log_workers:
                for worker in workers + standbys:
                    workerLogs[worker] = Log(["x", "y", "state", "isAtFactory", "hasHeadGear"])

        from ensembles import getEnsembles
        return components, getEnsembles(shifts)

    def stepCallback(self, components, ensembles, step):
        from ensembles import CancelLateWorkers
        for cancelled in filter(lambda c: isinstance(c, CancelLateWorkers), ensembles):
            for worker in cancelled.lateWorkers:
                cancelledWorkersLog.register([step, worker, worker.busArrivalTime, cancelled.shift])

        if self.args.log_workers:
            for worker in filter(lambda c: isinstance(c, Worker), components):
                workerLogs[worker].register([int(worker.location.x), int(worker.location.x), worker.state, worker.isAtFactory, worker.hasHeadGear])

    def simulationCallback(self, components, _ens, i, s):
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

        os.makedirs(CONFIGURATION.output / f"cancelled_workers/{i + 1}/", exist_ok=True)
        cancelledWorkersLog.export(CONFIGURATION.output / f"cancelled_workers/{i + 1}/{s + 1}.csv")

        os.makedirs(CONFIGURATION.output / f"workers/{i + 1}/", exist_ok=True)
        workersLog.export(CONFIGURATION.output / f"workers/{i + 1}/{s + 1}.csv")

        if self.args.log_workers:
            os.makedirs(CONFIGURATION.output / f"all_workers/{i+1}/{s+1}", exist_ok=True)
            for worker in filter(lambda c: isinstance(c, Worker), components):
                workerLogs[worker].export(CONFIGURATION.output / f"all_workers/{i+1}/{s+1}/{worker}.csv")

    def iterationCallback(self, i):
        global arrivedAtWorkplaceTimeAvgTimes
        avgTimesAverage = sum(arrivedAtWorkplaceTimeAvgTimes) / len(arrivedAtWorkplaceTimeAvgTimes)
        verbosePrint(f"Average arrival time in the iteration: {avgTimesAverage:.2f}", 1)
        arrivedAtWorkplaceTimeAvgTimes = []

        # save the NN
        self.lateWorkers.saveModel(str(i + 1))
        plotLateWorkersNN(self.lateWorkers, CONFIGURATION.output / f"nn_{i + 1}.png", f"Iteration {i + 1}", show=self.args.show_plots)

    def exportResults(self):
        self.outputFile.close()
        shiftsLog.export(CONFIGURATION.output / "shifts.csv")
        shiftsLog.exportAvg(CONFIGURATION.output / "shifts_avg.csv")
        plotStandbysAndLateness(shiftsLog, self.config.iterations, self.config.simulations, CONFIGURATION.output / "shifts.png", show=self.args.show_plots)


def main():
    parser = argparse.ArgumentParser(description='Smart factory simulation')
    parser.add_argument(
        '-c', '--configs', type=str, nargs='+',
        help='The configuration files',
        required=False, default=["experiments/factory.yaml", "experiments/estimators.yaml"]
    )
    parser.add_argument('-v', '--verbose', type=int, help='the verboseness between 0 and 4.', required=False, default="0")
    parser.add_argument('-s', '--seed', type=int, help='Random seed.', required=False, default=42)
    parser.add_argument('--threads', type=int, help='Number of CPU threads TF can use.', required=False, default=4)
    parser.add_argument('-o', '--output', type=str, help='Output folder for the logs.')
    parser.add_argument('-w', '--log_workers', action='store_true', help='Save logs of all workers.', required=False, default=False)
    parser.add_argument('-b', '--baseline', type=int, help="Cancel missing workers 'baseline' minutes before the shift starts.", required=False)
    parser.add_argument('-i', '--iterations', type=int, help="Number of iterations to run.", required=False)
    parser.add_argument('-p', '--show_plots', action='store_true', help='Show plots during the run.', required=False, default=False)
    args = parser.parse_args()

    for file in args.configs:
        CONFIGURATION.loadConfigurationFromFile(file)
    CONFIGURATION.updateLocals()

    if args.baseline:
        CONFIGURATION.cancellationBaseline = args.baseline
    if args.iterations:
        CONFIGURATION.iterations = args.iterations
    if args.output:
        CONFIGURATION.output = args.output
    CONFIGURATION.output = Path(CONFIGURATION.output)

    experiment = FactoryExperiment(args)
    CONFIGURATION.experiment = experiment
    import ensembles  # import to initialize Estimates
    experiment.run()
    experiment.exportResults()


if __name__ == "__main__":
    main()
