from typing import List

from configuration import setStandbyArrivedAtWorkplaceTime, CONFIGURATION
from ml_deeco.estimators import CategoricalFeature, BinaryFeature, NumericFeature
from ml_deeco.simulation import Ensemble, someOf
from ml_deeco.utils import verbosePrint

from components import Shift, Worker, WorkerState
from helpers import allow, now, DayOfWeek


class ShiftTeam(Ensemble):
    """All working workers in one shift (those not cancelled, incl. called standbys)."""

    shift: Shift

    def __init__(self, shift: Shift):
        super().__init__()
        self.shift = shift

    def priority(self):
        return 5

    workers = someOf(Worker, selectedAllAtOnce=True)

    @workers.select
    def workers(self, worker, otherEnsembles):
        return worker in (self.shift.assigned - self.shift.cancelled) or \
            worker in self.shift.calledStandbys

    def actuate(self):
        self.shift.workers = set(self.workers)


class AccessToFactory(Ensemble):

    shift: Shift

    def __init__(self, shift: Shift):
        super().__init__()
        self.shift = shift
        self.factory = shift.workPlace.factory

    def priority(self):
        return 4

    def situation(self):
        startTime = self.shift.startTime
        endTime = self.shift.endTime
        return startTime - 30 <= now() <= endTime + 30

    def actuate(self):
        allow(self.shift.workers, "enter", self.factory.entryDoor)


class AccessToDispenser(Ensemble):

    shift: Shift

    def __init__(self, shift: Shift):
        super().__init__()
        self.shift = shift
        self.dispenser = shift.workPlace.factory.dispenser

    def priority(self):
        return 4

    def situation(self):
        startTime = self.shift.startTime
        endTime = self.shift.endTime
        return startTime - 20 <= now() <= endTime

    def actuate(self):
        allow(self.shift.workers, "use", self.dispenser)


class AccessToWorkPlace(Ensemble):

    shift: Shift

    def __init__(self, shift: Shift):
        super().__init__()
        self.shift = shift
        self.workPlace = shift.workPlace

    def priority(self):
        return 3

    def situation(self):
        startTime = self.shift.startTime
        endTime = self.shift.endTime
        return startTime - 30 <= now() <= endTime + 30

    workers = someOf(Worker, selectedAllAtOnce=True)  # subset of self.shift.workers

    @workers.select
    def workers(self, worker, otherEnsembles):
        return worker in self.shift.workers and worker.hasHeadGear

    def actuate(self):
        allow(self.workers, "enter", self.workPlace.entryDoor)


class CancelLateWorkers(Ensemble):

    shift: Shift

    def __init__(self, shift: Shift):
        super().__init__()
        self.shift = shift

    def priority(self):
        return 2

    def situation(self):
        startTime = self.shift.startTime
        endTime = self.shift.endTime
        return startTime - 30 <= now() <= endTime

    # region late workers

    def isLateBaseline(self, _worker, timeToShift):
        """As a baseline, we assume that the worker will not arrive if they are not present 10 minutes before the shift starts."""
        return timeToShift <= CONFIGURATION.cancellationBaseline

    # lateWorkers = someOf(Worker).withTimeEstimate(collectOnlyIfMaterialized=False).using(ConstantEstimator(10))

    lateWorkers = someOf(Worker, selectedAllAtOnce=True)\
        .withValueEstimate(collectOnlyIfMaterialized=False)\
        .inTimeStepsRange(1, 20, trainingPercentage=0.2)\
        .using(CONFIGURATION.experiment.lateWorkers)\
        .withBaseline(isLateBaseline)

    @lateWorkers.select
    def lateWorkers(self, worker, otherEnsembles):
        if self.potentiallyLate(worker):
            # estimatedArrival = now() + self.lateWorkers.estimate(worker)
            # return estimatedArrival > self.shift.startTime
            timeToShift = self.shift.startTime - now()
            return self.lateWorkers.estimate(worker, timeToShift)
        return False

    # @lateWorkers.estimate.conditionsValid
    @lateWorkers.estimate.targetsValid
    def belongsToShift(self, worker):
        return worker in self.shift.assigned - self.shift.cancelled

    @lateWorkers.estimate.inputsValid
    def potentiallyLate(self, worker):
        return not worker.isAtFactory and self.belongsToShift(worker)

    # @lateWorkers.estimate.input(NumericFeature(0, CONFIGURATION.workersPerShift))
    # def alreadyPresentWorkers(self, worker):
    #     return len(list(filter(lambda w: w.isAtFactory, self.shift.workers)))

    @lateWorkers.estimate.input(CategoricalFeature(DayOfWeek))
    def dayOfWeek(self, worker):
        return CONFIGURATION.dayOfWeek

    # @lateWorkers.estimate.condition
    # def arrived(self, worker):
    #     return worker.isAtFactory

    @lateWorkers.estimate.target(BinaryFeature())
    def isLate(self, worker):
        return not worker.isAtFactory

    # endregion

    def actuate(self):
        self.shift.cancelled.update(self.lateWorkers)
        for worker in self.lateWorkers:
            worker.state = WorkerState.CANCELLED  # this is instead of the notification


class ReplaceLateWithStandbys(Ensemble):

    lateWorkersEnsemble: CancelLateWorkers
    shift: Shift

    def __init__(self, lateWorkersEnsemble: CancelLateWorkers):
        super().__init__()
        self.lateWorkersEnsemble = lateWorkersEnsemble
        self.shift = lateWorkersEnsemble.shift

    def priority(self):
        return 1

    def situation(self):
        return self.lateWorkersEnsemble.materialized

    standbys = someOf(Worker)

    @standbys.select
    def standbys(self, worker, otherEnsembles):
        # TODO: not in other shift
        return worker in self.shift.availableStandbys

    @standbys.cardinality
    def standbys(self):
        return 0, len(self.lateWorkersEnsemble.lateWorkers)

    def actuate(self):
        verbosePrint(str(self.standbys), 5)
        self.shift.calledStandbys.update(self.standbys)
        for standby in self.standbys:
            standby.state = WorkerState.CALLED_STANDBY  # this is instead of the notification of the standby
            setStandbyArrivedAtWorkplaceTime(standby)


def getEnsembles(shifts: List[Shift]):
    ensembles: List[Ensemble] = []

    ensembles.extend((ShiftTeam(shift) for shift in shifts))
    ensembles.extend((AccessToFactory(shift) for shift in shifts))
    ensembles.extend((AccessToDispenser(shift) for shift in shifts))
    ensembles.extend((AccessToWorkPlace(shift) for shift in shifts))

    for shift in shifts:
        lateWorkersEnsemble = CancelLateWorkers(shift)
        ensembles.append(lateWorkersEnsemble)
        ensembles.append(ReplaceLateWithStandbys(lateWorkersEnsemble))

    return ensembles
