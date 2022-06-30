import random
from typing import Tuple, List
import numpy as np
import numpy.random as npr

from ml_deeco.simulation import Point2D, Configuration as ml_deeco_Configuration

from components import WorkPlace, Factory, Door, Dispenser, Worker


class Configuration(ml_deeco_Configuration):
    """Holds the configuration of the experiment."""

    # reference to the experiment instance which is useful for setting the estimator inside CancelLateWorkers ensemble
    experiment = None

    shiftStart: int
    shiftEnd: int
    workersPerShift: int
    standbysPerShift: int

    cancellationBaseline: int

    # arrival configuration
    weekDayBus: int
    weekEndBus: int
    latePercentage: float
    lateWeekDayBus: int
    lateWeekEndBus: int
    standbyMean: int
    standbyStd: float

    dayOfWeek = None

    def updateLocals(self):
        self.__dict__.update(self.locals)

    def __init__(self):
        if 'CONFIGURATION' in locals():
            raise RuntimeError("Do not create a new instance of the Configuration. Use the CONFIGURATION global variable instead.")


CONFIGURATION = Configuration()


def createFactory() -> Tuple[Factory, List[WorkPlace], Point2D]:
    """Creates an example factory with work places and a bus stop (at which the workers arrive)."""

    factory = Factory()
    factory.entryDoor = Door(20, 90)
    factory.dispenser = Dispenser(30, 90)

    workplace1 = WorkPlace(factory)
    workplace1.entryDoor = Door(40, 50)
    workplace1.pathTo = [Point2D(30, 50)]

    workplace2 = WorkPlace(factory)
    workplace2.entryDoor = Door(120, 50)
    workplace2.pathTo = [Point2D(110, 90), Point2D(110, 50)]

    workplace3 = WorkPlace(factory)
    workplace3.entryDoor = Door(120, 110)
    workplace3.pathTo = [Point2D(110, 90), Point2D(110, 110)]

    busStop = Point2D(0, 90)

    return factory, [workplace1, workplace2, workplace3], busStop


def setArrivalTime(worker: Worker, dayOfWeek):
    """Sets a random arrival time for the worker (based on the configured mean and standard deviation)."""

    from helpers import DayOfWeek
    dayOfWeek = DayOfWeek(dayOfWeek % 7)
    randomDelay = int(np.round(npr.exponential()))

    if dayOfWeek in (DayOfWeek.SATURDAY, DayOfWeek.SUNDAY):
        if random.random() < CONFIGURATION.latePercentage:
            worker.busArrivalTime = CONFIGURATION.lateWeekEndBus + randomDelay
        else:
            worker.busArrivalTime = CONFIGURATION.weekEndBus + randomDelay
    else:
        if random.random() < CONFIGURATION.latePercentage:
            worker.busArrivalTime = CONFIGURATION.lateWeekDayBus + randomDelay
        else:
            worker.busArrivalTime = CONFIGURATION.weekDayBus + randomDelay


def setStandbyArrivedAtWorkplaceTime(standby: Worker):
    """We will not simulate the standby, just assume they will start working about an hour after they are called."""
    from helpers import now
    randomDelay = int(random.gauss(CONFIGURATION.standbyMean, CONFIGURATION.standbyStd))
    standby.arrivedAtWorkplaceTime = now() + randomDelay
