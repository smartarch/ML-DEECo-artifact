import random
from typing import Tuple, List
import numpy as np
import numpy.random as npr

from ml_deeco.simulation import Point2D, SIMULATION_GLOBALS

from helpers import DayOfWeek
from components import WorkPlace, Factory, Door, Dispenser, Worker


class Configuration:

    steps = 50
    shiftStart = 30
    shiftEnd = 50
    workersPerShift = 100
    standbysPerShift = 50
    dayOfWeek = None

    outputFolder = None
    cancellationBaseline = 16
    lateWorkersNN = None

    def __init__(self):
        if 'CONFIGURATION' in locals():
            raise RuntimeError("Do not create a new instance of the Configuration. Use the CONFIGURATION global variable instead.")


CONFIGURATION = Configuration()


def createFactory() -> Tuple[Factory, List[WorkPlace], Point2D]:
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


# workers arrive by a bus
weekDayBus = 6
weekEndBus = 0
# several workers miss the first bus and arrive by the late bus
latePercentage = 0.1
lateWeekDayBus = 12
lateWeekEndBus = 15
# standby needs about 30 minutes to arrive
standbyMean, standbyStd = 30, 2


def setArrivalTime(worker: Worker, dayOfWeek):
    dayOfWeek = DayOfWeek(dayOfWeek % 7)
    randomDelay = int(np.round(npr.exponential()))

    if dayOfWeek in (DayOfWeek.SATURDAY, DayOfWeek.SUNDAY):
        if random.random() < latePercentage:
            worker.busArrivalTime = lateWeekEndBus + randomDelay
        else:
            worker.busArrivalTime = weekEndBus + randomDelay
    else:
        if random.random() < latePercentage:
            worker.busArrivalTime = lateWeekDayBus + randomDelay
        else:
            worker.busArrivalTime = weekDayBus + randomDelay


# we will not simulate the standby, just assume they will start working about an hour after they are called
def setStandbyArrivedAtWorkplaceTime(standby: Worker):
    standby.arrivedAtWorkplaceTime = SIMULATION_GLOBALS.currentTimeStep + int(random.gauss(standbyMean, standbyStd))
