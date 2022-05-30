import enum
from collections import defaultdict
from typing import List, Set, Optional

from ml_deeco.utils import verbosePrint
from ml_deeco.simulation import StationaryComponent2D, MovingComponent2D, Component, Point2D, SIMULATION_GLOBALS


class SecurityComponent(Component):
    """Base class for components with security rules (Door, Dispenser)."""

    def __init__(self):
        super().__init__()
        self.allowed = defaultdict(set)

    def allow(self, subject, action):
        self.allowed[action].add(subject)

    def allows(self, subject, action):
        actionAllowed = subject in self.allowed[action]
        verbosePrint(f"{self}, {SIMULATION_GLOBALS.currentTimeStep + 1}: {'allowing' if actionAllowed else 'denying'} '{subject}' action '{action}'", 5)
        return actionAllowed


class Door(StationaryComponent2D, SecurityComponent):

    def __init__(self, x, y):
        super().__init__(Point2D(x, y))


class Dispenser(StationaryComponent2D, SecurityComponent):

    def __init__(self, x, y):
        super().__init__(Point2D(x, y))


class Factory(Component):

    entryDoor: Door
    dispenser: Dispenser


class WorkPlace(Component):

    factory: Factory
    entryDoor: Door
    pathTo: List[Point2D]  # waypoints from the dispenser to the workplace door

    def __init__(self, factory: Factory):
        super().__init__()
        self.factory = factory


class Shift(Component):

    def __init__(self, workPlace: WorkPlace, assigned, standbys):
        super().__init__()
        self.workPlace = workPlace
        from configuration import CONFIGURATION
        self.startTime = CONFIGURATION.shiftStart
        self.endTime = CONFIGURATION.shiftEnd
        self.assigned: Set['Worker'] = set(assigned)  # originally assigned for the shift
        self.standbys: Set['Worker'] = set(standbys)
        self.cancelled: Set['Worker'] = set()
        self.calledStandbys: Set['Worker'] = set()
        self.workers: Set['Worker'] = set()  # actually working (subset of assigned and standbys)

    @property
    def availableStandbys(self):
        return self.standbys - self.calledStandbys


class WorkerState(enum.IntEnum):
    NOT_ACTIVE_YET = 0
    WALKING_TO_FACTORY = 1
    AT_FACTORY_DOOR = 2
    WALKING_TO_DISPENSER = 3
    AT_DISPENSER = 4
    WALKING_TO_WORKPLACE = 5
    AT_WORKPLACE_DOOR = 6
    AT_WORKPLACE = 7
    CANCELLED = 8
    CALLED_STANDBY = 9


class Worker(MovingComponent2D):

    def __init__(self, workplace: WorkPlace, location):
        super().__init__(location, speed=10)
        # references
        self.workplace = workplace
        self.factory = workplace.factory
        # state variables
        self.hasHeadGear = False
        self.isAtFactory = False
        self.state = WorkerState.NOT_ACTIVE_YET
        self.pathToWorkplaceIndex = 0
        # configuration (set from the simulation)
        self.busArrivalTime: Optional[int] = None
        # logging
        self.arrivedAtFactoryTime: Optional[int] = None
        self.arrivedAtWorkplaceTime: Optional[int] = None

    def actuate(self):
        # activate the worker if he arrived by bus
        if self.state == WorkerState.NOT_ACTIVE_YET:
            if self.busArrivalTime is not None and SIMULATION_GLOBALS.currentTimeStep >= self.busArrivalTime:
                self.state = WorkerState.WALKING_TO_FACTORY
            return

        if self.state == WorkerState.CANCELLED or self.state == WorkerState.CALLED_STANDBY:
            return

        # walk to the factory
        if self.state == WorkerState.WALKING_TO_FACTORY:
            if self.move(self.factory.entryDoor.location):
                self.state = WorkerState.AT_FACTORY_DOOR

        # enter the factory
        if self.state == WorkerState.AT_FACTORY_DOOR:
            if self.factory.entryDoor.allows(self, 'enter'):
                self.state = WorkerState.WALKING_TO_DISPENSER
                self.isAtFactory = True
                self.arrivedAtFactoryTime = SIMULATION_GLOBALS.currentTimeStep
                verbosePrint(f"{self}: arrived at factory", 4)
                return

        # walk to the dispenser
        if self.state == WorkerState.WALKING_TO_DISPENSER:
            if self.move(self.factory.dispenser.location):
                self.state = WorkerState.AT_DISPENSER

        # use the dispenser
        if self.state == WorkerState.AT_DISPENSER:
            if self.factory.dispenser.allows(self, 'use'):
                self.state = WorkerState.WALKING_TO_WORKPLACE
                self.hasHeadGear = True
                return

        # work to the workplace
        if self.state == WorkerState.WALKING_TO_WORKPLACE:
            if self.pathToWorkplaceIndex >= len(self.workplace.pathTo):
                if self.move(self.workplace.entryDoor.location):
                    self.state = WorkerState.AT_WORKPLACE_DOOR
            else:
                if self.move(self.workplace.pathTo[self.pathToWorkplaceIndex]):
                    self.pathToWorkplaceIndex += 1

        # enter the workplace and start working
        if self.state == WorkerState.AT_WORKPLACE_DOOR:
            if self.workplace.entryDoor.allows(self, 'enter'):
                self.state = WorkerState.AT_WORKPLACE
                self.arrivedAtWorkplaceTime = SIMULATION_GLOBALS.currentTimeStep
                verbosePrint(f"{self}: arrived at workplace", 4)
