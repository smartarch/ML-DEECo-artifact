from typing import List, TYPE_CHECKING, Optional

from ml_deeco.simulation import SIMULATION_GLOBALS
from ml_deeco.utils import Log
from ml_deeco.estimators import NoEstimator

if TYPE_CHECKING:
    from ml_deeco.estimators import Estimator

MAX_RANDOM_POINTS = 100


class Environment:
    """
    Constants regarding the simulation environment.
    """

    maxSteps = 500
    mapWidth = 100
    mapHeight = 100

    droneRadius = 5
    birdSpeed = 1
    droneSpeed = 1

    totalAvailableChargingEnergy = 0.1
    droneBatteryRandomize = 0
    droneMovingEnergyConsumption = 0.01
    droneProtectingEnergyConsumption = 0.005
    chargingRate = 0.04
    currentChargingRate = 0.04
    chargerCapacity = 1
    droneStartPositionVariance = 0
    droneCount = 8
    birdCount = 20
    chargerCount = 2
    chargerPositions = []
    fieldCount = 2
    fieldPositions = []

    def __init__(self):
        if 'ENVIRONMENT' in locals():
            raise RuntimeError("Do not create a new instance of the Environment. Use the ENVIRONMENT global variable instead.")

    def loadConfig(self, config: dict):
        for conf, confValue in config.items():
            if conf in Environment.__dict__:
                self.__dict__[conf] = confValue
        self.droneCount = config['drones']
        self.birdCount = config['birds']
        self.chargerCount = len(config['chargers'])
        self.chargerPositions = config['chargers']
        self.fieldCount = len(config['fields'])
        self.fieldPositions = config['fields']
        self.currentChargingRate = self.chargingRate


ENVIRONMENT = Environment()


class World:
    """
    The simulated world.
    """

    def __init__(self):
        if 'WORLD' in locals():
            raise RuntimeError("Do not create a new instance of the World. Use the WORLD global variable instead.")
        waitingTimeEstimator: 'Estimator'
        waitingTimeBaseline = 0
        batteryEstimator: 'Estimator'

    @staticmethod
    def initEstimators():
        # import all Components and Ensembles with Estimates

        # noinspection PyUnresolvedReferences
        from components.drone import Drone
        # noinspection PyUnresolvedReferences
        from components.charger import Charger
        # noinspection PyUnresolvedReferences
        from ensembles.drone_charging import getEnsembles as droneChargingEnsembles

        SIMULATION_GLOBALS.initEstimators()

    # noinspection PyAttributeOutsideInit
    def reset(self):
        """
        Call this before the world is used.
        """
        from ml_deeco.simulation import Point2D
        from components.bird import Bird
        from components.field import Field
        from components.drone import Drone
        from components.charger import Charger
        import random

        def randomStartingPoint():
            variant = ENVIRONMENT.droneStartPositionVariance
            centerX = ENVIRONMENT.mapWidth / 2
            centerY = ENVIRONMENT.mapHeight / 2
            randomX = centerX + (random.choice([-1, 1]) * variant * random.random() * centerX)
            randomY = centerY + (random.choice([-1, 1]) * variant * random.random() * centerY)
            return Point2D(int(randomX), int(randomY))

        self.drones: List[Drone] = [Drone(randomStartingPoint()) for _ in range(ENVIRONMENT.droneCount)]
        self.birds: List[Bird] = [Bird(Point2D.random(0, 0, ENVIRONMENT.mapWidth, ENVIRONMENT.mapHeight)) for _ in range(ENVIRONMENT.birdCount)]
        self.chargers: List[Charger] = [Charger(Point2D(position)) for position in ENVIRONMENT.chargerPositions]
        self.fields: List[Field] = [Field(points) for points in ENVIRONMENT.fieldPositions]

        self.totalPlaces = sum([len(f.places) for f in self.fields])
        self.sortedFields = sorted(self.fields, key=lambda field: -len(field.places))

        self.emptyPoints = []
        for i in range(MAX_RANDOM_POINTS):
            p = Point2D.random(0, 0, ENVIRONMENT.mapWidth, ENVIRONMENT.mapHeight)
            if self.isPointField(p):
                i = i - 1
            else:
                self.emptyPoints.append(p)

        self.createLogs()

        components = []
        components.extend(WORLD.birds)

        if ENVIRONMENT.droneCount > 0:
            components.extend(WORLD.drones)
            components.extend(WORLD.chargers)
            from ensembles.field_protection import getEnsembles as fieldProtectionEnsembles
            from ensembles.drone_charging import getEnsembles as droneChargingEnsembles
            potentialEnsembles = fieldProtectionEnsembles() + droneChargingEnsembles()
        else:
            potentialEnsembles = []

        return components, potentialEnsembles

    # noinspection PyAttributeOutsideInit
    def createLogs(self):

        self.chargerLog = Log([
            "current_time_step",
            "drone_id",
            "battery",
            "estimated_waiting",
            "energy_to_fly_to_charger",
            "time_to_done_charging",
            "charger",
            "potential_drones_length",
            "waiting_drones_length",
            "accepted_queues_length",
            "charging_drones_length",
        ])

        self.chargerLogs = []
        for _ in self.chargers:
            self.chargerLogs.append(Log([
                "Charging Drones",
                "Accepted Drones",
                "Waiting Drones",
                "Potential Drones",
            ]))

    def isProtectedByDrone(self, point):
        for drone in self.drones:
            if drone.isProtecting(point):
                return True
        return False

    def isPointField(self, point):
        for field in self.fields:
            if field.isPointOnField(point):
                return True
        return False

    def findDrones(self, droneStates):
        return [drone for drone in self.drones if drone.state in droneStates]

    def exceptDrones(self, droneStates):
        return [drone for drone in self.drones if drone.state not in droneStates]

    def findBirds(self, birdStates):
        return [bird for bird in self.birds if bird.state in birdStates]

    def exceptBirds(self, birdStates):
        return [bird for bird in self.birds if bird.state not in birdStates]


WORLD = World()
