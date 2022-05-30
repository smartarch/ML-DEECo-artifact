import random
from typing import Optional, TYPE_CHECKING
from components.drone_state import DroneState
from ml_deeco.estimators import ValueEstimate, NumericFeature, CategoricalFeature, NoEstimator
from world import ENVIRONMENT, WORLD
from ml_deeco.simulation import MovingComponent2D, SIMULATION_GLOBALS

if TYPE_CHECKING:
    from components.charger import Charger


class Drone(MovingComponent2D):
    """
    The drones protect the fields from birds by moving to the field and scaring the flocks of birds away.

    In programming perspective, drone components have access to shared `WORLD` and they can find the position to protect.
    In a real-life scenario, it is assumed that additional sensors will perform the detection of birds, and it can be read from them.

    Attributes
    ----------
    droneRadius: int
        Protecting radius of drone.
    droneMovingEnergyConsumption: float
        The energy consumption per time-step for drone moving.
    droneProtectingEnergyConsumption: float
        The energy consumption per time-step for protecting/standing drone.
    battery: float
        The battery percentage of the drone.
    target: Point2D
        The target location of the drone.
    targetField: Field
        The target field to protect.
    targetCharger: Charger
        The selected charger.
    closestCharger: Charger
        the closest charger which is picked by pre-assignment ensemble.
    alert: float
        If computed battery is below this value, it is assumed a critical battery level.
    state: DroneState
        IDLE: a default state for drones.
        PROTECTING: when the drones are protecting the fields.
        MOVING_TO_CHARGER: when the drones are moving/queuing for a charger.
        CHARGING: when the drones are being charged.
        TERMINATED: when the drones' battery is 0, and they do not operate anymore.
    """
    def __init__(self, location):
        """
        Creates a drone object with the given location and ENVIRONMENT settings.

        Parameters
        ----------
        location : Point2D
            Starting point for the drone.
        """
        self.droneRadius = ENVIRONMENT.droneRadius
        self.droneMovingEnergyConsumption = ENVIRONMENT.droneMovingEnergyConsumption
        self.droneProtectingEnergyConsumption = ENVIRONMENT.droneProtectingEnergyConsumption
        self.battery = 1 - (ENVIRONMENT.droneBatteryRandomize * random.random())
        self._state = DroneState.IDLE
        self.target = None
        self.targetField = None
        self.targetCharger: Optional[Charger] = None
        self.closestCharger: Optional[Charger] = None
        self.alert = 0.1
        self.lastChargingTime = -1
        super().__init__(location, ENVIRONMENT.droneSpeed)

    @property
    def state(self) -> DroneState:
        return self._state

    @state.setter
    def state(self, value: DroneState):
        self._state = value

    # region battery estimate

    def computeBatteryAfterTime(self, time: float):
        """
        Computes the battery after given time (assuming the `self.droneMovingEnergyConsumption` energy consumption per time step).

        Parameters
        ----------
        time : float
            Time steps.

        Returns
        -------
        float
            Battery after spending energy in given time steps.
        """
        return self.battery - self.timeToEnergy(time)

    batteryAfterTime = ValueEstimate()\
        .inTimeStepsRange(1, 100, trainingPercentage=0.1)\
        .using(WORLD.batteryEstimator)\
        .withBaseline(computeBatteryAfterTime)

    @batteryAfterTime.input()
    @batteryAfterTime.target()  # NumericFeature(0, 1)
    def drone_battery(self):
        return self.battery

    # @batteryAfterTime.input(CategoricalFeature(DroneState))
    # def drone_state(self):
    #     return self.state

    @batteryAfterTime.inputsValid
    @batteryAfterTime.targetsValid
    def not_terminated(self):
        return self.state != DroneState.TERMINATED

    @batteryAfterTime.extra
    def current_time(self):
        return SIMULATION_GLOBALS.currentTimeStep

    @batteryAfterTime.recordValid
    def not_charging(self, inputs, targets, extra):
        return extra['current_time'] >= self.lastChargingTime

    # endregion

    # region helper functions

    def timeToEnergy(self, time, consumptionRate=None):
        """
        Computes the amount of energy which is consumed in the given time.

        Parameters
        ----------
        time : float
            The time (in time steps).
        consumptionRate : float, optional
            The battery consumption per time step, defaults to `self.droneMovingEnergyConsumption`.

        Returns
        -------
        float
            The amount of energy consumed in given time.
        """
        if consumptionRate is None:
            consumptionRate = self.droneMovingEnergyConsumption
        return time * consumptionRate

    def findClosestCharger(self):
        """
        Finds the closest charger comparing the drone distance and WORLD chargers.

        Returns
        -------
        charger
            The closest charger to the drone.
        """
        return min(WORLD.chargers, key=lambda charger: self.location.distance(charger.location))

    def timeToFlyToCharger(self, charger=None):
        """
        Computes the time needed to get to the charger or the closest charger.

        Parameters
        ----------
        charger : Charger, optional
            Specify the charger for measuring the distance, defaults to `self.closestCharger`.

        Returns
        -------
        float
            The time steps needed to fly to the (given) charger.
        """
        if charger is None:
            charger = self.closestCharger
        return self.location.distance(charger.location) / self.speed

    def energyToFlyToCharger(self, charger=None):
        """
        Computes the energy needed to fly to the specified charger.

        Parameters
        ----------
        charger : Charger, optional
             Specify the charger for measuring the distance.

        Returns
        -------
        float
            The energy required to fly to the closest or given charger.
        """
        return self.timeToEnergy(self.timeToFlyToCharger(charger))

    # endregion

    def timeToDoneCharging(self):
        """
        Computes how long it will take to fly to the closest charger and get fully charged, assuming the charger is free.

        Returns
        -------
        int
            The time which the drone will be done charging.
        """
        batteryWhenGetToCharger = self.battery - self.energyToFlyToCharger()
        timeToCharge = (1 - batteryWhenGetToCharger) * self.closestCharger.chargingRate
        return self.timeToFlyToCharger() + timeToCharge

    def needsCharging(self, waitingTime: float) -> bool:
        """
        Checks if the drone needs charging assuming it will have to fly to the charger and wait there until it is available.

        Parameters
        ----------
        waitingTime : float
            The time the drone has to wait, after it gets to the charger, before it can start charging.
        """
        if self.state == DroneState.TERMINATED:
            return False

        if self.energyToFlyToCharger() > self.battery:  # the drone cannot be saved
            return False

        futureBattery = self.batteryAfterTime(self.timeToFlyToCharger() + waitingTime)

        return futureBattery < self.alert

    def checkBattery(self):
        """
        It checks the battery if is below or equal to 0, it is assumed the drone is dead, and it will get removed from the given tasks.
        """
        if self.battery <= 0:
            self.battery = 0
            self.state = DroneState.TERMINATED
            if self.targetField is not None:
                self.targetField.unassign(self)
                
    def move(self, target=None):
        """
        It moves the drone by using the MovingComponent2D.move method, with addition of decreasing the battery in moving consumption rate.
        """
        self.battery = self.battery - self.droneMovingEnergyConsumption
        super().move(self.target)

    def actuate(self):
        """Performs the actions of the drone in one time-step. For each state the actions are different."""
        if self.state == DroneState.TERMINATED:  # no action
            return

        if self.state in (DroneState.IDLE, DroneState.PROTECTING, DroneState.MOVING_TO_FIELD):
            if self.targetCharger is not None:  # drone is accepted for charging -> fly towards the charger
                self.state = DroneState.MOVING_TO_CHARGER
            else:
                if self.targetField is None:  # no target field to protect
                    self.state = DroneState.IDLE  # remain IDLE (it will not consume battery -- it is assumed, that the drone landed in a real-life scenario)
                    return
                # fly to a place in the targetField which needs protection
                self.target = self.targetField.assignPlace(self)
                self.state = DroneState.MOVING_TO_FIELD

        if self.state == DroneState.MOVING_TO_CHARGER:
            if self.targetField is not None:
                self.targetField.unassign(self)

            # fly towards the charger
            self.target = self.targetCharger.provideLocation(self)
            if self.location != self.targetCharger.location:
                self.move()
            else:
                # When the charger is reached, the drone will consume standing energy until it lands on the charger.
                # We are not always certain that by the time drone gets to the charger there is free slot as the actual charging rate can vary based on how many other drones are being charged somewhere else.
                self.battery = self.battery - self.droneProtectingEnergyConsumption

        if self.state == DroneState.MOVING_TO_FIELD:
            if self.location == self.target:
                self.state = DroneState.PROTECTING
                self.battery = self.battery - self.droneProtectingEnergyConsumption
            else:
                self.move()

        if self.state == DroneState.CHARGING:
            self.lastChargingTime = self.current_time()

        self.checkBattery()

    def isProtecting(self, point):
        """
        Checks if the given point is protected by the drone.

        Parameters
        ----------
        point : Point2D
            A given point on the field.

        Returns
        -------
        bool
            True if the given point is within the radius of the drone and drone's state is protecting.
        """
        return (self.state == DroneState.PROTECTING or self.state == DroneState.MOVING_TO_FIELD) and self.location.distance(
            point) <= self.droneRadius

    def protectRadius(self):
        """
        Gives the radius of the drone as form of rectangle to be presented in visualization.

        Returns
        -------
        Point2D, Point2D, Point2D, Point2D
            Top X, Top Y, Bottom X and Bottom Y
        """
        startX = self.location.x - self.droneRadius
        endX = self.location.x + self.droneRadius
        startY = self.location.y - self.droneRadius
        endY = self.location.y + self.droneRadius
        startX = 0 if startX < 0 else startX
        startY = 0 if startY < 0 else startY
        return startX, startY, endX, endY

    def __repr__(self):
        """
        Returns
        -------
        string
            Represent the drone instance in one line.
        """
        return f"{self.id}: state={str(self.state)}, battery={self.battery}"
