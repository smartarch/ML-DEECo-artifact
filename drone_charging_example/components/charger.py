import random
from typing import List, TYPE_CHECKING

from world import ENVIRONMENT, WORLD
from components.drone_state import DroneState
from ml_deeco.simulation import StationaryComponent2D, Point2D

if TYPE_CHECKING:
    from components.drone import Drone


class Charger(StationaryComponent2D):
    """
    The charger class represents the charger stations providing energy for drones in the simulation.
    The charging rate and capacity is defined in the WORLD and ENVIRONMENT objects shared with all components.

    Attributes
    ----------
    chargingRate : float
        The charging rate the charger is supposed to provide per time-step for landing drones.
    acceptedCapacity : int
        How many drones could be charged at the same time.
    potentialDrones : list
        The potential drones for the charger, the close drone ones despite their battery level.
    waitingDrones : list
        The list of drones that are in need of charging, but not accepted yet.
    acceptedDrones : list
        The list of accepted drones that are moving toward the charger.
    chargingDrones : list
        The list of drones that are being charged.
    """

    def __init__(self, location):
        """
        Initiate the charger instance with constant position on the map.

        Parameters
        ----------
        location : Point2D
            The location of the charger (constant).
        """
        super().__init__(location)
        self.chargingRate = ENVIRONMENT.chargingRate
        self.acceptedCapacity = ENVIRONMENT.chargerCapacity
        self.potentialDrones: List[Drone] = []  # these belong to this charger and are not waiting or being charged
        self.waitingDrones: List[Drone] = []  # drones in need of being charged, waiting for acceptance
        self.acceptedDrones: List[Drone] = []  # drones accepted for charging, they move to the charger
        self.chargingDrones: List[Drone] = []  # drones currently being charged

    def startCharging(self, drone: 'Drone'):
        """
        This is called when the drone is in the correct location and starts charging.
        """
        self.acceptedDrones.remove(drone)
        self.chargingDrones.append(drone)
        drone.state = DroneState.CHARGING

    def doneCharging(self, drone: 'Drone'):
        """
        This is called when the drone is fully charged.
        """
        drone.battery = 1
        drone.targetCharger = None
        drone.state = DroneState.IDLE
        self.chargingDrones.remove(drone)

    def timeToDoneCharging(self, alreadyAccepted=0):
        """
        Computes how long it will take for the charger to have a free slot.

        Parameters
        ----------
        alreadyAccepted : int, optional
            Number of drones already accepted for charging.

        Returns
        -------
        float
            Time steps until a free charging slot.
        """
        batteries = sorted(map(lambda d: d.battery, self.chargingDrones), reverse=True)
        if len(batteries) > alreadyAccepted:
            nthMaxBattery = batteries[alreadyAccepted]
        else:
            nthMaxBattery = 1
        return (1 - nthMaxBattery) / self.chargingRate

    def randomNearLocation(self) -> Point2D:
        """Returns a random location near the charger."""
        return Point2D(self.location.x + random.randint(1, 3), self.location.y + random.randint(1, 3))

    def provideLocation(self, drone):
        """
        Gives the location of the charger to the drone.
        If the drone is accepted it can fly to the charger, the standby time is only due to unexpected latency in charging.

        Parameters
        ----------
        drone : Drone
            The drone which is asking for the location.

        Returns
        -------
        Point2D
            The point to be set as the target of drone.
        """
        if drone in self.chargingDrones or drone in self.acceptedDrones:
            return self.location
        else:
            return self.randomNearLocation()

    def actuate(self):
        # Charging rate depends on the number of drones currently being charged.
        # For example if ENVIRONMENT.totalAvailableChargingEnergy = 0.12, and the charging rate is 0.04, then it means 3 drones could simultaneously change at one or different chargers.
        # But for instance with 0.12, if there are 4 drones, they will get 0.03 charge rate.
        totalChargingDrones = sum([len(charger.chargingDrones) for charger in WORLD.chargers])
        if totalChargingDrones > 0:
            currentChargingRate = min(totalChargingDrones * ENVIRONMENT.chargingRate,
                                      ENVIRONMENT.totalAvailableChargingEnergy) / totalChargingDrones
            ENVIRONMENT.currentChargingRate = currentChargingRate
        else:
            currentChargingRate = 0

        # charge the drones
        for drone in self.chargingDrones:
            drone.battery += currentChargingRate
            if drone.battery >= 1:
                self.doneCharging(drone)

        # move drones from accepted to charging
        freeChargingPlaces = self.acceptedCapacity - len(self.chargingDrones)
        for i in range(freeChargingPlaces):
            for drone in self.acceptedDrones:
                if drone.location == self.location:
                    self.startCharging(drone)
                    break

    def __repr__(self):
        """
        Represent the charger in one line.

        Returns
        -------
        str
            Prints information about all the queues of the charger.
        """
        return f"{self.id}: C={len(self.chargingDrones)}, A={len(self.acceptedDrones)}, W={len(self.waitingDrones)}, P={len(self.potentialDrones)}"
