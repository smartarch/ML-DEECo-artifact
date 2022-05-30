import math
from typing import List, TYPE_CHECKING

from world import ENVIRONMENT, WORLD
from components.drone_state import DroneState
from components.drone import Drone

from ml_deeco.estimators import NumericFeature, CategoricalFeature
from ml_deeco.simulation import Ensemble, someOf
from ml_deeco.utils import verbosePrint

if TYPE_CHECKING:
    from components.charger import Charger


# All the ensembles are instantiated per charger.
# The order of the ensembles is:
#  1. DroneChargingPreassignment -- divide the drones among the chargers (each drone to its closest charger).
#  2. DroneChargingAssignment -- groups the drones in need of charging.
#  3. AcceptedDronesAssignment -- groups the drones accepted for charging.
# The order of the instances of each ensemble type can be arbitrary as the instances are independent.
# Note that for field protection ensembles, the highest possible priority is 1, which is less than the priority of all charging ensembles. The charging ensembles will thus be always materialized before the field protection ensembles.


class DroneChargingPreAssignment(Ensemble):
    """
    Ensembles for assignment of drones to their closest charger.

    There is an instance of the `DroneChargingPreAssignment` ensemble per charger (which is a static role).
    The ensemble groups all drones for which this charger is the closest (using dynamic role `drones`).
    The drones are saved to the `charger.potentialDrones` list.
    """
    charger: 'Charger'  # static role

    def __init__(self, charger: 'Charger'):
        """
        Parameters
        ----------
        charger : Charger
            The charger (static role member).
        """
        super().__init__()
        self.charger = charger

    def priority(self):
        """Arbitrarily set to 4 to be the highest among charging-related ensembles."""
        return 4

    # dynamic role
    drones: List[Drone] = someOf(Drone)

    @drones.select
    def drones(self, drone: 'Drone', otherEnsembles):
        """Drone can be selected if not terminated, not charging and the charger is the closest one."""
        return drone.state not in (DroneState.TERMINATED, DroneState.MOVING_TO_CHARGER, DroneState.CHARGING) and \
            drone.findClosestCharger() == self.charger

    def actuate(self):
        """Save the selected drones to the `potentialDrones` and update their ˙closestCharger˙."""
        verbosePrint(f"DroneChargingPreassignment: assigned {len(self.drones)} to {self.charger.id}", 4)

        self.charger.potentialDrones = self.drones
        for drone in self.drones:
            drone.closestCharger = self.charger


class DroneChargingAssignment(Ensemble):
    """
    Groups the drones which require charge.

    Again, there is an instance of the `DroneChargingAssignment` ensemble per charger (which is a static role).
    The ensemble groups all drones which need charging (using dynamic role `drones`) among those pre-assigned to the charger.
    The drones are saved to the `charger.waitingDrones` list.

    The ML model is used to estimate the waiting time for a charger slot in order to determine whether the drone needs charging now.
    """
    charger: 'Charger'  # static role

    def __init__(self, charger: 'Charger'):
        """
        Parameters
        ----------
        charger : Charger
            The charger (static role member).
        """
        super().__init__()
        self.charger = charger

    def priority(self):
        """Arbitrarily set to 3 to be materialized second among the charging-related ensembles."""
        return 3

    def waitingTimeBaseline(self, drone):
        return WORLD.waitingTimeBaseline

    # dynamic role with estimate
    drones: List[Drone] = someOf(Drone)\
        .withTimeEstimate()\
        .using(WORLD.waitingTimeEstimator)\
        .withBaseline(waitingTimeBaseline)

    @drones.select
    def drones(self, drone: 'Drone', otherEnsembles):
        """
        Assesses whether a drone needs charging.

        1. We only work with drones pre-assigned to the charger (`potentialDrones`).
        2. We compute an estimated waiting time for a charger slot.
        3. Based on the waiting time estimate and time needed to reach the charger, the drone decides whether it needs charging.
        """
        if drone not in self.charger.potentialDrones:
            return False

        waitingTimeEstimate = self.drones.estimate(drone)
        return drone.needsCharging(waitingTimeEstimate)

    # region ML features

    @drones.estimate.input(NumericFeature(0, 1))
    def battery(self, drone):
        return drone.battery

    @drones.estimate.input(CategoricalFeature(DroneState))
    def drone_state(self, drone):
        return drone.state

    @drones.estimate.input(NumericFeature(0, math.sqrt(ENVIRONMENT.mapWidth ** 2 + ENVIRONMENT.mapHeight ** 2)))
    def charger_distance(self, drone):
        return self.charger.location.distance(drone.location)

    @drones.estimate.input(NumericFeature(0, ENVIRONMENT.chargerCapacity))
    def accepted_drones_count(self, drone):
        return len(self.charger.acceptedDrones)

    @drones.estimate.input(NumericFeature(0, ENVIRONMENT.chargerCapacity * ENVIRONMENT.chargerCount))
    def charger_capacity(self, drone):
        return ENVIRONMENT.chargerCapacity

    @drones.estimate.input(NumericFeature(0, 1))
    def neighbor_drones_average_battery(self, drone):
        if drone.targetField is not None:
            k = len(drone.targetField.protectingDrones)
            if k == 0:
                return 0
            return sum([drone.battery for drone in drone.targetField.protectingDrones]) / k
        else:
            return 0

    @drones.estimate.input(NumericFeature(0, 1))
    def neighbor_drones(self, drone):
        if drone.targetField is not None:
            return len(drone.targetField.protectingDrones) / len(drone.targetField.places)
        else:
            return 0

    @drones.estimate.input(NumericFeature(0, 1))
    def potential_drones(self, drone):
        return len(self.charger.potentialDrones)

    @drones.estimate.input(NumericFeature(0, ENVIRONMENT.chargerCapacity))
    def accepted_drones_missing_battery(self, drone):
        return sum([1 - drone.battery for drone in self.charger.acceptedDrones])

    @drones.estimate.input(NumericFeature(0, ENVIRONMENT.chargerCapacity))
    def charging_drones_count(self, drone):
        return len(self.charger.chargingDrones)

    @drones.estimate.input(NumericFeature(0, ENVIRONMENT.chargerCapacity))
    def charging_drones_missing_battery(self, drone):
        return sum([1 - drone.battery for drone in self.charger.chargingDrones])

    @drones.estimate.input(NumericFeature(0, ENVIRONMENT.droneCount))
    def potential_drones_with_lower_battery(self, drone):
        return len([d for d in self.charger.potentialDrones if d.battery < drone.battery])

    @drones.estimate.input(NumericFeature(0, ENVIRONMENT.chargerCapacity))
    def waiting_drones_count(self, drone):
        return len(self.charger.waitingDrones)

    @drones.estimate.input(NumericFeature(0, ENVIRONMENT.droneCount))
    def waiting_drones_with_lower_battery(self, drone):
        return len([d for d in self.charger.waitingDrones if d.battery < drone.battery])

    # endregion

    @drones.estimate.inputsValid
    @drones.estimate.conditionsValid
    def is_preassigned(self, drone):
        """We only collect data for the ML if the drone is pre-assigned to the charger."""
        return drone in self.charger.potentialDrones

    @drones.estimate.condition
    def is_accepted(self, drone):
        """Condition for the estimate of waiting time. The waiting ends when the drone is accepted for charging."""
        return drone in self.charger.acceptedDrones

    def actuate(self):
        """Save the selected drones to the `waitingDrones` list of the charger."""
        verbosePrint(f"DroneChargingAssignment: assigned {len(self.drones)} to {self.charger.id}", 4)

        self.charger.waitingDrones = self.drones


class AcceptedDronesAssignment(Ensemble):
    """
    Groups the drones accepted for charging. These are drones which fly towards the charger, and they will start charging when they get there.

    Again, there is an instance of the `AcceptedDronesAssignment` ensemble per charger (which is a static role).
    The ensemble groups all drones accepted for charging on this charger (using dynamic role `drones`).
    The drones are saved to the `charger.acceptedDrones` list.
    """
    charger: 'Charger'  # static role

    def __init__(self, charger: 'Charger'):
        """
        Parameters
        ----------
        charger : Charger
            The charger (static role member).
        """
        super().__init__()
        self.charger = charger

    def priority(self):
        """Arbitrarily set to 2 to be materialized last among the charging-related ensembles (but before the field protection ensembles)."""
        return 2

    # dynamic role
    drones: List[Drone] = someOf(Drone)

    @drones.cardinality
    def drones(self):
        """The capacity of the charger is the upper limit of how many drones can be selected for this role."""
        return 0, self.charger.acceptedCapacity

    @drones.select
    def drones(self, drone: 'Drone', otherEnsembles):
        """
        Decides which drones should be accepted for charging.

        a) Drones which were accepted earlier are selected again (until they reach the charger)
        b) Among the drones in need of charging (`waitingDrones`), we consider those, for which there would be a free charging slot when they reached the charger if they started flying towards it now.
        """
        return drone in self.charger.acceptedDrones or \
            drone in self.charger.waitingDrones and \
            self.charger.timeToDoneCharging(len(self.drones)) <= drone.timeToFlyToCharger()

    @drones.utility
    def drones(self, drone: 'Drone'):
        """Orders the drones by the time needed to finish charging them (time to reach the charger + time to charge the battery). The drones accepted before have higher utility than all the new drones."""
        if drone in self.charger.acceptedDrones:
            return 1  # keep the accepted drones from previous time steps
        return -drone.timeToDoneCharging()

    def actuate(self):
        """Saves the selected drones to the `acceptedDrones` list and updates their `targetCharger`."""
        verbosePrint(f"AcceptedDronesAssignment: assigned {len(self.drones)} to {self.charger.id}", 4)

        self.charger.acceptedDrones = self.drones
        for drone in self.drones:
            drone.targetCharger = self.charger


def getEnsembles() -> List[Ensemble]:
    """
    One instance of each ensemble type for each charger.
    """

    ensembles = \
        [DroneChargingPreAssignment(charger) for charger in WORLD.chargers] + \
        [DroneChargingAssignment(charger) for charger in WORLD.chargers] + \
        [AcceptedDronesAssignment(charger) for charger in WORLD.chargers]

    return ensembles
