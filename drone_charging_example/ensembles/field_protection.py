from typing import TYPE_CHECKING, List
from world import WORLD
from components.drone_state import DroneState
from components.drone import Drone
from ml_deeco.simulation import Ensemble, oneOf
from ml_deeco.utils import verbosePrint
if TYPE_CHECKING:
    from components.field import Field


class FieldProtection(Ensemble):
    """
    Ensemble for management of the field protection. We create one instance of the ensemble per field.
    In every time step, one idle drone can be selected to become a member of the ensemble.
    The member is the assigned the task to protect the field.
    """

    field: 'Field'  # static role

    def __init__(self, field: 'Field'):
        """
        Parameters
        ----------
        field : Field
            The field which needs protection.
        """
        super().__init__()
        self.field = field

    # dynamic role
    drone: Drone = oneOf(Drone)

    def priority(self):
        """
        The priority is the fraction of places in the field which are not protected.
        Note that the highest possible priority is 1, which is less than the priority of all charging ensembles. The charging ensembles will thus be always materialized before the field protection ensembles.
        """
        places = len(self.field.places)
        protectedPlaces = len(self.field.protectingDrones)
        unprotectedPlaces = places - protectedPlaces
        return unprotectedPlaces / places

    def situation(self):
        """Only materialize the ensemble if we don't have enough protecting drones."""
        return len(self.field.protectingDrones) < len(self.field.places)

    @drone.select
    def drone(self, drone: 'Drone', otherEnsembles):
        """Select only idle drones not selected by the otehr ensembles."""
        if drone in (ens.drone for ens in otherEnsembles if isinstance(ens, FieldProtection)):
            return False
        return drone.state == DroneState.IDLE

    @drone.utility
    def drone(self, drone: 'Drone'):
        """Orders the drones by the distance from the field (smaller distance first)."""
        return -self.field.closestDistanceToDrone(drone)

    def actuate(self):
        """
        Assign the selected drone to the field by setting its targetField property.
        """
        self.drone.targetField = self.field
        verbosePrint(f"Protecting Ensemble: assigning {self.drone.id} to {self.field.id}", 4)


def getEnsembles() -> List[Ensemble]:
    """
    One instance of field protection ensemble for each field.
    """
    ensembles = [FieldProtection(field) for field in WORLD.fields]
    return ensembles
