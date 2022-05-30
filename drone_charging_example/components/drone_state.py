from enum import IntEnum


class DroneState(IntEnum):
    """
    Drone states:
        IDLE: a default state for drones.
        PROTECTING: when the drones are protecting the fields.
        MOVING_TO_CHARGING: when the drones are moving/queuing for a charger.
        CHARGING: when the drones are being charged.
        TERMINATED: when the drones' battery is 0, and they do not operate anymore.
    """
    IDLE = 0
    PROTECTING = 1
    MOVING_TO_FIELD = 2
    MOVING_TO_CHARGER = 3
    CHARGING = 4
    TERMINATED = 5