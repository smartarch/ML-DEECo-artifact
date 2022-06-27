import enum
from typing import TYPE_CHECKING

from ml_deeco.utils import verbosePrint

if TYPE_CHECKING:
    from components import SecurityComponent


def allow(subjects, action, object: 'SecurityComponent'):
    subjects = list(subjects)
    for s in subjects:
        object.allow(s, action)
    if len(subjects) > 0:
        verbosePrint(f"Allowing {subjects} '{action}' '{object}'", 6)


def now():
    from configuration import CONFIGURATION
    return CONFIGURATION.experiment.currentTimeStep


class DayOfWeek(enum.IntEnum):
    MONDAY = 0
    TUESDAY = 1
    WEDNESDAY = 2
    THURSDAY = 3
    FRIDAY = 4
    SATURDAY = 5
    SUNDAY = 6
