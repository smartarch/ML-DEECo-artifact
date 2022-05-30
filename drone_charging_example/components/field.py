import math
import random
from world import ENVIRONMENT
from components.drone_state import DroneState
from ml_deeco.simulation import Point2D


class Field:
    """

    A Field that the simulation is running on.
    This defines an Area, which is technically a set of points.
    The set of points consist of 4 doubles as [x1, y1, x2, y2]

        (x1,y1) .__________
                |          |
                |__________|.(x2,y2)

    Attributes
    -------
    droneRadius : int
        The protecting radius of drones. The field is divided into places according to this value.
    topLeft : List
        [x1, y1] indicating the top left point of the field rectangle.
    bottomRight : List
        [x2, y2] indicating the bottom right point of the field rectangle.
    places : List
        List of all places that require protecting.
    protectingDrones : dict (drone -> place)
        Map of current protecting drones to the assigned place.
    memory : dict (drone -> place)
        Map of all-time protecting drones to the assigned place.
    crops : dict (crops -> int)
        Map of crops with corresponding damage value.
    damaged : list
        List of all damaged crops.
    damage : int
        Total damage.
    allCrops : int
        Total crops.
    
    Static Members:
    --------
    DAMAGE_DEPTH : int
        How many times a crop must be attacked to be considered as damaged.
    """
    # Field counter
    Count = 0
    DAMAGE_DEPTH = 2
    topLeft: Point2D
    bottomRight: Point2D

    def __init__(self, pointLists):
        """

        Initiate a field of places and crops. Each filed has N places, with M crops.
        M = Height X Width; thus Area
        N = M / Drone Radius
        Places are saved as a list, and crops are saved as a Dictionary.
        The damaged list is only for visualization.
        
        Parameters
        ----------
        pointLists : List
            [x1,y1,x2,y2] to create a rectangle.
        """
        Field.Count = Field.Count + 1
        self.droneRadius = ENVIRONMENT.droneRadius
        self.id = f"FIELD_{Field.Count}"
        self.topLeft = Point2D(pointLists[0], pointLists[1])
        self.bottomRight = Point2D(pointLists[2], pointLists[3])
        self.places = []
        self.protectingDrones = {}
        self.memory = {}
        self.crops = {}  # for birds
        self.damaged = []  # for visualization
        self.damage = 0
        for i in range(self.topLeft.x + self.droneRadius, self.bottomRight.x, round(self.droneRadius )):
            for j in range(self.topLeft.y + self.droneRadius, self.bottomRight.y, round(self.droneRadius)):
                self.places.append(Point2D(i, j))
        for i in range(self.topLeft.x, self.bottomRight.x):
            for j in range(self.topLeft.y, self.bottomRight.y):
                self.crops[(i, j)] = 0
        self.allCrops = len(self.crops)

    def locationPoints(self):
        """

        Converst the field into W X H of points.

        Returns
        -------
        list
            List of points.
        """
        points = []
        for x in range(self.topLeft.x, self.bottomRight.x):
            for y in range(self.topLeft.y, self.bottomRight.y):
                points.append(Point2D(x, y))
        return points

    def isPointOnField(self, point):
        """

        Checks if the given point is on the map.
        This works in close reality of birds detecting crops. 

        Parameters
        ----------
        point : Point2D
            A point asked by the birds.

        Returns
        -------
        bool
            Returns True if the point is on the map.
        """
        return self.topLeft.x <= point.x < self.bottomRight.x and \
               self.topLeft.y <= point.y < self.bottomRight.y

    def closestDistanceToDrone(self, drone):
        """

        finds the nearest point to a drone. To assign the nearest place.

        Parameters
        ----------
        drone : Drone
            Any given drone.

        Returns
        -------
        int
            minimum distance to the drone.
        """
        distances = []
        for place in self.places:
            dx = place.x - drone.location.x
            dy = place.y - drone.location.y
            distances.append(math.sqrt(dx * dx + dy * dy))
        return min(distances)

    def assignPlace(self, drone):
        """

        The drone asks the field for a place to protect.
        The Field assigns a place to the drone as:
            1- if the drone had been protecting a place before being IDLE again, the field returns the same place using Memory Map.
            2- if the drone is new to the field, *nearest* non-protected place will be assigned to it.
            3- if all places are already booked, a random place will be selected to

        In the case there are more drones than places, 2 drones might protect the same area.

        Parameters
        ----------
        drone : Drone
            The drone which is asking to protect this field. 

        Returns
        -------
        Point2D
            The protecting point (center of place).
        """
        if drone not in self.protectingDrones:
            if drone not in self.memory:
                listOfEmptyPlaces = [place for place in self.places if place not in [self.memory[d] for d in self.memory]]
                if len(listOfEmptyPlaces) <= 0:
                    self.protectingDrones[drone] = random.choice(self.places)
                else:
                    self.protectingDrones[drone] = min(listOfEmptyPlaces, key=lambda p: p.distance(drone.location))
                self.memory[drone] = self.protectingDrones[drone]
            else:
                self.protectingDrones[drone] = self.memory[drone]
        return self.protectingDrones[drone]

    def unassign(self, drone):
        """

        unassign the place for the drones that depart for charging. 
        The drone will remain in memory of the field, unless the drone is dead.

        Parameters
        ----------
        drone : [type]
            [description]
        """
        if drone in self.protectingDrones:
            del self.protectingDrones[drone]
            if drone.state == DroneState.TERMINATED:
                del self.memory[drone]

    def randomLocation(self):
        """

        Returns a random point within the field.

        Returns
        -------
        Point2D
            A random point within the field.
        """
        return Point2D.random(self.topLeft.x, self.topLeft.y, self.bottomRight.x, self.bottomRight.y)

    def locationDamaged(self, location):
        """

        Marks the location as *eaten*, it if happens Field.DAMAGE_DEPTH times,
        the crops will be marked as damaged and will be deleted from crops list.

        Parameters
        ----------
        location : Point2D
            The eaten location.
        """
        p = (location.x, location.y)
        if p in self.crops:
            self.crops[p] = self.crops[p] + 1
            if self.crops[p] == Field.DAMAGE_DEPTH:
                del self.crops[p]
                self.damaged.append(Point2D(location.x, location.y))
                self.damage = self.damage + 1

    def randomUndamagedCrop(self):
        """

        Finds an undamaged crop, chosen randomly.

        Returns
        -------
        Point2D
            A random point which is not yet fully damaged.
        """
        if len(self.crops) == 0:
            return None
        safe = random.choice([p for p in self.crops])
        return Point2D(safe[0], safe[1])
 
    def __str__(self):
        """

        [extended_summary]

        Returns
        -------
        str
            Description of the field in one line.
        """
        return f"{self.id},{self.topLeft},{self.bottomRight}"
