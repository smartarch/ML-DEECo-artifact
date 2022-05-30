import random
from enum import Enum
from world import ENVIRONMENT, WORLD
from ml_deeco.simulation import MovingComponent2D

class BirdState(Enum):
    """
        An enumerate property for the birds.
        IDLE: a default state for birds, when they are out of fields.
        ATTACKING: a state where a bird has a field in mind, and attacking it.
        FLEEING: a state where a bird is running away from drones
    """
    IDLE = 0
    MOVING_TO_FIELD = 1
    OBSERVING = 2
    EATING = 3
    FLEEING = 4

class Bird(MovingComponent2D):
    """

    The birds are the threats to the crops of the fields. 
    They find undamaged crops and eat them in one or multiple visits. 
    If they see a drone around, they flee to random places of the map (which are not fields). The birds behavior is flavoured with random factors that would change results of the same running simulation, thus one ought to attempt multiple runs and average the results.

    Attributes
    ----------
    target : Point2D
        The target location, field or empty place.
    state: BirdState
        The state of birds.
    field: Field
        The target field for attacking.

    Static Members
    -------
        StayProbability: how likely for a bird to stay in the same place.
        AttackProbability: how likely for a bird to attack a field.
        ReplaceProbability: how likely for a bird to change to another field.
    """
    StayProbability = 0.85
    AttackProbability = 0.12
    ReplaceProbability = 0.03

    def __init__( self,location):
        """

        Initiate the bird instance. 

        Parameters
        ----------
        location : Point2D
            the point which is given by the World.
        """
        self.state = BirdState.IDLE
        self.target = None
        self.field = None
        super().__init__(location, ENVIRONMENT.birdSpeed)

    def moveToNewField(self):
        """

        Moves the bird to the new undamaged field. If such place does not exist, the bird goes IDLE.
        """
        self.field = random.choice(WORLD.fields)
        newTarget = self.field.randomUndamagedCrop()
        if newTarget is None:
            self.field = None
            self.state = BirdState.IDLE
        else:
            self.target = newTarget
            self.state = BirdState.MOVING_TO_FIELD

    def moveToNoField(self):
        """

        Flee from the drones and fly away to an empty place.
        """
        self.field = None
        self.target = random.choice(WORLD.emptyPoints)
        self.state = BirdState.FLEEING

    def moveWithinSameField(self):
        """
        Move to another area within the same field.
        """
        if self.field == None:
            self.moveToNewField()
        else:
            newTarget = self.field.randomUndamagedCrop()
            if newTarget is None:
                self.field = None
                self.state = BirdState.IDLE
            else:
                self.target = newTarget
                self.state = BirdState.MOVING_TO_FIELD

    def actuate(self):
        """
        it perform the actions of the bird in one time-step.
        For each state it performs differently:
        IDLE:
            Get a random value and if it was bigger than StayProbability and less than ReplaceProbability, it will attack an intact crops of field. The state will change to MOVING_TO_FIELD. If the probability stated over ReplaceProbability, then it will move to another place. This behavior represents that the birds are not always particularly attacking.
        MOVING_TO_FIELD:
            When reached to a field, it will first observe the area, the state changes to OBSERVING.
        OBSERVING:
            When the bird is observing around and sees a drone, it flees away.
        FLEEING:
            The bird flees to a non-field part of the map. To simplify the calculations, we have set 100 points on the map that are not fields, the birds simply chooses one of them.
        EATING:
            First the bird will perform same actions as in OBSERVING. If no drone is around, the bird starts eating 1 unit of the field crops. To be considered in this simulation, when a crop is eaten (2 times), it is declared as damaged.
            After the eating, the bird performs  the same actions as IDLE.
        """
        if self.state == BirdState.IDLE:
            probability = random.random()
            # bird moves if we crossed threshold of StayProbability randomly
            if probability > Bird.StayProbability:
                probability -= Bird.StayProbability
                # to decide wether attack a farm or just move to another place
                if probability < Bird.AttackProbability:
                    self.moveToNewField()
                else:
                    self.moveToNoField()
        if self.state == BirdState.MOVING_TO_FIELD:
            if self.location == self.target:
                self.state = BirdState.OBSERVING
            else:
                self.move(self.target)
        if self.state == BirdState.FLEEING:
            if self.location == self.target:
                self.state = BirdState.IDLE
            else:
                self.move(self.target)
        if self.state == BirdState.OBSERVING or self.state == BirdState.EATING:
            if WORLD.isProtectedByDrone(self.location):
                self.moveToNoField()
                self.state = BirdState.FLEEING
            else:
                self.state = BirdState.EATING
        if self.state == BirdState.EATING:
            self.field.locationDamaged(self.location)
            probability = random.random()
            if probability > Bird.StayProbability:
                probability -= Bird.StayProbability
                if probability < Bird.AttackProbability:
                    self.moveWithinSameField()
                else:
                    self.moveToNoField()
                    self.state = BirdState.FLEEING
            else:
                self.moveWithinSameField()

    def __repr__(self):
        """

        Returns
        -------
        str
            Represnet the bird object in one line
        """
        return f"{self.id}: state={self.state}"