from py2pddl import Domain, create_type
from py2pddl import predicate, action, goal, init

'''
To solve with fast downward
./fast-downward.py domain.pddl problem.pddl --search "astar(lmcut())"
(define 
    (domain blocksDomain)
    ; Predicates (defines what arguments they have)
    (:predicates
        (BLOCK ?x)
        (clear ?x)
        (on ?x,?y)

    )
    ; Action with preconditions (used to determine edges from state in graph search)
    (:action move
        :parameters (?block ?underObject ?newUnderObject)
        :precondition (and
            (BLOCK ?block) 
            (on ?block ?underObject)
            (clear ?block)
            (clear ?newUnderObject))
        :effect (and (on ?block ?newUnderObject)
            (clear ?underObject)
            (not (on ?block ?underObject))
            (not (clear ?newUnderObject)))
    )
)
'''
class blocksDomain(Domain):
    Object = create_type("Object")
    
    @predicate(Object)
    def Block(self,object):
         # true if object is a block
        pass
    
    @predicate(Object)
    def fixed(self,object):
         # true if object is fixed
        pass

    @predicate(Object,Object)
    def on(self,objectA,objectB):
         # true if objectA is on objectB
        pass

    @predicate(Object)
    def clear(self,blockA):
        # true if blockA can be grasped without knocking over other blocks i.e. blockA is on top
        pass
    
    @action(Object,Object,Object)
    def move(self,block,underObject,newUnderObject):
        # precondition is that block is of type Block
        # underObject is object currently underneath block (Location or Block)
        # newUnderObject is object desired to be underneath block 
        precond = [~self.fixed(block),self.Block(block),self.on(block,underObject),self.clear(block),self.clear(newUnderObject)]
        effect = [~self.on(block,underObject),self.on(block,newUnderObject),self.clear(block),self.clear(underObject),~self.clear(newUnderObject)]
        return precond,effect

'''
(define (problem blocksProblem)
   (:domain blocksDomain)
   ;PDDL code for objects
   ; All objects are initially equal until a predicate "type" is assigned to them
   (:objects
    locA
    locB
    locC
    blockA
    blockB
    blockC
   )
   ; PDDL code for initial state
   ; Asserting predicates that hold true initially in this world
   (:init
   ; Which blocks are on top of blocks
    (on blockC locA)
    (on blockB blockC)
    (on blockA blockB)

    ; Assigning type BLOCK to objects
    (BLOCK blockA)
    (BLOCK blockB)
    (BLOCK blockC)
    (clear blockA)
    (clear locB)
    (clear locC)
   ) 
        
   ;Specifying goal of system
   (:goal (and 
    (on blockC locC)
    (on blockB blockC)
    (on blockA blockB)
   ))
)


'''

class blocksProblem(blocksDomain):
    def __init__(self):
        super().__init__()
        self.objects = blocksDomain.Object.create_objs(["blockA","blockB","blockC","locA","locB","locC"],prefix="")

    @init
    def init(self) -> list:
        initTruths = [
            self.on(self.objects["blockC"],self.objects["locA"]),
            self.on(self.objects["blockB"],self.objects["blockC"]),
            self.on(self.objects["blockA"],self.objects["blockB"]),
            self.clear(self.objects["blockA"]),
            self.clear(self.objects["locB"]),
            self.clear(self.objects["locC"]),
            self.Block(self.objects["blockA"]),
            self.Block(self.objects["blockB"]),
            self.Block(self.objects["blockC"]),
            self.fixed(self.objects["locA"]),
            self.fixed(self.objects["locB"]),
            self.fixed(self.objects["locC"])
        ]
        return initTruths
    
    @goal
    def goal(self) -> list:
        goalTruths = [
            self.on(self.objects["blockC"],self.objects["locC"]),
            self.on(self.objects["blockB"],self.objects["blockC"]),
            self.on(self.objects["blockA"],self.objects["blockB"])
        ]
        return goalTruths
