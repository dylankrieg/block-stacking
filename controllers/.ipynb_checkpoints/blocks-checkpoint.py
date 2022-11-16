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

from py2pddl import Domain, create_type
from py2pddl import predicate, action, goal, init
 
class blocksProblem(blocksDomain):
    def __init__(self,data):
        super().__init__()
        self.objects = blocksDomain.Object.create_objs(["blockA","blockB","blockC","locA","locB","locC"],prefix="")
        self.initTruths = [self.on(self.objects["blockC"],self.objects["locA"]),
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
        self.fixed(self.objects["locC"])]

    @init
    def init(self) -> list:
        return self.initTruths
    
    @goal
    def goal(self) -> list:
        goalTruths = [
            self.on(self.objects["blockC"],self.objects["locC"]),
            self.on(self.objects["blockB"],self.objects["blockC"]),
            self.on(self.objects["blockA"],self.objects["blockB"])
        ]
        return goalTruths


# data = None
# problem = blocksProblem(data)
# print("Creating domain pddl file")
# problem.generate_domain_pddl()
