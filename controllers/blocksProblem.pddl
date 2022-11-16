; To solve with fast downward
; ./fast-downward.py blocksDomain.pddl blocksProblem.pddl --search "astar(lmcut())"

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
