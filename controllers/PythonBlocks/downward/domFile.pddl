(define 
    (domain domFile)
    ;PDDL code for predicates
    (:predicates 
        (INSTRUCTION ?x)
        (complete ?x)
    )
    ;PDDL code for first action
    (:action runIns
        :parameters (?ins)
        :precondition (INSTRUCTION ?ins)
        :effect (complete ?ins)
    )
    ;PDDL code for last action
)
