(define 
    (domain blocksDomain)
    ; Predicates (defines what arguments they have)
    (:predicates
        (INSTRUCTION ?x)
        (BLOCK ?x)
        (complete ?x)
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



