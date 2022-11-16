(define
	(problem blocks)
	(:domain blocks)
	(:objects
		blockA blockB blockC locA locB locC - object
	)
	(:init (on blockC locA) (on blockB blockC) (on blockA blockB) (clear blockA) (clear locB) (clear locC) (Block blockA) (Block blockB) (Block blockC) (fixed locA) (fixed locB) (fixed locC))
	(:goal (and (on blockC locC) (on blockB blockC) (on blockA blockB)))
)
