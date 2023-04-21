import random

import dfa

import dfa_mutate


INITIAL_DFA = dfa.dict2dfa({
    0: ('x', {0: 0, 1: 1}),
    1: ('y', {0: 1, 1: 2}),
    2: ('z', {0: 2, 1: 3}), 
    3: ('x', {0: 3, 1: 0})
}, start = 0)
STATES = INITIAL_DFA.states()


def test_mutation():
    random.seed(0)
    mutations = [
        dfa_mutate.add_state,
        dfa_mutate.relabel_state,
        dfa_mutate.change_transition,
        dfa_mutate.change_start,
        dfa_mutate.sample_mutation
    ]
    for mutate in mutations:
        new_dfa = mutate(INITIAL_DFA)    
        assert new_dfa != INITIAL_DFA
