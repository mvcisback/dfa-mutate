import random
from itertools import cycle
from typing import Iterable
from uuid import uuid1

import funcy as fn
from dfa import DFA
from dfa.utils import dfa2dict, dict2dfa


__all__ = [
    "add_state",
    "change_start",
    "change_transition",
    "generate_mutations"
    "relabel_state",
    "sample_mutation"
]


def generate_mutations(orig: DFA, rng=random) -> Iterable[DFA]:
    yield from (f(orig, rng) for f in cycle(MUTATIONS))


def sample_mutation(orig: DFA, n=20, score=fn.constantly(1.0), rng=random) -> DFA:
    import numpy as np

    dfas = fn.take(n, generate_mutations(orig, rng=rng))
    scores = fn.lmap(score, dfas)
    softmax_weights = np.exp(scores - np.max(scores))
    return rng.choices(dfas, softmax_weights)[0]


def add_state(orig: DFA, rng=random) -> DFA:
    orig = orig.normalize()  # States are now [0..n].
    new_state = len(orig.states())

    output = rng.choice(list(orig.outputs))
    old_state = rng.choice(list(orig.states()))
    sym = rng.choice(list(orig.inputs))

    def label(s):
        return orig._label(s) if s != new_state else output

    def transition(s, c):
        if s == new_state:
            return new_state  # New state is a sink.
        if (s, c) == (old_state, sym):
            return new_state  # Force new_state to be reachable.
        return orig._transition(s, c)

    return DFA(
        label=label, transition=transition,
        start=orig.start, inputs=orig.inputs, outputs=orig.outputs,
    )


def change_transition(orig: DFA, rng=random) -> DFA | None:
    if (len(orig.inputs) <= 1) or (len(orig.states()) <= 1):
        return None

    state1 = rng.choice(list(orig.states()))
    state2 = rng.choice(list(orig.states()))
    sym = rng.choice(list(orig.inputs))

    def transition(s, c):
        if (s, c) == (state1, sym):
            return state2
        return orig._transition(s, c)

    return DFA(
        label=orig._label, transition=transition,
        start=orig.start, inputs=orig.inputs, outputs=orig.outputs,
    )


def change_start(orig: DFA, rng=random) -> DFA | None:
    if len(orig.states()) == 1:
        return None

    start = rng.choice(list(orig.states() - {orig.start}))
    return DFA(
        label=orig._label, transition=orig._transition,
        start=start, inputs=orig.inputs, outputs=orig.outputs
    )


def relabel_state(orig: DFA, rng=random) -> DFA | None:
    if len(orig.outputs) <= 1:
        return None

    state = rng.choice(list(orig.states()))
    current_label = orig._label(state)
    new_label = rng.choice(list(orig.outputs - {current_label}))

    def label(s):
        return new_label if s == state else orig._label(s)

    return DFA(
        label=label, transition=orig._transition,
        start=orig.start, inputs=orig.inputs, outputs=orig.outputs
    )


MUTATIONS = [add_state, change_start, change_transition, relabel_state] 
