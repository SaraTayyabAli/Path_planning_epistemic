from collections import defaultdict
import random


class HumanTransitionModel:
    """Original human-motion probability rule, exposed without Monte Carlo sampling."""

    def __init__(self, env, goal):
        self.env = env
        self.goal = goal
        self._neighbors = {}

    def neighbors(self, state):
        if state not in self._neighbors:
            self._neighbors[state] = tuple(self.env.get_legal_neighbors_human(state))
        return self._neighbors[state]

    def distribution(self, state, back_point, zeta):
        if state == self.goal:
            return ((state, None, 1.0),)

        next_states = list(self.neighbors(state))

        if self.goal in next_states:
            return ((self.goal, None, 1.0),)

        if len(next_states) <= 1:
            return ((state, back_point, 1.0),)

        ignored_for_priority = {state}
        if back_point is not None and back_point != state:
            ignored_for_priority.add(back_point)

        candidates = [s for s in next_states if s not in ignored_for_priority]
        if not candidates:
            if back_point is None:
                return ((state, back_point, 1.0),)
            # The original HumanAgent returns before updating back_point here.
            return ((back_point, back_point, 1.0),)

        distances = [
            self.env.manhattan_distance(s, self.goal)
            if s not in ignored_for_priority
            else float("inf")
            for s in next_states
        ]
        min_distance = min(distances)
        preferred = {
            i for i, distance in enumerate(distances) if distance == min_distance
        }

        n_preferred = len(preferred)
        n_other = len(next_states) - n_preferred
        remaining = 1.0 - n_other * zeta

        if remaining < -1e-12:
            raise ValueError(
                f"zeta={zeta} makes the preferred-action probability negative "
                f"at human state {state}. Choose a smaller interval."
            )

        preferred_probability = max(remaining, 0.0) / n_preferred
        probabilities = defaultdict(float)

        for i, chosen in enumerate(next_states):
            probability = preferred_probability if i in preferred else zeta
            next_back = state if chosen != state else back_point
            probabilities[(chosen, next_back)] += probability

        total = sum(probabilities.values())
        if not 0.999999 <= total <= 1.000001:
            raise RuntimeError(
                f"Human transition probabilities sum to {total} at {state}"
            )

        return tuple(
            (human_next, back_next, probability / total)
            for (human_next, back_next), probability in probabilities.items()
            if probability > 0.0
        )

    def sample(self, state, back_point, zeta, rng=None):
        rng = rng or random
        transitions = self.distribution(state, back_point, zeta)
        draw = rng.random()
        cumulative = 0.0

        for human_next, back_next, probability in transitions:
            cumulative += probability
            if draw <= cumulative:
                return human_next, back_next

        human_next, back_next, _ = transitions[-1]
        return human_next, back_next
