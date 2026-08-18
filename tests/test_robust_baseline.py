import unittest

from environment import Env
from human_transition import HumanTransitionModel
from method.robust_mprl import RobustQIteration


class RobustBaselineTests(unittest.TestCase):
    def setUp(self):
        self.env = Env((0, 0), (2, 2), set(), (3, 3))
        self.human_goal = (0, 2)

    def test_human_probabilities_match_original_rule(self):
        model = HumanTransitionModel(self.env, self.human_goal)
        transitions = model.distribution((2, 0), None, 0.1)
        probabilities = {(h, b): p for h, b, p in transitions}

        self.assertAlmostEqual(sum(probabilities.values()), 1.0)
        self.assertAlmostEqual(probabilities[((2, 0), None)], 0.1)
        self.assertAlmostEqual(probabilities[((1, 0), (2, 0))], 0.45)
        self.assertAlmostEqual(probabilities[((2, 1), (2, 0))], 0.45)

    def test_zero_width_interval_matches_nominal(self):
        planner = RobustQIteration(
            self.env,
            robot_goal=(2, 2),
            human_goal=self.human_goal,
            gamma=0.9,
        )
        nominal_q, nominal_policy, _ = planner.solve_nominal(0.1, tolerance=1e-7)
        (
            lower_q,
            upper_q,
            robust_policy,
            _,
            _,
            _,
            _,
        ) = planner.solve_robust(0.1, 0.1, tolerance=1e-7)

        state = ((0, 0), (2, 0), None)
        self.assertEqual(nominal_policy[state], robust_policy[state])
        for action in nominal_q[state]:
            self.assertAlmostEqual(nominal_q[state][action], lower_q[state][action], places=6)
            self.assertAlmostEqual(lower_q[state][action], upper_q[state][action], places=6)

    def test_upper_envelope_is_not_below_lower_envelope(self):
        planner = RobustQIteration(
            self.env,
            robot_goal=(2, 2),
            human_goal=self.human_goal,
            gamma=0.9,
        )
        lower_q, upper_q, _, _, bounds, _, _ = planner.solve_robust(
            0.07, 0.13, tolerance=1e-7
        )

        for state, action_values in lower_q.items():
            for action, lower_value in action_values.items():
                self.assertGreaterEqual(upper_q[state][action] + 1e-10, lower_value)
                self.assertAlmostEqual(
                    bounds[(state, action)]["q_width"],
                    upper_q[state][action] - lower_value,
                    places=8,
                )

    def test_ties_are_reported_as_multiple_optimal_actions(self):
        planner = RobustQIteration(
            self.env,
            robot_goal=(2, 2),
            human_goal=self.human_goal,
        )
        state = ((0, 0), (2, 0), None)
        q_values = planner._empty_q()
        actions = list(q_values[state])
        q_values[state][actions[0]] = 1.0
        q_values[state][actions[1]] = 1.0

        optimal = planner.optimal_actions(q_values, state)
        self.assertIn(actions[0], optimal)
        self.assertIn(actions[1], optimal)
        self.assertEqual(planner.action_margin(q_values, state), 0.0)


if __name__ == "__main__":
    unittest.main()
