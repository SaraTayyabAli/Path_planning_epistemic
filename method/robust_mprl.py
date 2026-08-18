from human_transition import HumanTransitionModel


class RobustQIteration:
    """Bellman solver for one human and interval-valued zeta."""

    def __init__(
        self,
        env,
        robot_goal,
        human_goal,
        gamma=0.9,
        step_reward=-0.1,
        goal_reward=10.0,
        collision_penalty=-2.0,
    ):
        self.env = env
        self.robot_goal = robot_goal
        self.human_goal = human_goal
        self.gamma = gamma
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.collision_penalty = collision_penalty
        self.human_model = HumanTransitionModel(env, human_goal)

        self.robot_states = tuple(sorted(env.able_state))
        self.legal_actions = {
            state: tuple(
                action
                for action in range(env.n_actions)
                if env.get_state_action_space(state, action) in env.able_state
            )
            for state in self.robot_states
        }

        self.human_memory_states = self._build_human_memory_states()
        self.states = tuple(
            (robot, human, back_point)
            for robot in self.robot_states
            for human, back_point in self.human_memory_states
        )
        self._human_transition_cache = {}

    def _build_human_memory_states(self):
        states = set()
        for human in self.robot_states:
            states.add((human, None))
            states.add((human, human))
            for neighbor in self.human_model.neighbors(human):
                if neighbor != human:
                    states.add((human, neighbor))
        return tuple(sorted(states, key=lambda item: (item[0], str(item[1]))))

    def _human_transitions(self, human, back_point, zeta):
        key = (human, back_point, zeta)
        if key not in self._human_transition_cache:
            self._human_transition_cache[key] = self.human_model.distribution(
                human, back_point, zeta
            )
        return self._human_transition_cache[key]

    def _reward(self, robot, human, robot_next, human_next):
        reward = self.goal_reward if robot_next == self.robot_goal else self.step_reward

        same_cell = robot_next == human_next
        edge_swap = robot_next == human and human_next == robot
        if same_cell or edge_swap:
            reward += self.collision_penalty

        return reward

    def _empty_q(self):
        q_values = {}
        for state in self.states:
            robot = state[0]
            if robot == self.robot_goal:
                q_values[state] = {}
            else:
                q_values[state] = {
                    action: 0.0 for action in self.legal_actions[robot]
                }
        return q_values

    def expected_target(self, state, action, q_values, zeta):
        robot, human, back_point = state
        robot_next = self.env.get_state_action_space(robot, action)
        expected = 0.0

        for human_next, back_next, probability in self._human_transitions(
            human, back_point, zeta
        ):
            reward = self._reward(robot, human, robot_next, human_next)

            if robot_next == self.robot_goal:
                continuation = 0.0
            else:
                next_q = q_values[(robot_next, human_next, back_next)]
                continuation = max(next_q.values()) if next_q else 0.0

            expected += probability * (reward + self.gamma * continuation)

        return expected

    def solve_nominal(self, zeta, tolerance=1e-6, max_iterations=500):
        q_values, iterations = self._solve_envelope(
            zetas=(zeta,),
            envelope="nominal",
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
        policy = self.greedy_policy(q_values)
        return q_values, policy, iterations

    def solve_robust(
        self,
        zeta_low,
        zeta_high,
        tolerance=1e-6,
        max_iterations=500,
        tie_tolerance=1e-10,
    ):
        if zeta_low > zeta_high:
            raise ValueError("zeta_low must be smaller than or equal to zeta_high")

        zetas = (zeta_low, zeta_high)

        lower_q, lower_iterations = self._solve_envelope(
            zetas=zetas,
            envelope="lower",
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
        upper_q, upper_iterations = self._solve_envelope(
            zetas=zetas,
            envelope="upper",
            tolerance=tolerance,
            max_iterations=max_iterations,
        )

        robust_policy = self.greedy_policy(lower_q)
        optimal_actions = {
            state: self.optimal_actions(lower_q, state, tie_tolerance)
            for state in robust_policy
        }
        bounds = self._build_bounds(lower_q, upper_q, zeta_low, zeta_high)

        return (
            lower_q,
            upper_q,
            robust_policy,
            optimal_actions,
            bounds,
            lower_iterations,
            upper_iterations,
        )

    def _solve_envelope(self, zetas, envelope, tolerance, max_iterations):
        q_values = self._empty_q()

        for iteration in range(1, max_iterations + 1):
            new_q = {state: values.copy() for state, values in q_values.items()}
            delta = 0.0

            for state, action_values in q_values.items():
                for action, old_value in action_values.items():
                    endpoint_values = [
                        self.expected_target(state, action, q_values, zeta)
                        for zeta in zetas
                    ]

                    if envelope == "lower":
                        target = min(endpoint_values)
                    elif envelope == "upper":
                        target = max(endpoint_values)
                    else:
                        target = endpoint_values[0]

                    new_q[state][action] = target
                    delta = max(delta, abs(target - old_value))

            q_values = new_q
            if delta < tolerance:
                return q_values, iteration

        raise RuntimeError(f"{envelope.capitalize()} Q iteration did not converge")

    def greedy_policy(self, q_values):
        return {
            state: max(action_values, key=action_values.get)
            for state, action_values in q_values.items()
            if action_values
        }

    @staticmethod
    def optimal_actions(q_values, state, tolerance=1e-10):
        action_values = q_values[state]
        if not action_values:
            return tuple()

        best_value = max(action_values.values())
        return tuple(
            action
            for action, value in action_values.items()
            if best_value - value <= tolerance
        )

    @staticmethod
    def action_margin(q_values, state):
        values = sorted(q_values[state].values(), reverse=True)
        if len(values) < 2:
            return 0.0
        return values[0] - values[1]

    def _build_bounds(self, lower_q, upper_q, zeta_low, zeta_high):
        bounds = {}

        for state, action_values in lower_q.items():
            for action in action_values:
                low_backup = self.expected_target(
                    state, action, lower_q, zeta_low
                )
                high_backup = self.expected_target(
                    state, action, lower_q, zeta_high
                )

                lower_value = lower_q[state][action]
                upper_value = upper_q[state][action]
                if upper_value < lower_value and abs(upper_value - lower_value) < 1e-10:
                    upper_value = lower_value

                worst_case_zeta = zeta_low if low_backup <= high_backup else zeta_high

                bounds[(state, action)] = {
                    "zeta_low_lower_backup": low_backup,
                    "zeta_high_lower_backup": high_backup,
                    "worst_case_zeta": worst_case_zeta,
                    "lower_q": lower_value,
                    "upper_q": upper_value,
                    "q_width": upper_value - lower_value,
                    "endpoint_span": abs(high_backup - low_backup),
                }

        return bounds
