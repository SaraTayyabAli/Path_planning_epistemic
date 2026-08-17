from human_transition import HumanTransitionModel


class RobustQIteration:
    """Statewise maximin Bellman solver for one human and interval-valued zeta."""

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
        states = []
        for human in self.robot_states:
            states.append((human, None))
            for neighbor in self.human_model.neighbors(human):
                if neighbor != human:
                    states.append((human, neighbor))
        return tuple(states)

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
        q = {}
        for state in self.states:
            robot = state[0]
            if robot == self.robot_goal:
                q[state] = {}
            else:
                q[state] = {action: 0.0 for action in self.legal_actions[robot]}
        return q

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
        return self._solve(
            zetas=(zeta,),
            robust=False,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )

    def solve_robust(self, zeta_low, zeta_high, tolerance=1e-6, max_iterations=500):
        if zeta_low > zeta_high:
            raise ValueError("zeta_low must be smaller than or equal to zeta_high")

        return self._solve(
            zetas=(zeta_low, zeta_high),
            robust=True,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )

    def _solve(self, zetas, robust, tolerance, max_iterations):
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

                    # The human probabilities are affine in zeta, so the interval
                    # minimum is attained at one of its endpoints.
                    target = min(endpoint_values) if robust else endpoint_values[0]
                    new_q[state][action] = target
                    delta = max(delta, abs(target - old_value))

            q_values = new_q
            if delta < tolerance:
                break
        else:
            raise RuntimeError("Robust Q iteration did not converge")

        policy = {
            state: max(action_values, key=action_values.get)
            for state, action_values in q_values.items()
            if action_values
        }

        bounds = {}
        if robust:
            for state, action_values in q_values.items():
                for action in action_values:
                    low_endpoint = self.expected_target(
                        state, action, q_values, zetas[0]
                    )
                    high_endpoint = self.expected_target(
                        state, action, q_values, zetas[-1]
                    )
                    bounds[(state, action)] = {
                        "zeta_low_value": low_endpoint,
                        "zeta_high_value": high_endpoint,
                        "lower_q": min(low_endpoint, high_endpoint),
                        "upper_q": max(low_endpoint, high_endpoint),
                        "q_width": abs(high_endpoint - low_endpoint),
                    }

        return q_values, policy, bounds, iteration
