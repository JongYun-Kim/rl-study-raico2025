# policy_value_iter_algos.py
"""
RL Basics: Policy Iteration vs Value Iteration
Compare the two algorithms in a GridWorld environment.

Note on goal state's value V(G):
- The +1 reward is granted upon ENTERING the goal cell.
- The goal cell itself is terminal, so there are no future rewards from it.
- Therefore, by convention, V(G) = 0. The +1 is already collected on the transition into G.
"""

import numpy as np
from typing import Tuple, Dict
import time


# Environment: GridWorld
class GridWorld:
    """
    Simple deterministic GridWorld.

    In a 4x4 grid:
    - S: start position (0, 0)
    - G: goal position (3, 3) — entering G yields +1 reward
    - X: obstacle — cannot enter
    - .: normal cell — step cost -0.04 to encourage shorter paths

    Actions: up(0), down(1), left(2), right(3)

    Important: The goal state is terminal. The +1 reward is received when you MOVE INTO the goal cell.
    Once you are in the goal cell, there are no further rewards or transitions. Hence the value of the
    goal state itself is V(G) = 0; the reward was already collected upon entry.
    """

    def __init__(self, size=4):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4  # up, down, left, right

        # Special positions
        self.start_state = (0, 0)
        self.goal_state = (3, 3)
        self.obstacles = [(1, 1), (2, 1)]  # obstacle positions

        # Action mapping
        self.actions = {
            0: (-1, 0),  # up
            1: (1, 0),  # down
            2: (0, -1),  # left
            3: (0, 1)  # right
        }

        self.action_names = {0: '↑', 1: '↓', 2: '←', 3: '→'}

    def state_to_pos(self, state: int) -> Tuple[int, int]:
        """Convert state index to (row, col)."""
        return (state // self.size, state % self.size)

    def pos_to_state(self, pos: Tuple[int, int]) -> int:
        """Convert (row, col) to state index."""
        return pos[0] * self.size + pos[1]

    def is_valid_pos(self, pos: Tuple[int, int]) -> bool:
        """Check if a position is valid (in bounds and not an obstacle)."""
        row, col = pos
        if row < 0 or row >= self.size or col < 0 or col >= self.size:
            return False
        if pos in self.obstacles:
            return False
        return True

    def is_terminal(self, state: int) -> bool:
        """Check if a state is terminal."""
        return self.state_to_pos(state) == self.goal_state

    def get_next_state(self, state: int, action: int) -> Tuple[int, float]:
        """
        Transition function (deterministic).

        Returns:
            next_state: next state index
            reward: reward received
        """
        if self.is_terminal(state):
            return state, 0

        current_pos = self.state_to_pos(state)
        action_delta = self.actions[action]
        next_pos = (current_pos[0] + action_delta[0],
                    current_pos[1] + action_delta[1])

        # Invalid moves keep you in place
        if not self.is_valid_pos(next_pos):
            next_pos = current_pos

        next_state = self.pos_to_state(next_pos)

        # Reward
        if next_pos == self.goal_state:
            reward = 1.0
        else:
            reward = -0.04  # step penalty

        return next_state, reward

    def print_grid(self, values=None, policy=None):
        """Visualize the grid."""
        print("\n" + "=" * 50)
        for row in range(self.size):
            for col in range(self.size):
                pos = (row, col)
                state = self.pos_to_state(pos)

                if pos == self.goal_state:
                    print("  G  ", end="")
                elif pos in self.obstacles:
                    print("  X  ", end="")
                elif policy is not None:
                    action = policy[state]
                    print(f"  {self.action_names[action]}  ", end="")
                elif values is not None:
                    print(f"{values[state]:5.2f}", end="")
                else:
                    print("  .  ", end="")
            print()
        print("=" * 50)


# Policy Iteration
class PolicyIteration:
    """
    Policy Iteration

    Steps:
    1. Policy Evaluation: Evaluate the current policy until convergence.
    2. Policy Improvement: Improve the policy greedily using the current value function.
    3. Stop if the policy no longer changes.
    """

    def __init__(self, env: GridWorld, gamma=0.9, theta=0.001):
        self.env = env
        self.gamma = gamma  # discount factor
        self.theta = theta  # convergence threshold

        # Initial policy: all "right"
        self.policy = np.zeros(env.n_states, dtype=int)
        self.policy[:] = 3  # right

        # Value function
        self.V = np.zeros(env.n_states)

        self.iteration_count = 0

    def policy_evaluation(self):
        """Evaluate the current policy to get its value function."""
        iteration = 0
        while True:
            delta = 0
            new_V = np.copy(self.V)

            for state in range(self.env.n_states):
                if self.env.is_terminal(state):
                    continue

                # Action under current policy
                action = self.policy[state]
                next_state, reward = self.env.get_next_state(state, action)

                # Bellman equation for policy evaluation
                new_V[state] = reward + self.gamma * self.V[next_state]

                delta = max(delta, abs(new_V[state] - self.V[state]))

            self.V = new_V
            iteration += 1

            # Convergence check
            if delta < self.theta:
                print(f"    Policy Evaluation converged in {iteration} iterations")
                break

    def policy_improvement(self) -> bool:
        """
        Improve the policy greedily with respect to current value function.

        Returns:
            policy_stable: True if policy did not change.
        """
        policy_stable = True

        for state in range(self.env.n_states):
            if self.env.is_terminal(state):
                continue

            old_action = self.policy[state]

            # Compute Q-values for all actions
            action_values = np.zeros(self.env.n_actions)
            for action in range(self.env.n_actions):
                next_state, reward = self.env.get_next_state(state, action)
                action_values[action] = reward + self.gamma * self.V[next_state]

            # Greedy policy
            self.policy[state] = np.argmax(action_values)

            if old_action != self.policy[state]:
                policy_stable = False

        return policy_stable

    def run(self, max_iterations=100):
        """Run Policy Iteration."""
        print("\n" + "🔄 Start Policy Iteration".center(50, "="))
        start_time = time.time()

        for i in range(max_iterations):
            self.iteration_count = i + 1
            print(f"\n[Iteration {self.iteration_count}]")

            # Step 1: Policy Evaluation
            self.policy_evaluation()

            # Step 2: Policy Improvement
            policy_stable = self.policy_improvement()

            if policy_stable:
                print(f"\n✅ Policy converged! (total {self.iteration_count} iterations)")
                break

        elapsed_time = time.time() - start_time
        print(f"⏱️  Elapsed time: {elapsed_time:.4f}s")

        return self.V, self.policy


# Value Iteration
class ValueIteration:
    """
    Value Iteration

    Steps:
    1. For each state, apply the Bellman optimality equation to update V.
    2. Stop when V converges.
    3. Extract the optimal policy from the converged V.
    """

    def __init__(self, env: GridWorld, gamma=0.9, theta=0.001):
        self.env = env
        self.gamma = gamma
        self.theta = theta

        # Value function
        self.V = np.zeros(env.n_states)

        self.iteration_count = 0

    def run(self, max_iterations=100):
        """Run Value Iteration."""
        print("\n" + "🎯 Start Value Iteration".center(50, "="))
        start_time = time.time()

        for i in range(max_iterations):
            self.iteration_count = i + 1
            delta = 0
            new_V = np.copy(self.V)

            for state in range(self.env.n_states):
                if self.env.is_terminal(state):
                    continue

                # Compute Q-values for all actions
                action_values = np.zeros(self.env.n_actions)
                for action in range(self.env.n_actions):
                    next_state, reward = self.env.get_next_state(state, action)
                    action_values[action] = reward + self.gamma * self.V[next_state]

                # Bellman optimality update
                new_V[state] = np.max(action_values)

                delta = max(delta, abs(new_V[state] - self.V[state]))

            self.V = new_V

            print(f"[Iteration {self.iteration_count}] Delta: {delta:.6f}")

            # Convergence check
            if delta < self.theta:
                print(f"\n✅ Value function converged! (total {self.iteration_count} iterations)")
                break

        elapsed_time = time.time() - start_time
        print(f"⏱️  Elapsed time: {elapsed_time:.4f}s")

        # Extract optimal policy
        policy = self.extract_policy()

        return self.V, policy

    def extract_policy(self):
        """Extract the optimal policy from the current value function."""
        policy = np.zeros(self.env.n_states, dtype=int)

        for state in range(self.env.n_states):
            if self.env.is_terminal(state):
                continue

            action_values = np.zeros(self.env.n_actions)
            for action in range(self.env.n_actions):
                next_state, reward = self.env.get_next_state(state, action)
                action_values[action] = reward + self.gamma * self.V[next_state]

            policy[state] = np.argmax(action_values)

        return policy


# Run and Compare
def compare_algorithms():
    """Run both algorithms and compare the results."""

    print("\n" + "🎮 GridWorld RL Demo".center(60, "="))
    print("\nEnvironment:")
    print("  - 4x4 grid")
    print("  - S (0,0): start position")
    print("  - G (3,3): goal (+1 reward upon entry)")
    print("  - X: obstacle")
    print("  - Step penalty: -0.04 (to encourage shorter paths)")
    print("\nNote: The goal state is terminal, so V(G) = 0. The +1 reward is received upon entering G.")

    # Create environment
    env = GridWorld(size=4)

    print("\nInitial grid:")
    env.print_grid()

    # Policy Iteration
    pi = PolicyIteration(env, gamma=0.9, theta=0.001)
    V_pi, policy_pi = pi.run()

    print("\n📊 Policy Iteration Results:")
    print("\nValue Function:")
    env.print_grid(values=V_pi)
    print("\nOptimal Policy:")
    env.print_grid(policy=policy_pi)

    # Value Iteration
    vi = ValueIteration(env, gamma=0.9, theta=0.001)
    V_vi, policy_vi = vi.run()

    print("\n📊 Value Iteration Results:")
    print("\nValue Function:")
    env.print_grid(values=V_vi)
    print("\nOptimal Policy:")
    env.print_grid(policy=policy_vi)

    # Compare
    print("\n" + "📈 Algorithm Comparison".center(60, "="))
    print(f"Policy Iteration:  {pi.iteration_count} iterations")
    print(f"Value Iteration:   {vi.iteration_count} iterations")
    print(f"\nValue Function L2 Difference: {np.linalg.norm(V_pi - V_vi):.6f}")
    print(f"Policies identical: {np.array_equal(policy_pi, policy_vi)}")

    # Path simulation
    print("\n" + "🚶 Optimal Path Simulation (using PI policy)".center(60, "="))
    state = env.pos_to_state(env.start_state)
    path = [env.start_state]

    print(f"Start: {env.start_state}")
    for step in range(20):  # max 20 steps
        if env.is_terminal(state):
            break
        action = policy_pi[state]
        next_state, reward = env.get_next_state(state, action)
        next_pos = env.state_to_pos(next_state)
        path.append(next_pos)
        print(f"Step {step + 1}: {next_pos} (action: {env.action_names[action]}, reward: {reward:.2f})")
        state = next_state

    print(f"\nReached goal! Total steps: {len(path) - 1}")


# Clarification
def _print_note():
    print("\n" + "💡 Key Takeaways".center(60, "="))
    print("""
    Policy Iteration:
      - Fully evaluates the current policy, then improves it (policy evaluation + improvement).
      - Typically converges in fewer outer iterations.
      - Each iteration can be more expensive.

    Value Iteration:
      - Repeatedly applies the Bellman optimality update to V.
      - Often needs more iterations.
      - Each iteration is cheaper.
      - Equivalent to doing a single sweep of policy evaluation at each step.

    Both algorithms converge to the optimal policy in this deterministic GridWorld.

    Note on V(G):
      - The +1 reward is received when transitioning into the goal.
      - The goal state itself is terminal with no further returns.
      - Therefore, V(G) = 0 by convention; the reward was already collected upon entry.
    """)

# Main
if __name__ == "__main__":
    compare_algorithms()
    _print_note()
 
