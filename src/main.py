from pprint import pprint as print


class QLearning:
    def __init__(self):
        self._q_table: dict[str, dict[str, float]] = {
            "waiting": {"up": 0, "down": 0},
            "fighting": {"up": 0, "down": 0},
            "final": {"up": 0, "down": 0},
        }
        self._learning_rate = 0.1
        self._discount_factor = 0.9
        self._state = "waiting"
        self._next_state = "fighting"
        self._time = 0
        self._people = 25
        self._process = 0

    @property
    def time(self) -> int:
        return self._time

    def _max_q(self) -> float:
        """
        Get the max q_value of the next state
        """
        return max(self._q_table[self._next_state].values())

    def _select_action(self) -> str:
        return max(
            self._q_table[self._state].keys(),
            key=lambda x: self._q_table[self._state][x],
        )

    def _step(self, action: str):
        self._time += 1

        # update state
        if action == "up":
            self._next_state = "fighting"
        elif action == "down":
            self._next_state = "waiting"
            self._process = 0
        else:
            raise ValueError(f"Invalid action {action}")

        # update process
        if self._state == "fighting":
            self._process += 1 / (self._people + 1)  # avoid zero division
            if self._process >= 0.2:
                self._state = "final"
                return

        # update people
        if self._state == "waiting":
            self._people = max(self._people - 5, 0)
        elif self._state == "fighting":
            self._people = max(self._people - 1, 0)

        r = -self._time

        self._q_table[self._state][action] = (1 - self._learning_rate) * self._q_table[
            self._state
        ][action] + self._learning_rate * (r + self._discount_factor * self._max_q())

        self._state = self._next_state

    def _reset(self):
        self._time = 0
        self._state = "waiting"
        self._people = 25
        self._process = 0

    def train(self):
        for i in range(100):
            print(f"epoch {i}")
            while True:
                if self._state == "final":
                    print(f"time cost {self._time}")
                    break
                action = self._select_action()
                print(action)
                self._step(action)
            self._reset()


def main():
    ql = QLearning()
    ql.train()


if __name__ == "__main__":
    main()
