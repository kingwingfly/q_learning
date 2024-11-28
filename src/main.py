from pprint import pprint as print
from random import uniform, choice


class QLearning:
    def __init__(self):
        self._people = 25
        self._next_people = self._people
        self._q_table: dict[tuple[int, float], dict[str, float]] = dict(
            (
                ((i, round(0.1 * j, 1)), {"up": 0, "down": 0})
                for i in range(self._people + 1)
                for j in range(0, 11)
            )
        )
        self._learning_rate = 0.1
        self._discount_factor = 0.9
        self._state = "waiting"
        self._next_state = "fighting"
        self._time = 0
        self._process = 0
        self._next_process = 0

    @property
    def time(self) -> int:
        return self._time

    def _max_q(self) -> float:
        """
        Get the max q_value of the next state
        """
        return max(self._q_table[(self._next_people, self._next_process)].values())

    def _select_action(self) -> str:
        if any((v == 0 for v in self._q_table[(self._people, self._process)].values())):
            return choice(
                [
                    k
                    for (k, v) in self._q_table[(self._people, self._process)].items()
                    if v == 0
                ]
            )
        total = sum(
            (-1 / v for v in self._q_table[(self._people, self._process)].values())
        )
        rand = uniform(0, total)
        tmp = 0
        for k, v in self._q_table[(self._people, self._process)].items():
            if tmp - 1 / v > rand:
                return k
            tmp -= 1 / v
        raise RuntimeError("Unreachable code reached")

    def _step(self, action: str):
        self._time += 1

        # update state
        if action == "up":
            self._next_state = "fighting"
        elif action == "down":
            self._next_state = "waiting"
        else:
            raise ValueError(f"Invalid action {action}")

        if self._state == "waiting":
            self._next_process = 0
            self._next_people = max(self._people - 8, 0)
        elif self._state == "fighting":
            self._next_process = round(self._process + 1 / (self._people + 1), 1)
            if self._next_process >= 1:
                self._state = "final"
                return
            self._next_people = max(self._people - 2, 0)

        r = -self._time

        self._q_table[(self._people, self._process)][action] *= 1 - self._learning_rate
        self._q_table[(self._people, self._process)][action] += self._learning_rate * (
            r + self._discount_factor * self._max_q()
        )

        self._state = self._next_state
        self._process = self._next_process
        self._people = self._next_people

    def _reset(self):
        self._time = 0
        self._state = "waiting"
        self._people = 25
        self._next_people = self._people
        self._process = 0
        self._next_process = 0

    def train(self):
        for i in range(100):
            print(f"epoch {i}")
            while True:
                action = self._select_action()
                self._step(action)
                if self._state == "final":
                    print(f"time cost {self._time}")
                    break
                print(action)
            self._reset()


def main():
    ql = QLearning()
    ql.train()


if __name__ == "__main__":
    main()
