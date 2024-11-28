from pprint import pprint as print


class QLearning:
    def __init__(self):
        self.q_table: dict[str, dict[str, float]] = {
            "waiting": {"up": 0, "down": 0},
            "fighting": {"up": 0, "down": 0},
            "final": {"up": 0, "down": 0},
        }
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.state = "waiting"
        self.next_state = "fighting"
        self.time = 0
        self.people = 25
        self.process = 0

    def max_q(self) -> float:
        """
        Get the max q_value of the next state
        """
        return max(self.q_table[self.next_state].values())

    def select_action(self) -> str:
        return max(
            self.q_table[self.state].keys(), key=lambda x: self.q_table[self.state][x]
        )

    def step(self, action: str):
        self.time += 1

        if action == "up":
            self.next_state = "fighting"
        elif action == "down":
            self.next_state = "waiting"

        if self.next_state == "waitng":
            self.process = 0

        if self.state == "fighting":
            self.process += 1 / (self.people + 1) # avoid zero division

        if self.state == "waiting":
            self.people = max(self.people - 5 ,0)
        elif self.state == "fighting":
            self.people = max(self.people - 1 ,0)

        if self.process >= 0.2:
            self.next_state = "final"

        r = -self.time

        self.q_table[self.state][action] = (1 - self.learning_rate) * self.q_table[
            self.state
        ][action] + self.learning_rate * (r + self.discount_factor * self.max_q())

        self.state = self.next_state

    def train(self):
        for i in range(100):
            print(f"epoch {i}")
            while True:
                if self.state == "final":
                    print(self.time)
                    break
                action = self.select_action()
                print(action)
                self.step(action)
            self.time = 0
            self.state = "waiting"
            self.people = 25
            self.process = 0


def main():
    ql = QLearning()
    ql.train()


if __name__ == "__main__":
    main()
