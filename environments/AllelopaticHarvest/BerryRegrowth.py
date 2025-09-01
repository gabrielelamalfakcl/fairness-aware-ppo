import random

class RegrowthFunction:
    def regrowth(self):
        raise NotImplementedError("This method should be overridden by subclasses")

class LinearRegrowth(RegrowthFunction):
    def regrowth(self, red_count, blue_count):
        """
        Linear regrowth function based on the proportion of berries.
        """
        total_count = red_count + blue_count
        if total_count == 0:
            return random.choice(['red', 'blue'])
        red_probability = red_count / total_count
        return 'red' if random.random() < red_probability else 'blue'

class CubicRegrowth(RegrowthFunction):
    def regrowth(self, red_count, blue_count):
        """
        Cubic regrowth function based on the proportion of berries.
        """
        total_count = red_count + blue_count
        if total_count == 0:
            return random.choice(['red', 'blue'])
        red_probability = (red_count / total_count) ** 3
        blue_probability = (blue_count / total_count) ** 3
        total_probability = red_probability + blue_probability
        red_probability /= total_probability
        return 'red' if random.random() < red_probability else 'blue'