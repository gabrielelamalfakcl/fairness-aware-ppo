import random

class MovementPolicy:
    def move(self, player, environment):
        raise NotImplementedError("This method should be overridden by subclasses")

class RandomMovementPolicy(MovementPolicy):
    def move(self, player, environment):
        directions = ["up", "down", "left", "right"]
        random.shuffle(directions)  # Shuffle directions to randomize movement
        for direction in directions:
            old_x, old_y = player.x, player.y
            player.move(environment, direction)
            if (old_x, old_y) != (player.x, player.y):
                break  # Move was successful

class StationaryPolicy(MovementPolicy):
    def move(self, player, environment):
        # Player doesn't move
        pass
