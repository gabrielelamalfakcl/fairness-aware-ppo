import random
import numpy as np
from .BerryRegrowth import LinearRegrowth
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_base_dir = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(project_base_dir)
# from AgentsPolicy import RandomMovementPolicy, StationaryPolicy
# from BerryRegrowth import LinearRegrowth, CubicRegrowth
from collections import deque

class Player:
    class Preference:
        def __init__(self, berry_type):
            self.berry_type = berry_type

        def get_preference(self):
            return self.berry_type

    def __init__(self, name, x, y, policy, preference_berry_type, has_disability=False, verbose=False):
        self.name = name
        self.x = x
        self.y = y
        self.reward = 0
        self.policy = policy
        self.preference = self.Preference(preference_berry_type)
        self.has_disability = has_disability
        self.policy_counter = 0
        self.is_obstacolated = False
        self.obstacolation_cooldown = 0
        self.state_history = []
        self.move_counter = 0
        self.last_action_position = None
        self.verbose = verbose

    def move(self, environment, direction):
        """
        Moves the agent in the specified direction within the environment.
        """
        possible_moves = {
            "up": (0, -1),
            "down": (0, 1),
            "left": (-1, 0),
            "right": (1, 0)
        }
        
        if direction not in possible_moves:
            return False
        
        dx, dy = possible_moves[direction]
        new_x, new_y = self.x + dx, self.y + dy
        
        if 0 <= new_x < environment.x_dim and 0 <= new_y < environment.y_dim:
            if environment.grid[new_x][new_y] is None:
                environment.grid[self.x][self.y] = None
                self.x, self.y = new_x, new_y
                environment.grid[self.x][self.y] = self
                self.reward = 0
                return True
        else:
            return False
        
    def ripe_bush(self, environment):
        """
        Finds an adjacent bush and, if it is not ripe, ripens it.
        """
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for dx, dy in directions:
            target_x, target_y = self.x + dx, self.y + dy
            if 0 <= target_x < environment.x_dim and 0 <= target_y < environment.y_dim:
                target_entity = environment.grid[target_x][target_y]
                if isinstance(target_entity, Bush):
                    # Only ripen if it's not already ripe.
                    if not target_entity.is_ripe:
                        self.reward = self.ripe(target_entity)
                    else:
                        self.reward = 0 # No reward for trying to ripen a ripe bush
                    break
        return self.reward

    def eat_bush(self, environment):
        """
        Finds an adjacent bush and, if it is ripe, eats from it.
        """
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for dx, dy in directions:
            target_x, target_y = self.x + dx, self.y + dy
            if 0 <= target_x < environment.x_dim and 0 <= target_y < environment.y_dim:
                target_entity = environment.grid[target_x][target_y]
                if isinstance(target_entity, Bush):
                    self.eat_fruit(target_entity)
                    break
        return self.reward
                      
    def change_bush_color(self, environment):
        """
        Change the color of a bush in the adjacent cells if bush is not of color preference.
        """
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for dx, dy in directions:
            target_x, target_y = self.x + dx, self.y + dy
            if 0 <= target_x < environment.x_dim and 0 <= target_y < environment.y_dim:
                target_entity = environment.grid[target_x][target_y]
                if isinstance(target_entity, Bush) and target_entity.berry_type != self.preference.get_preference():
                    self.reward = self.change_color(target_entity)
                    break
        
        return self.reward
                    
    def interact_with_nearby_player(self, environment):
        """
        Interact with a player in the adjacent cells and obstruct them.
        """
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for dx, dy in directions:
            target_x, target_y = self.x + dx, self.y + dy
            if 0 <= target_x < environment.x_dim and 0 <= target_y < environment.y_dim:
                target_entity = environment.grid[target_x][target_y]
                if isinstance(target_entity, Player):
                    self.reward = self.obstacolate_player(target_entity)
                    break
                
        return self.reward
                
    def eat_fruit(self, bush):
        """
        Eat a berry from a bush only if it is ripe.
        """
        if bush.is_ripe and bush.berry_type is not None:
            if bush.berry_type == self.preference.get_preference():
                self.reward = 5
            else:
                self.reward = 1
            
            # Reset the berry once it is eaten
            bush.current_berry_type = None
            bush.is_ripe = False
            bush.time_step = 0
        else:
            # If bush is not ripe or has no berry, action fails.
            self.reward = 0
        
        return self.reward
                
    def ripe(self, bush):
        """
        Ripen a berry on a bush.
        """
        if bush.is_ripe:
            self.reward = 0
        else:
            if bush.berry_type == self.preference.get_preference():
                self.reward = 3
            else:
                self.reward = 1
            # Bush becomes ripe
            bush.is_ripe = True
        
        return self.reward
        
    def change_color(self, bush, new_color=None):
        """
        Changes the color of the berry currently on the bush.
        """
        old_color = bush.berry_type
        if new_color is None:
            new_color = self.preference.get_preference()

        bush.berry_type = new_color
        bush.is_ripe = False
        bush.time_step = 0
        
        if old_color != self.preference.get_preference():
            self.reward = 5
        else:
            self.reward = 0
        
        return self.reward
        
    def obstacolate_player(self, other_player):
        """
        Obstacolate a player.
        """
        other_player.is_obstacolated = True
        self.obstacolation_cooldown = 1
        other_player.obstacolation_cooldown = 1
        if other_player.preference.get_preference() != self.preference.get_preference():
            self.reward = 3
        else:
            self.reward = 0
        
        return self.reward
        
    def get_nearest_bush_distance(self, environment, bush_type):
        distances = [
            abs(self.x - bush.x) + abs(self.y - bush.y)
            for bush in environment.bushes if bush.berry_type == bush_type
        ]

        return min(distances) if distances else environment.x_dim + environment.y_dim
    
    def get_nearest_player_distance(self, environment):
        distances = [
            abs(self.x - player.x) + abs(self.y - player.y)
            for player in environment.players if player != self
        ]

        return min(distances) if distances else environment.x_dim + environment.y_dim

    def get_state(self, environment):
        """
        Returns the state of the player as a numpy array.
        """
        state = [self.x, self.y, self.has_disability, self.reward]
        preference_num = 1 if self.preference.get_preference() == "red" else 2
        state.append(preference_num)
        
        red_bush_count = sum(1 for bush in environment.bushes if bush.berry_type == 'red')
        blue_bush_count = sum(1 for bush in environment.bushes if bush.berry_type == 'blue')
        state.extend([red_bush_count, blue_bush_count])

        red_bush_distance = self.get_nearest_bush_distance(environment, 'red')
        blue_bush_distance = self.get_nearest_bush_distance(environment, 'blue')
        state.extend([red_bush_distance, blue_bush_distance])

        nearest_player_distance = self.get_nearest_player_distance(environment)
        state.append(nearest_player_distance)
        
        return np.array(state, dtype=np.float32)

    def can_interact_with_bush(self, environment):
        """
        Checks if there's a bush in an adjacent cell.
        """
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for dx, dy in directions:
            target_x, target_y = self.x + dx, self.y + dy
            if 0 <= target_x < environment.x_dim and 0 <= target_y < environment.y_dim:
                if isinstance(environment.grid[target_x][target_y], Bush):
                    return True
        return False

    def can_interact_with_player(self, environment):
        """
        Checks if there's a player in an adjacent cell.
        """
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for dx, dy in directions:
            target_x, target_y = self.x + dx, self.y + dy
            if 0 <= target_x < environment.x_dim and 0 <= target_y < environment.y_dim:
                if isinstance(environment.grid[target_x][target_y], Player):
                    return True
        return False

    def execute_policy_with_action(self, environment, action: int):
        # action space (9)
        num2action = {
            0: "stay", 1: "move_up", 2: "move_down", 3: "move_left", 4: "move_right",
            5: "eat_bush", 6: "change_bush_color", 7: "interact_with_nearby_player",
            8: "ripe_bush"
        }
        available_actions = list(num2action.keys())
        self.current_action = num2action.get(action, "unknown")
        self.reward = 0
        retry_limit = 3
        retries = 0
        self.move_counter += 1

        if self.is_obstacolated:
            self.obstacolation_cooldown -= 1
            return self.reward

        if self.has_disability and (self.move_counter % 2) != 0:
            return self.reward

        while retries < retry_limit:
            retries += 1
            if action == 0:
                return self.reward
            
            if action in [1, 2, 3, 4]:
                if self.move(environment, num2action[action]):
                    return self.reward
                else:
                    available_actions.remove(action)
                    action = random.choice(available_actions) if available_actions else 0
            
            elif action == 5:  # eat_bush
                if self.can_interact_with_bush(environment):
                    self.reward = self.eat_bush(environment)
                    return self.reward
                else:
                    available_actions.remove(action)
                    action = random.choice(available_actions) if available_actions else 0
            
            elif action == 6:  # change_bush_color
                if self.can_interact_with_bush(environment):
                    self.reward = self.change_bush_color(environment)
                    return self.reward
                else:
                    available_actions.remove(action)
                    action = random.choice(available_actions) if available_actions else 0
            
            elif action == 7:  # interact_with_nearby_player
                if self.can_interact_with_player(environment):
                    self.reward = self.interact_with_nearby_player(environment)
                    return self.reward
                else:
                    available_actions.remove(action)
                    action = random.choice(available_actions) if available_actions else 0
            
            elif action == 8: # ripe_bush
                if self.can_interact_with_bush(environment):
                    self.reward = self.ripe_bush(environment)
                    return self.reward
                else:
                    available_actions.remove(action)
                    action = random.choice(available_actions) if available_actions else 0

        return self.reward

        
class Bush:
    def __init__(self, x, y, berry_type, regrowth_rate, regrowth_function, max_lifespan, spont_growth_rate):
        self.x = x
        self.y = y
        self.berry_type = berry_type
        self.current_berry_type = None
        self.regrowth_rate = regrowth_rate
        self.time_step = 0
        self.is_ripe = False
        self.regrowth_function = regrowth_function
        self.max_lifespan = max_lifespan
        self.spont_growth_rate = spont_growth_rate
        self.lifespan = 0

    def update(self):
        self.lifespan += 1
        if self.lifespan >= self.max_lifespan:
            return False
        
        if self.current_berry_type is not None:
            self.time_step += 1
            if self.time_step >= self.regrowth_rate:
                self.current_berry_type = self.berry_type
                self.is_ripe = True
                self.time_step = 0
        
        return True


class Environment:
    def __init__(self, x_dim, y_dim, max_steps, num_players, num_bushes, 
                 red_player_percentage, blue_player_percentage, 
                 red_bush_percentage, blue_bush_percentage, 
                 disability_percentage, max_lifespan, regrowth_rate, spont_growth_rate, verbose=False, random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.max_steps = max_steps
        self.num_players = num_players
        self.num_bushes = num_bushes
        self.red_player_percentage = red_player_percentage
        self.blue_player_percentage = blue_player_percentage
        self.red_bush_percentage = red_bush_percentage
        self.blue_bush_percentage = blue_bush_percentage
        self.disability_percentage = disability_percentage
        self.grid = [[None for _ in range(y_dim)] for _ in range(x_dim)]
        self.players = []
        self.bushes = []
        self.current_step = 0
        self.last_action_position = None
        self.max_lifespan = max_lifespan
        self.regrowth_rate = regrowth_rate
        self.spont_growth_rate = spont_growth_rate
        self.verbose = verbose

    def add_player(self, name, x, y, policy, preference_berry_type, has_disability=False):
        player = Player(name, x, y, policy, preference_berry_type, has_disability)
        self.players.append(player)
        self.grid[x][y] = player

    def add_bush(self, x, y, berry_type, regrowth_rate, regrowth_function, max_lifespan, spont_growth_rate):
        bush = Bush(x, y, berry_type, regrowth_rate, regrowth_function, max_lifespan, spont_growth_rate)
        self.bushes.append(bush)
        self.grid[x][y] = bush

    def update_bushes(self):
        for bush in self.bushes[:]:
            if not bush.update():
                self.bush_lifespan_end(bush)

    def bush_spontaneous_growth(self, regrowth_rate, regrowth_function, max_lifespan, spont_growth_rate):
        red_bush_count = sum(1 for bush in self.bushes if bush.berry_type == 'red')
        blue_bush_count = sum(1 for bush in self.bushes if bush.berry_type == 'blue')
        
        if red_bush_count == 0 and blue_bush_count == 0:
            berry_type = None
        else:
            berry_type = regrowth_function(red_bush_count, blue_bush_count)
        
        max_attempts = 100
        for _ in range(max_attempts):
            x = random.randint(0, self.x_dim - 1)
            y = random.randint(0, self.y_dim - 1)
            if self.grid[x][y] is None:
                self.add_bush(x, y, berry_type, regrowth_rate, regrowth_function, max_lifespan, spont_growth_rate)
                break       

    def bush_lifespan_end(self, bush):
        self.grid[bush.x][bush.y] = None
        self.bushes.remove(bush)
        
    def global_features(self):
        if not self.bushes:
            return np.zeros(3, dtype=np.float32)

        total = len(self.bushes)
        ripe_count = sum(b.is_ripe for b in self.bushes)
        blue_count = sum(b.berry_type == "blue" for b in self.bushes)
        red_count = sum(b.berry_type == "red"  for b in self.bushes)

        ripe_ratio = ripe_count / total
        blue_ratio = blue_count / total
        red_ratio  = red_count  / total
        return np.array([ripe_ratio, blue_ratio, red_ratio], dtype=np.float32)

    def randomize_positions(self, regrowth_rate, max_lifespan, spont_growth_rate):
        available_positions = [(x, y) for x in range(self.x_dim) for y in range(self.y_dim)]
        random.shuffle(available_positions)
        
        num_red_players = int(self.num_players * self.red_player_percentage)
        num_players_with_disability = int(self.num_players * self.disability_percentage)
        
        player_indices = list(range(self.num_players))
        random.shuffle(player_indices)
        players_with_disability = set(player_indices[:num_players_with_disability])
        
        policy = None # RandomMovementPolicy()

        for i in range(num_red_players):
            if not available_positions:
                continue
            x, y = available_positions.pop()
            has_disability = i in players_with_disability
            self.add_player(f"Player{i+1}", x, y, policy, "red", has_disability)

        num_blue_players = self.num_players - num_red_players
        for i in range(num_blue_players):
            if not available_positions:
                continue
            x, y = available_positions.pop()
            has_disability = (i + num_red_players) in players_with_disability
            self.add_player(f"Player{i+num_red_players+1}", x, y, policy, "blue", has_disability)
        
        num_red_bushes = int(self.num_bushes * self.red_bush_percentage)
        num_blue_bushes = int(self.num_bushes * self.blue_bush_percentage)
        
        regrowth_function = None # LinearRegrowth().regrowth

        # Add red and blue bushes if positions are available
        for i in range(num_red_bushes):
            if not available_positions:
                print(f"Failed to place red bush {i+1}: No available positions.")
                continue
            x, y = available_positions.pop()
            regrowth_function = LinearRegrowth().regrowth
            self.add_bush(x, y, "red", regrowth_rate, regrowth_function, max_lifespan, spont_growth_rate)
            
        for i in range(num_blue_bushes):
            if not available_positions:
                print(f"Failed to place blue bush {i+1}: No available positions.")
                continue
            x, y = available_positions.pop()
            regrowth_function = LinearRegrowth().regrowth
            self.add_bush(x, y, "blue", regrowth_rate, regrowth_function, max_lifespan, spont_growth_rate)      
        
    def update_players(self):
        for player in self.players:
            if player.obstacolation_cooldown > 0:
                player.obstacolation_cooldown -= 1

    def reset(self):
        self.grid = [[None for _ in range(self.y_dim)] for _ in range(self.x_dim)]
        self.players = []
        self.bushes = []
        self.current_step = 0
        self.randomize_positions(self.regrowth_rate, self.max_lifespan, self.spont_growth_rate)
        
        if self.verbose:
            self.print_matrix()
        
        return self.get_state(self)

    def get_state(self, environment):
        states = [player.get_state(environment) for player in self.players]
        return np.vstack(states)

    def step(self, actions, environment, regrowth_rate, regrowth_function, max_lifespan, spont_growth_rate):
        num_players = len(self.players)
        rewards = np.zeros(num_players, dtype=np.float32)

        for i, (player, action) in enumerate(zip(self.players, actions)):
            rewardfromaction = player.execute_policy_with_action(self, action)
            rewards[i] = rewardfromaction
            self.last_action_position = player.last_action_position

        self.update_bushes()
        self.update_players()

        if self.current_step > 0 and self.current_step % self.spont_growth_rate == 0:
            self.bush_spontaneous_growth(regrowth_rate, regrowth_function, max_lifespan, spont_growth_rate)

        self.current_step += 1
        next_state = self.get_state(environment)
        done = self.current_step >= self.max_steps

        if done and self.verbose:
            print(f"Episode finished after {self.current_step} steps")
        
        if self.verbose:
            self.print_matrix()
            print(f"Step {self.current_step} completed. Rewards: {rewards}")

        return next_state, rewards, done

    def print_matrix(self):
        # Implementation for printing the matrix
        pass