import pygame
import random
from yahtzee_game_probabilities import YhatzeeProbabilities

# Die Model
class DieModel:
    def __init__(self):
        self.value = 1
        self.selected = False
        self.disabled = False
        self.select_disabled = True

    def roll(self):
        if self.selected:
            return
        if self.disabled:
            return
        self.value = random.randint(1, 6)
    
    def reset(self):
        self.__init__()


# Die ViewModel
class DieViewModel:
    def __init__(self, die_model):
        self.die_model = die_model

    def handle_roll(self):
        self.handle_enable_select()
        if self.die_model.disabled == True:
            return False
        else:
            self.die_model.roll()
            return True
        
    def handle_toggle_select(self):
        if self.die_model.select_disabled:
            return False
        else:
            self.die_model.selected = not self.die_model.selected
            return True

    def handle_disable(self):
        self.die_model.disabled = True

    def handle_enable(self):
        self.die_model.disabled = False

    def handle_select(self):
        if self.die_model.select_disabled:
            return False
        else:
            self.die_model.selected = True
            return True

    def handle_unselect(self):
        if self.die_model.select_disabled:
            return False
        else:
            self.die_model.selected = False
            return True
    
    def handle_enable_select(self):
        self.die_model.select_disabled = False
    
    def handle_disable_select(self):
        self.die_model.select_disabled = True
    
    def handle_reset(self):
        self.die_model.reset()


# Die View
class DieView:
    def __init__(self, x, y, size, die_view_model):
        self.x = x
        self.y = y
        self.size = size
        self.die_view_model = die_view_model
        self.enable_color = (0, 0, 0)
        self.disable_color = (100, 100, 100)

    def draw(self, screen):
        rect = pygame.Rect(self.x, self.y, self.size, self.size)
        if self.die_view_model.die_model.disabled:
            pygame.draw.rect(screen, self.disable_color, rect)
        else:
            pygame.draw.rect(screen, self.enable_color, rect, 2)

        dot_positions = {
            1: [(self.size // 2, self.size // 2)],
            2: [(self.size // 4, self.size // 4), (self.size - self.size // 4, self.size - self.size // 4)],
            3: [(self.size // 4, self.size // 4), (self.size // 2, self.size // 2),
                (self.size - self.size // 4, self.size - self.size // 4)],
            4: [(self.size // 4, self.size // 4), (self.size - self.size // 4, self.size - self.size // 4),
                (self.size - self.size // 4, self.size // 4), (self.size // 4, self.size - self.size // 4)],
            5: [(self.size // 4, self.size // 4), (self.size - self.size // 4, self.size - self.size // 4),
                (self.size - self.size // 4, self.size // 4), (self.size // 4, self.size - self.size // 4),
                (self.size // 2, self.size // 2)],
            6: [
                (self.size // 4, self.size // 4), (self.size - self.size // 4, self.size - self.size // 4),
                (self.size // 4, self.size - self.size // 4), (self.size - self.size // 4, self.size // 4),
                (self.size // 4, self.size // 2), (self.size - self.size // 4, self.size // 2),
                (self.size // 4, self.size - self.size // 2), (self.size - self.size // 4, self.size - self.size // 2)
            ]
        }
        
        for dot in dot_positions.get(self.die_view_model.die_model.value, []):
            pygame.draw.circle(screen, (255, 0, 0), (self.x + dot[0], self.y + dot[1]), 6)

        if self.die_view_model.die_model.selected:
            pygame.draw.rect(screen, (255, 0, 0), rect, 2)

    def handle_events(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                self.die_view_model.handle_roll()
                return True
        if event.type == pygame.MOUSEBUTTONDOWN:  
            pos = pygame.mouse.get_pos()
            if self.x <= pos[0] <= self.x + self.size and self.y <= pos[1] <= self.y + self.size:
                self.die_view_model.handle_toggle_select()        
                return False
        return False


class ScoreboardModel:
    def __init__(self):
        self.top_categories = ["Ones", "Twos", "Threes", "Fours", "Fives", "Sixes"]
        self.bottom_categories = ["Three of a Kind", "Four of a Kind", "Full House", "Small Straight", "Large Straight", "Yahtzee", "Chance"]
        self.categories = self.top_categories + ["Score Top", "Score Bonus"] + self.bottom_categories + ["Score Bottom", "Full Score"]

        self.scoreboard_data = {}
        for category in self.categories:
            self.scoreboard_data[category] = {
                "user_score": None,
                "user_score_used": False,
                "score": 0,
                "probabilities_1": 0.0,
                "probabilities_2": 0.0,
                "probabilities_3": 0.0
            }

        # Mark certain scores as used
        self.scoreboard_data["Score Top"]["user_score_used"] = True
        self.scoreboard_data["Score Bonus"]["user_score_used"] = True
        self.scoreboard_data["Score Bottom"]["user_score_used"] = True
        self.scoreboard_data["Full Score"]["user_score_used"] = True
        self.selected_button = None
        self.disable_all = True
        self.scoreboard_finished = False
        self.yahtzee_game_probabilities = YhatzeeProbabilities()
    
    def calculate_probabilities(self, n_throws, dice_values):
        probabilities = self.yahtzee_game_probabilities.calculate_score_probabilities(n_throws, dice_values)
        for category, probability in probabilities.items():
            if len(probability) == 3:
                self.scoreboard_data[category]['probabilities_1'] = probability[0]
                self.scoreboard_data[category]['probabilities_2'] = probability[1]
                self.scoreboard_data[category]['probabilities_3'] = probability[2]
            elif len(probability) == 2:
                self.scoreboard_data[category]['probabilities_1'] = 0.0
                self.scoreboard_data[category]['probabilities_2'] = probability[1]
                self.scoreboard_data[category]['probabilities_3'] = probability[2]
            elif len(probability) == 1:
                self.scoreboard_data[category]['probabilities_1'] = 0.0
                self.scoreboard_data[category]['probabilities_2'] = 0.0
                self.scoreboard_data[category]['probabilities_3'] = probability[2]
            else:
                self.scoreboard_data[category]['probabilities_1'] = 0.0
                self.scoreboard_data[category]['probabilities_2'] = 0.0
                self.scoreboard_data[category]['probabilities_3'] = 0.0


    def get_user_scores(self):        
        user_scores = []
        for c in self.scoreboard_data:
            if 'Score' in c:
                continue
            user_scores.append(self.scoreboard_data[c]['user_score'])
        return user_scores


    def calculate_scores(self, dice_values):
        #self.calculate_probabilities(0, [1, 2, 3, 4, 5])

        for i in range(1, 7):
            count = dice_values.count(i)
            self.scoreboard_data[self.top_categories[i - 1]]["score"] = count * i

        sum_top = sum(filter(None, [self.scoreboard_data[c]["user_score"] for c in self.top_categories]))
        self.scoreboard_data["Score Top"]["score"] = sum_top
        if sum_top >= 63:
            self.scoreboard_data["Score Bonus"]["score"] = sum_top + 35
        else:
            self.scoreboard_data["Score Bonus"]["score"] = sum_top

        category_functions = {
            "Three of a Kind": lambda: sum(dice_values) if any(dice_values.count(i) >= 3 for i in dice_values) else 0,
            "Four of a Kind": lambda: sum(dice_values) if any(dice_values.count(i) >= 4 for i in dice_values) else 0,
            "Full House": lambda: 25 if any(dice_values.count(i) == 3 for i in dice_values) and any(dice_values.count(i) == 2 for i in dice_values) else 0,
            "Small Straight": lambda: 30 if {1, 2, 3, 4}.issubset(set(dice_values)) or {2, 3, 4, 5}.issubset(set(dice_values)) or {3, 4, 5, 6}.issubset(set(dice_values)) else 0,
            "Large Straight": lambda: 40 if {1, 2, 3, 4, 5}.issubset(set(dice_values)) or {2, 3, 4, 5, 6}.issubset(set(dice_values)) else 0,
            "Yahtzee": lambda: 50 if any(dice_values.count(i) == 5 for i in dice_values) else 0,
            "Chance": lambda: sum(dice_values)
        }

        for category, function in category_functions.items():
            self.scoreboard_data[category]["score"] = function()

        sum_bottom = sum(filter(None, [self.scoreboard_data[c]["user_score"] for c in self.bottom_categories]))
        self.scoreboard_data["Score Bottom"]["score"] = sum_bottom
        self.scoreboard_data["Full Score"]["score"] = sum_bottom + self.scoreboard_data["Score Bonus"]["score"]

        if all(self.scoreboard_data[c]["user_score_used"] for c in self.categories):
            self.scoreboard_finished = True
    
    def reset(self):
        self.__init__()



class ScoreboardViewModel:
    def __init__(self, scoreboard_model):
        self.scoreboard_model = scoreboard_model

    def handle_end_turn_click(self):
        if self.scoreboard_model.selected_button is not None:
            selected_category = self.scoreboard_model.categories[self.scoreboard_model.selected_button]
            self.scoreboard_model.scoreboard_data[selected_category]["user_score_used"] = True
            self.scoreboard_model.selected_button = None
            self.scoreboard_model.disable_all = True

    def handle_scoreboard_selection(self, index):
        # Check if the index is None (invalid), return False
        if index is None:
            return False
        
        # Get the currently selected button index
        selected_button = self.scoreboard_model.selected_button
        selected_category = self.scoreboard_model.categories[index]
        disable_all = self.scoreboard_model.disable_all   

        # If the selected scoreboard entry is already used or all entries are disabled, return False
        if self.scoreboard_model.scoreboard_data[selected_category]["user_score_used"] or disable_all:
            return False
                 
        # Check if the currently selected button is the same as the newly selected index
        if selected_button == index:
            # If it is, deselect the button
            self.scoreboard_model.selected_button = None
            self.scoreboard_model.scoreboard_data[selected_category]["user_score"] = None  # Set the score to None to indicate deselection
        else:
            # If another button is already selected, deselect it first
            if selected_button is not None:
                prev_selected_category = self.scoreboard_model.categories[selected_button]
                self.scoreboard_model.scoreboard_data[prev_selected_category]["user_score"] = None  # Set the score to None for the previously selected button

            # Select the new button and assign its score
            self.scoreboard_model.selected_button = index
            self.scoreboard_model.scoreboard_data[selected_category]['user_score'] = self.scoreboard_model.scoreboard_data[selected_category]["score"]

        return True  # Return True indicating a change in selection or deselection
    
    def handle_reset(self):
        self.scoreboard_model.reset()

    def is_finished(self):
        return self.scoreboard_model.scoreboard_finished



class ScoreboardView:
    def __init__(self, scoreboard_view_model, yahtzee_game_view_model):
        self.scoreboard_view_model = scoreboard_view_model
        self.yahtzee_game_view_model = yahtzee_game_view_model
        self.button_rects = []
        self.end_turn_button_rect = pygame.Rect(700, 500, 100, 40)
        self.end_turn_text = "End Turn"

    def draw_table(self, surface):
        # Fill the surface with a white background
        surface.fill((255, 255, 255))
        self.button_rects = []
        # Define initial positions and font for the table
        table_x = 700
        table_y = 0
        table_row_height = 30
        table_font = pygame.font.Font(None, 24)

        # Loop through each category in the Yahtzee table
        for i, category in enumerate(self.scoreboard_view_model.scoreboard_model.scoreboard_data):
            # Render and display the category name text on the surface
            category_text = table_font.render(category, True, (0, 0, 0))
            surface.blit(category_text, (table_x, table_y + i * table_row_height))

            # Create a button rectangle for the category and append it to the list of button rectangles
            button_rect = pygame.Rect(table_x, table_y + i * table_row_height, 100, 25)
            self.button_rects.append(button_rect)

            # Highlight the button rectangle if it's currently selected
            if self.scoreboard_view_model.scoreboard_model.selected_button == i:
                pygame.draw.rect(surface, (255, 0, 0), button_rect, 2)

            # If the category is used or all categories are disabled, show it as disabled
            if self.scoreboard_view_model.scoreboard_model.scoreboard_data[category]["user_score_used"] or self.scoreboard_view_model.scoreboard_model.disable_all:
                pygame.draw.rect(surface, (150, 150, 150), button_rect)
                surface.blit(category_text, (table_x, table_y + i * table_row_height))

            # Render and display the user score for the category
            user_score_text = table_font.render(str(self.scoreboard_view_model.scoreboard_model.scoreboard_data[category]["user_score"]), True, (0, 0, 0))
            surface.blit(user_score_text, (table_x + 150, table_y + i * table_row_height))

            # Render and display the total score for the category
            score_text = table_font.render(str(self.scoreboard_view_model.scoreboard_model.scoreboard_data[category]["score"]), True, (0, 0, 0))
            surface.blit(score_text, (table_x + 200, table_y + i * table_row_height))

            # Render and display the probabilities for the category
            prob_text = table_font.render("{:.2f}".format(self.scoreboard_view_model.scoreboard_model.scoreboard_data[category]["probabilities_1"]*100.0), True, (0, 0, 0))
            surface.blit(prob_text, (table_x + 250, table_y + i * table_row_height))
            prob_text = table_font.render("{:.2f}".format(self.scoreboard_view_model.scoreboard_model.scoreboard_data[category]["probabilities_2"]*100.0), True, (0, 0, 0))
            surface.blit(prob_text, (table_x + 350, table_y + i * table_row_height))
            prob_text = table_font.render("{:.2f}".format(self.scoreboard_view_model.scoreboard_model.scoreboard_data[category]["probabilities_3"]*100.0), True, (0, 0, 0))
            surface.blit(prob_text, (table_x + 450, table_y + i * table_row_height))

        # Determine the color of the "End Turn" button based on game state
        if self.scoreboard_view_model.scoreboard_model.disable_all or self.scoreboard_view_model.scoreboard_model.selected_button is None:
            end_color = (100, 100, 100)  # Greyed out if all disabled or no button selected
        else:
            end_color = (0, 100, 0)  # Green color if active

        # Draw the "End Turn" button rectangle and render its text
        pygame.draw.rect(surface, end_color, self.end_turn_button_rect)
        end_turn_font = pygame.font.Font(None, 24)
        end_turn_text_render = end_turn_font.render(self.end_turn_text, True, (255, 255, 255))
        surface.blit(end_turn_text_render, (720, 510))

        if self.scoreboard_view_model.is_finished():            
            fin_text_render = table_font.render('Finished', True, (255, 0, 0))
            surface.blit(fin_text_render, (720, 550))

    def handle_events(self, event):
        if not event.type == pygame.MOUSEBUTTONDOWN:
            return 
        
        pos = pygame.mouse.get_pos()
        for index, rect in enumerate(self.button_rects):
            if rect.collidepoint(pos):  # Check if the mouse click is inside the button rectangle
                self.scoreboard_view_model.handle_scoreboard_selection(index)  # Toggle the selection of the clicked button

        if self.end_turn_button_rect.collidepoint(pos):
            if not self.scoreboard_view_model.scoreboard_model.selected_button is None:
                self.scoreboard_view_model.handle_end_turn_click()
                self.yahtzee_game_view_model.handle_end_turn_click()





class YahtzeeGameModel:
    def __init__(self):
        self.n_rolls = 0
        self.n_dice = 5
        self.rolling = False

    def reset(self):
        self.n_rolls = 0

    def handle_roll(self):
        if self.n_rolls > 2:
            return
        self.n_rolls += 1
    


class YahtzeeGameViewModel:
    def __init__(self, yahtzee_game_model, scoreboard_view_model, dice_view_model):
        self.yahtzee_game_model = yahtzee_game_model
        self.scoreboard_view_model = scoreboard_view_model
        self.dice_view_model = dice_view_model

    def handle_roll(self):
        if self.yahtzee_game_model.n_rolls >= 2:
            for die_view_model in self.dice_view_model:
                die_view_model.handle_disable()

        if self.yahtzee_game_model.n_rolls < 3:
            if not self.yahtzee_game_model.rolling:
                self.yahtzee_game_model.rolling = True
                self.scoreboard_view_model.handle_scoreboard_selection(self.scoreboard_view_model.scoreboard_model.selected_button)
                self.scoreboard_view_model.scoreboard_model.disable_all = False
                self.yahtzee_game_model.n_rolls += 1

    def handle_reset(self):
        self.yahtzee_game_model.reset()
        self.scoreboard_view_model.handle_reset()
        for die_view_model in self.dice_view_model:
            die_view_model.handle_reset()

    def handle_end_turn_click(self):
        self.yahtzee_game_model.reset()
        for die_view_model in self.dice_view_model:
            die_view_model.handle_unselect()
            die_view_model.handle_enable()
            die_view_model.handle_disable_select()


class YahtzeeGameView:
    def __init__(self, yahtzee_game_view_model):        
        self.yahtzee_game_view_model = yahtzee_game_view_model
        self.WIDTH, self.HEIGHT = 1300, 700
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Yahtzee")
        self.WHITE = (255, 255, 255)
        self.reset_button_rect = pygame.Rect(820, 500, 100, 40)
        self.reset_text = "Reset Game"
        self.text = pygame.font.Font(None, 36).render("Press 'R' to roll the dice", True, (0, 0, 0))

    def draw_reset_button(self):
        pygame.draw.rect(self.screen, (255, 0, 0), self.reset_button_rect)
        reset_font = pygame.font.Font(None, 24)
        reset_text_render = reset_font.render(self.reset_text, True, (255, 255, 255))
        self.screen.blit(reset_text_render, (820, 510))

    def render_screen(self):
        self.screen.fill(self.WHITE)

        text_no = pygame.font.Font(None, 36).render('Throw No.: ' + str(self.yahtzee_game_view_model.yahtzee_game_model.n_rolls), True, (0, 0, 0))

        text_rect = self.text.get_rect(topleft=(50, 200))
        self.screen.blit(self.text, text_rect)

        text_no_rect = text_no.get_rect(topleft=(50, 250))
        self.screen.blit(text_no, text_no_rect)
        self.draw_reset_button()

    def handle_events(self, event):
        if not event.type == pygame.MOUSEBUTTONDOWN:
            return
        
        pos = pygame.mouse.get_pos()
        if self.reset_button_rect.collidepoint(pos):
            self.yahtzee_game_view_model.handle_reset()
        return True  



class YahtzeeGame:
    def __init__(self):
        pygame.init()
        self.dice_model = [DieModel() for _ in range(5)]
        self.dice_view_model = [DieViewModel(self.dice_model[i]) for i in range(5)]
        self.dice_view = [DieView(50 + i * 120, 300, 80, self.dice_view_model[i]) for i in range(5)]

        self.scoreboard_model = ScoreboardModel()
        self.scoreboard_view_model = ScoreboardViewModel(self.scoreboard_model)
        

        self.yahtzee_game_model = YahtzeeGameModel()
        self.yahtzee_game_view_model = YahtzeeGameViewModel(self.yahtzee_game_model, self.scoreboard_view_model, self.dice_view_model)

        self.scoreboard_view = ScoreboardView(self.scoreboard_view_model, self.yahtzee_game_view_model)
        self.yahtzee_game_view = YahtzeeGameView(self.yahtzee_game_view_model)
        self.clock = pygame.time.Clock()
        
    
    def handle_external_commands(self):
        if self.message_queue is None:
            return
        
        if self.message_queue.empty():
            return

        command = self.message_queue.get()
        print(command)
        if command == 'roll':
            for die_view_model in self.dice_view_model:
                die_view_model.handle_roll()
                self.yahtzee_game_view_model.handle_roll()
        if 'select_dice' in command: #select_dice[1,1,1,1,0]
            command = command.replace('select_dice', '')
            dice_selection = list(map(str.strip, command.strip('][').replace('"', '').split(',')))
            for i, die_view_model in enumerate(self.dice_view_model):
                if dice_selection[i] == 0:
                    die_view_model.handle_unselect()
                else:
                    die_view_model.handle_select()
        if 'select_category' in command: #select_category[0]
            command = command.replace('select_category', '')
            command = command.strip('][')
            index = int(command)
            mapping = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15]
            self.scoreboard_view_model.handle_scoreboard_selection(mapping[index])
            self.scoreboard_view_model.handle_end_turn_click()       
        if command == 'reset':
            self.__init__()         
 
    def handle_quit(self, event):
        if event.type == pygame.QUIT:
            pygame.quit()
            return False
        

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():                
                self.handle_quit(event)
                                   
                # Handle events for dice, scoreboard, etc.
                rolled = False
                for die_view in self.dice_view:
                    res = die_view.handle_events(event)
                    rolled = rolled | res
                self.scoreboard_view.handle_events(event)
                self.yahtzee_game_view.handle_events(event)
                if rolled == True:
                    self.yahtzee_game_view_model.handle_roll()
            # Update game state here based on events or other logic
            dice_values = [die_model.value for die_model in self.dice_model]
            self.scoreboard_model.calculate_scores(dice_values)
            self.scoreboard_model.calculate_probabilities(self.yahtzee_game_model.n_rolls, dice_values)
            self.yahtzee_game_view_model.yahtzee_game_model.rolling = False
            # Render the game
            self.yahtzee_game_view.render_screen()
            self.yahtzee_game_view.draw_reset_button()

            self.scoreboard_view.draw_table(self.yahtzee_game_view.screen)       

            for die_view in self.dice_view:
                die_view.draw(self.yahtzee_game_view.screen)

            pygame.display.flip()
            # Control the frame rate
            self.clock.tick(60)  # Limit to 60 frames per second

        pygame.quit()



# Main Game Execution
if __name__ == "__main__":

    yahtzee_game = YahtzeeGame()
    yahtzee_game.run()


