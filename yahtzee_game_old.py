import pygame
import random



class Die:
    def __init__(self, x, y, size):
        self.x = x
        self.y = y
        self.size = size
        self.value = 1
        self.selected = False
        self.disabled = False
        self.enable_color = (0, 0, 0)
        self.disable_color = (100, 100, 100)

    def reset(self):
        self.value = 1
        self.selected = False
        self.disabled = False
        self.enable_color = (0, 0, 0)
        self.disable_color = (100, 100, 100)

    def roll(self):
        self.value = random.randint(1, 6)

    def disable(self):
        self.disabled = True

    def enable(self):
        self.disabled = False

    def draw(self, screen):
        rect = pygame.Rect(self.x, self.y, self.size, self.size)
        if self.disabled == True:
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

        for dot in dot_positions.get(self.value, []):
            pygame.draw.circle(screen, (255, 0, 0), (self.x + dot[0], self.y + dot[1]), 6)

        if self.selected:
            pygame.draw.rect(screen, (255, 0, 0), rect, 2)

class Scoreboard:
    def __init__(self):
        self.button_rects = []  # To store button rectangles
        self.selected_button = None  # To store the index of the selected button
        self.yahtzee_table = [
            "Ones", "Twos", "Threes", "Fours", "Fives", "Sixes", "Score Top", "Score Bonus",
            "Three of a Kind", "Four of a Kind", "Full House", "Small Straight", "Large Straight", "Yahtzee", "Chance", "Score Bottom", "Full Score"
        ] 
        self.user_scores = [0] * len(self.yahtzee_table)
        self.user_scores_used = [False] * len(self.yahtzee_table)
        self.user_scores_used[6] = True
        self.user_scores_used[7] = True
        self.user_scores_used[15] = True
        self.user_scores_used[16] = True
        self.scores = [0] * len(self.yahtzee_table)
        self.end_turn_button_rect = pygame.Rect(700, 500, 100, 40)  # Rect for the end turn button
        self.end_turn_text = "End Turn"
        self.disable_all = True

    def reset(self):
        self.__init__()
 
    def draw_table(self, surface):
        surface.fill((255, 255, 255))

        # Draw the "End Turn" button
        if self.disable_all or self.selected_button is None:
            end_color =  (100, 100, 100)
        else:
            end_color =  (0, 100, 0)
        pygame.draw.rect(surface, end_color, self.end_turn_button_rect)  # Green button
        end_turn_font = pygame.font.Font(None, 24)
        end_turn_text_render = end_turn_font.render(self.end_turn_text, True, (255, 255, 255))  # White text
        surface.blit(end_turn_text_render, (720, 510))

        table_font = pygame.font.Font(None, 24)
        table_x = 700
        table_y = 0
        table_row_height = 30

        self.button_rects = []  # Clear the previous button rectangles

        for i, category in enumerate(self.yahtzee_table):
            category_text = table_font.render(category, True, (0, 0, 0))
            surface.blit(category_text, (table_x, table_y + i * table_row_height))

            # Create buttons for each category and store their rectangles
            button_rect = pygame.Rect(table_x, table_y + i * table_row_height, 100, 25)
            self.button_rects.append(button_rect)

            # Draw a border around the selected button
            if self.selected_button == i:
                pygame.draw.rect(surface, (255, 0, 0), button_rect, 2)

            # Check if the category is already used and disable its button visually
            if self.user_scores_used[i] or self.disable_all:
                pygame.draw.rect(surface, (150, 150, 150), button_rect)  # Gray out the button
                surface.blit(category_text, (table_x, table_y + i * table_row_height))

            # Display a third column initialized with zeros
            user_score_text = table_font.render(str(self.user_scores[i]), True, (0, 0, 0))
            surface.blit(user_score_text, (table_x + 150, table_y + i * table_row_height))



    def handle_end_turn_click(self):                    
        # Implement functionality for ending the turn here
        self.user_scores_used[self.selected_button] = True
        self.selected_button = None 
        self.disable_all = True

    def select_button(self, index):
        if index is None:
            return
        
        if self.user_scores_used[index] == True:
            return
        
        if self.disable_all == True:
            return
        
        if self.selected_button == index:  # If the same button is clicked again            
            self.selected_button = None  # Deselect the button
            self.user_scores[index] = 0
        else:
            if not self.selected_button is None:
                self.user_scores[self.selected_button] = 0
            self.selected_button = index  # Otherwise, select the new button
            self.user_scores[index] = self.scores[index]

    def calculate_scores(self, dice_values):
        sum_top = 0
        for i in range(1, 7):
            self.scores[i - 1] = dice_values.count(i) * i
            sum_top += self.user_scores[i - 1]
        
        self.scores[6] = sum_top
        if sum_top >= 63:
            sum_top += 35
            self.scores[7] = sum_top
        else:
            self.scores[7] = sum_top

        if any(dice_values.count(i) >= 3 for i in dice_values):
            self.scores[8] = sum(dice_values)
        else:
            self.scores[8] = 0

        if any(dice_values.count(i) >= 4 for i in dice_values):
            self.scores[9] = sum(dice_values)
        else:
            self.scores[9] = 0

        if any(dice_values.count(i) == 3 for i in dice_values) and any(dice_values.count(i) == 2 for i in dice_values):
            self.scores[10] = 25
        else:
            self.scores[10] = 0

        if {1, 2, 3, 4}.issubset(set(dice_values)) or {2, 3, 4, 5}.issubset(set(dice_values)) or {3, 4, 5, 6}.issubset(set(dice_values)):
            self.scores[11] = 30
        else:
            self.scores[11] = 0

        if {1, 2, 3, 4, 5}.issubset(set(dice_values)) or {2, 3, 4, 5, 6}.issubset(set(dice_values)):
            self.scores[12] = 40
        else:
            self.scores[12] = 0

        if any(dice_values.count(i) == 5 for i in dice_values):
            self.scores[13] = 50
        else:
            self.scores[13] = 0

        self.scores[14] = sum(dice_values)

        sum_bottom = 0
        for i in range(8, 15):
            sum_bottom += self.user_scores[i]
        self.scores[15] = sum_bottom 

        self.scores[16] = sum_bottom + sum_top

        return self.scores

class YahtzeeGame():
    def __init__(self):
        pygame.init()

        self.WIDTH, self.HEIGHT = 1000, 700
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Yahtzee")

        self.WHITE = (255, 255, 255)
        self.dice = [Die(50 + i * 120, 300, 80) for i in range(5)]
        self.rolling = False
        self.scores = [None] * 13
        self.scoreboard = Scoreboard()
        self.n_rolls = 0
        self.reset_button_rect = pygame.Rect(820, 500, 100, 40)  # Rect for the reset button
        self.reset_text = "Reset Game"


    def draw_reset_button(self, surface):
        pygame.draw.rect(surface, (255, 0, 0), self.reset_button_rect)  
        reset_font = pygame.font.Font(None, 24)
        reset_text_render = reset_font.render(self.reset_text, True, (255, 255, 255))  # White text
        surface.blit(reset_text_render, (820, 510))

    def handle_events(self):
        for event in pygame.event.get():
            res = self.handle_quit(event)
            if res == False: 
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.handle_roll()

            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()

                self.handle_scoreboard_selection(pos)
                self.handle_dice_selection(pos)
                self.handle_end_turn(pos)
                self.handle_reset(pos)

        return True
    
    def action_space_reset(self):
        self.reset()
        
    def handle_reset(self, pos):
        # Handle "Reset Game" button click
        if self.reset_button_rect.collidepoint(pos):
            self.reset()

    def reset(self):
        self.__init__()
    
    def handle_quit(self, event):
        if event.type == pygame.QUIT:
            pygame.quit()
            return False
        
    def action_space_roll(self):
        self.handle_roll()

    def handle_roll(self):
        if self.n_rolls >= 2:
            for die in self.dice:
                die.disable()
        if self.n_rolls < 3:
            if not self.rolling:
                self.rolling = True
                self.scoreboard.select_button(self.scoreboard.selected_button)
                self.scoreboard.disable_all = False
                self.n_rolls = self.n_rolls + 1


    def action_space_scoreboard_selection(self, index):
        mapping = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15]
        self.scoreboard.select_button(mapping[index])  
        self.handle_end_turn((761, 522))


    def handle_scoreboard_selection(self, pos):
        for index, rect in enumerate(self.scoreboard.button_rects):
            if rect.collidepoint(pos):  # Check if the mouse click is inside the button rectangle
                self.scoreboard.select_button(index)  # Toggle the selection of the clicked button

    def action_space_dice_selection(self, index, selected):
        self.dice[index].selected = selected 

    def handle_dice_selection(self, pos):
        # Handle dice selection (if necessary)
        if self.scoreboard.disable_all == False:
            for die in self.dice:
                if die.x <= pos[0] <= die.x + die.size and die.y <= pos[1] <= die.y + die.size:
                    die.selected = not die.selected

    def handle_end_turn(self, pos):
        # Handle "End Turn" button click
        # Check if the mouse click is inside the "End Turn" button rectangle
        if self.scoreboard.end_turn_button_rect.collidepoint(pos):
            if not self.scoreboard.selected_button is None:
                self.scoreboard.handle_end_turn_click()  # Call method to handle end turn click
                self.n_rolls = 0
                for die in self.dice:
                    die.selected = False
                    die.enable()

    def update_scores(self):
        if self.rolling:
            for die in self.dice:
                if not die.selected:
                    die.roll()
            self.rolling = False

            dice_values = [die.value for die in self.dice]
            self.scores = self.scoreboard.calculate_scores(dice_values)

    def render_screen(self, text):
        self.screen.fill(self.WHITE)
        
        text_no = pygame.font.Font(None, 36).render('Throw No.: ' + str(self.n_rolls), True, (0, 0, 0))

        self.scoreboard.draw_table(self.screen)
        for die in self.dice:
            die.draw(self.screen)

        table_font = pygame.font.Font(None, 24)
        table_x = 400 + 500
        table_y = 0
        table_row_height = 30
        for i, score in enumerate(self.scores):
            if score is not None:
                score_text = table_font.render(str(score), True, (0, 0, 0))
                text_width = score_text.get_width()
                self.screen.blit(score_text, (table_x - text_width, table_y + i * table_row_height))

        text_rect = text.get_rect(topleft=(50, 200))
        self.screen.blit(text, text_rect)

        text_no_rect = text_no.get_rect(topleft=(50, 250))
        self.screen.blit(text_no, text_no_rect)

        # Draw the reset button
        self.draw_reset_button(self.screen)

        pygame.display.flip()


    def run(self):
        clock = pygame.time.Clock()
        text = pygame.font.Font(None, 36).render("Press 'R' to roll the dice", True, (0, 0, 0))

        while True:
            if not self.handle_events():
                break

            self.update_scores()
            self.render_screen(text)

            clock.tick(30)

        pygame.quit()

if __name__ == "__main__":
    game = YahtzeeGame()
    game.run()