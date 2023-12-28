import pygame
import random

# Die Model
class DieModel:
    def __init__(self):
        self.value = 1
        self.selected = False
        self.disabled = False

    def roll(self):
        if self.disabled == True:
            return
        if self.selected == True:
            return
        self.value = random.randint(1, 6)


# Die ViewModel
class DieViewModel:
    def __init__(self, die_model):
        self.die_model = die_model

    def handle_roll(self):
        self.die_model.roll()

    def handle_disable(self):
        self.die_model.disabled = True

    def handle_enable(self):
        self.die_model.disabled = False


# Die View
class DieView:
    def __init__(self, x, y, size, die_model):
        self.x = x
        self.y = y
        self.size = size
        self.die_model = die_model
        self.enable_color = (0, 0, 0)
        self.disable_color = (100, 100, 100)

    def draw(self, screen):
        rect = pygame.Rect(self.x, self.y, self.size, self.size)
        if self.die_model.disabled:
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

        for dot in dot_positions.get(self.die_model.value, []):
            pygame.draw.circle(screen, (255, 0, 0), (self.x + dot[0], self.y + dot[1]), 6)

        if self.die_model.selected:
            pygame.draw.rect(screen, (255, 0, 0), rect, 2)




class ScoreboardModel:
    def __init__(self):
        self.scoreboard_data = {
            "yahtzee_table": [
                "Ones", "Twos", "Threes", "Fours", "Fives", "Sixes", "Score Top", "Score Bonus",
                "Three of a Kind", "Four of a Kind", "Full House", "Small Straight", "Large Straight",
                "Yahtzee", "Chance", "Score Bottom", "Full Score"
            ],
            "user_scores": [0] * 17,
            "user_scores_used": [False] * 17,
            6: True,
            7: True,
            15: True,
            16: True,
            "scores": [0] * 17
        }
        self.selected_button = None
        self.disable_all = True

    def calculate_scores(self, dice_values):
        data = self.scoreboard_data

        scores = [0] * 17

        # Calculate scores based on dice values and update the 'scores' array
        for i in range(1, 7):
            scores[i - 1] = dice_values.count(i) * i

        sum_top = sum(scores[:6])
        scores[6] = sum_top if sum_top >= 63 else 0
        scores[7] = sum_top + 35 if sum_top >= 63 else 0

        # Three of a kind
        if any(dice_values.count(i) >= 3 for i in dice_values):
            scores[8] = sum(dice_values)
        else:
            scores[8] = 0

        # Four of a kind
        if any(dice_values.count(i) >= 4 for i in dice_values):
            scores[9] = sum(dice_values)
        else:
            scores[9] = 0

        # Full House
        if any(dice_values.count(i) == 3 for i in dice_values) and any(dice_values.count(i) == 2 for i in dice_values):
            scores[10] = 25
        else:
            scores[10] = 0

        # Small Straight
        if {1, 2, 3, 4}.issubset(set(dice_values)) or {2, 3, 4, 5}.issubset(set(dice_values)) or {3, 4, 5, 6}.issubset(set(dice_values)):
            scores[11] = 30
        else:
            scores[11] = 0

        # Large Straight
        if {1, 2, 3, 4, 5}.issubset(set(dice_values)) or {2, 3, 4, 5, 6}.issubset(set(dice_values)):
            scores[12] = 40
        else:
            scores[12] = 0

        # Yahtzee
        if any(dice_values.count(i) == 5 for i in dice_values):
            scores[13] = 50
        else:
            scores[13] = 0

        scores[14] = sum(dice_values)

        sum_bottom = sum(data["user_scores"][8:15])
        scores[15] = sum_bottom

        scores[16] = sum_bottom + scores[6]

        data["scores"] = scores
        return scores



class ScoreboardViewModel:
    def __init__(self, scoreboard_model):
        self.scoreboard_model = scoreboard_model

    def handle_end_turn_click(self):
        if self.scoreboard_model.selected_button is not None:
            self.scoreboard_model.scoreboard_data["user_scores_used"][self.scoreboard_model.selected_button] = True
            self.scoreboard_model.selected_button = None
            self.scoreboard_model.disable_all = True

    def handle_scoreboard_selection(self, index):
        if index is None:
            return
        
        user_scores_used = self.scoreboard_model.scoreboard_data["user_scores_used"]
        user_scores = self.scoreboard_model.scoreboard_data["user_scores"]
        scores = self.scoreboard_model.scoreboard_data["scores"]
        disable_all = self.scoreboard_model.disable_all

        if user_scores_used[index] or disable_all:
            return
        
        selected_button = self.scoreboard_model.selected_button
        if selected_button == index:
            self.scoreboard_model.selected_button = None
            user_scores[index] = 0
        else:
            if selected_button is not None:
                user_scores[selected_button] = 0
            self.scoreboard_model.selected_button = index
            user_scores[index] = scores[index]



class ScoreboardView:
    def __init__(self, scoreboard_model):
        self.scoreboard_model = scoreboard_model
        self.button_rects = []
        self.end_turn_button_rect = pygame.Rect(700, 500, 100, 40)
        self.end_turn_text = "End Turn"

    def draw_table(self, surface):
        surface.fill((255, 255, 255))
        table_x = 700
        table_y = 0
        table_row_height = 30
        table_font = pygame.font.Font(None, 24)

        for i, category in enumerate(self.scoreboard_model.scoreboard_data["yahtzee_table"]):
            category_text = table_font.render(category, True, (0, 0, 0))
            surface.blit(category_text, (table_x, table_y + i * table_row_height))

            button_rect = pygame.Rect(table_x, table_y + i * table_row_height, 100, 25)
            self.button_rects.append(button_rect)

            if self.scoreboard_model.selected_button == i:
                pygame.draw.rect(surface, (255, 0, 0), button_rect, 2)

            if self.scoreboard_model.scoreboard_data["user_scores_used"][i] or self.scoreboard_model.disable_all:
                pygame.draw.rect(surface, (150, 150, 150), button_rect)
                surface.blit(category_text, (table_x, table_y + i * table_row_height))

            user_score_text = table_font.render(str(self.scoreboard_model.scoreboard_data["user_scores"][i]), True, (0, 0, 0))
            surface.blit(user_score_text, (table_x + 150, table_y + i * table_row_height))

        end_color = (100, 100, 100) if self.scoreboard_model.disable_all or self.scoreboard_model.selected_button is None else (0, 100, 0)
        pygame.draw.rect(surface, end_color, self.end_turn_button_rect)
        end_turn_font = pygame.font.Font(None, 24)
        end_turn_text_render = end_turn_font.render(self.end_turn_text, True, (255, 255, 255))
        surface.blit(end_turn_text_render, (720, 510))




class YahtzeeGameModel:
    def __init__(self):
        self.n_rolls = 0
        self.n_dice = 5
        self.dice = [DieModel() for _ in range(self.n_dice )]
        self.scoreboard_model = ScoreboardModel()

    def reset(self):
        self.n_rolls = 0

    def handle_roll(self):
        self.n_rolls += 1
    


class YahtzeeGameViewModel:
    def __init__(self, yahtzee_game_model):
        self.yahtzee_game_model = yahtzee_game_model
        self.scoreboard_view_model = ScoreboardViewModel(self.yahtzee_game_model.scoreboard_model)

    def handle_roll(self):
        if self.yahtzee_game_model.n_rolls >= 2:
            for die in self.yahtzee_game_model.dice:
                DieViewModel(die).handle_disable()

        if self.yahtzee_game_model.n_rolls < 3:
            if not self.yahtzee_game_model.rolling:
                self.yahtzee_game_model.rolling = True
                self.scoreboard_view_model.select_button(self.yahtzee_game_model.scoreboard_model.selected_button)
                self.yahtzee_game_model.scoreboard_model.disable_all = False
                self.yahtzee_game_model.n_rolls += 1

    def handle_reset(self, pos):
        self.yahtzee_game_model.reset()

    def handle_end_turn_click(self):
        self.yahtzee_game_model.reset()


class YahtzeeGameView:
    def __init__(self, yahtzee_game_model, yahtzee_game_view_model):
        self.yahtzee_game_model = yahtzee_game_model
        self.yahtzee_game_view_model = yahtzee_game_view_model
        self.WIDTH, self.HEIGHT = 1000, 700
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Yahtzee")
        self.WHITE = (255, 255, 255)
        self.reset_button_rect = pygame.Rect(820, 500, 100, 40)
        self.reset_text = "Reset Game"

    def draw_reset_button(self):
        pygame.draw.rect(self.screen, (255, 0, 0), self.reset_button_rect)
        reset_font = pygame.font.Font(None, 24)
        reset_text_render = reset_font.render(self.reset_text, True, (255, 255, 255))
        self.screen.blit(reset_text_render, (820, 510))

    def render_screen(self, text):
        self.screen.fill(self.WHITE)

        text_no = pygame.font.Font(None, 36).render('Throw No.: ' + str(self.yahtzee_game_model.n_rolls), True, (0, 0, 0))

        self.yahtzee_game_view_model.scoreboard_view_model.draw_table(self.screen)

        scores = self.yahtzee_game_view_model.get_scores()
        table_font = pygame.font.Font(None, 24)
        table_x = 400 + 500
        table_y = 0
        table_row_height = 30
        for i, score in enumerate(scores):
            if score is not None:
                score_text = table_font.render(str(score), True, (0, 0, 0))
                text_width = score_text.get_width()
                self.screen.blit(score_text, (table_x - text_width, table_y + i * table_row_height))

        text_rect = text.get_rect(topleft=(50, 200))
        self.screen.blit(text, text_rect)

        text_no_rect = text_no.get_rect(topleft=(50, 250))
        self.screen.blit(text_no, text_no_rect)

        self.draw_reset_button()

        pygame.display.flip()



class YahtzeeGame:
    def __init__(self):
        pygame.init()
        self.yahtzee_game_model = YahtzeeGameModel()
        self.yahtzee_game_view_model = YahtzeeGameViewModel(self.yahtzee_game_model)
        self.yahtzee_game_view = YahtzeeGameView(self.yahtzee_game_model, self.yahtzee_game_view_model)
        self.clock = pygame.time.Clock()
        self.text = pygame.font.Font(None, 36).render("Press 'R' to roll the dice", True, (0, 0, 0))

    def run(self):
        while True:
            if not self.yahtzee_game_view_model.handle_events():
                break

            self.yahtzee_game_view_model.update_scores()
            self.yahtzee_game_view.render_screen(self.text)

            self.clock.tick(30)

        pygame.quit()



# Main Game Execution
if __name__ == "__main__":
    yahtzee_game = YahtzeeGame()
    yahtzee_game.run()


