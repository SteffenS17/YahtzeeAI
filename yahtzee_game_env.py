import numpy as np
import gym
from gym import spaces
import torch
import pygame
import time
import random
from itertools import product
import yahtzee_game


#https://github.com/patrickloeber/snake-ai-pytorch/blob/main/game.py
#https://github.com/mahowald/tictactoe/blob/master/tictactoe/env.py

class YhatzeeEnv(gym.Env):

    def __init__(self, summary: dict = None):
        super(YhatzeeEnv, self).__init__()

        pygame.init()
        self.dice_model = [yahtzee_game.DieModel() for _ in range(5)]
        self.dice_view_model = [yahtzee_game.DieViewModel(self.dice_model[i]) for i in range(5)]
        self.dice_view = [yahtzee_game.DieView(50 + i * 120, 300, 80, self.dice_view_model[i]) for i in range(5)]

        self.scoreboard_model = yahtzee_game.ScoreboardModel()
        self.scoreboard_view_model = yahtzee_game.ScoreboardViewModel(self.scoreboard_model)
        

        self.yahtzee_game_model = yahtzee_game.YahtzeeGameModel()
        self.yahtzee_game_view_model = yahtzee_game.YahtzeeGameViewModel(self.yahtzee_game_model, self.scoreboard_view_model, self.dice_view_model)

        self.scoreboard_view = yahtzee_game.ScoreboardView(self.scoreboard_view_model, self.yahtzee_game_view_model)
        self.yahtzee_game_view = yahtzee_game.YahtzeeGameView(self.yahtzee_game_view_model)
        self.clock = pygame.time.Clock()


        # Define the number of sides on a dice, the number of dice, and the number of categories
        self.NUM_DICE = 5
        self.dice_selection_permutations = list(product([0, 1], repeat=5))[0:-1]
        self.NUM_DICE_SELECTIONS = np.power(2, 5) - 1 #Select all is invalid 
        self.NUM_CATEGORIES = 13
        
        self.DICE_POSITION = 0
        self.ROLL_POSITION = 5
        self.CATEGORIES_POSITION = 6

        
        # Define the observation space (state space)
        #Observation Space: Dice State (Number of pips); Reroll State (Number of Rolls); Categories State (Score) 
        self.observation_space = self.NUM_DICE + 1 + self.NUM_CATEGORIES #spaces.Box(low=0, high=self.NUM_SIDES, shape=(self.NUM_DICE,), dtype=np.int32)        
        #ToDo add Possibilities?
        # Define the action space
        self.action_space = self.NUM_DICE_SELECTIONS + self.NUM_CATEGORIES
        
        # Initialize variables to hold the current state and done flag
        self.current_state = None
        self.done = False
        self.old_score = 0

        

        if summary is None:
            summary = {
                "total games": 0,
                "illegal moves": 0,
            }
        self.summary = summary

    def seed(self, seed=None):
        pass

    def reset(self):
        #Todo Reset for all viewmodels
         # Initialize or reset the state of the game
        self.yahtzee_game_view_model.handle_reset()
        self.current_state = [1,1,1,1,1] + [0] +[-1]*13
        self.done = False
        self.old_score = 0
        
        return self.current_state


    def get_legal_actions(self):
        legal_actions = []
        legal_selections = []
        user_scores = self.scoreboard_model.get_user_scores()

        for i, score in enumerate(user_scores):
            if score is None:
                legal_selections.append(i + self.NUM_DICE_SELECTIONS)

        #Third Roll:
        if self.yahtzee_game_model.n_rolls == 3:
            legal_actions = legal_selections
            return legal_actions  

        #Game start:
        if self.dice_view_model[0].die_model.select_disabled == True:
            legal_actions = [0]
            return legal_actions
        
        if self.scoreboard_view_model.scoreboard_model.disable_all == True:
            legal_actions = [0]
            return legal_actions

        #First and Second Roll
        if self.dice_view_model[0].die_model.disabled == False:
            legal_actions = list(range(self.NUM_DICE_SELECTIONS))
            legal_actions += legal_selections
            return legal_actions

        #Third Roll
        if self.dice_view_model[0].die_model.disabled == True:
            legal_actions = legal_selections
            return legal_actions  
        

    def get_category_completion_reward(self, i, score):
        max_reward = 20
        optimal_scores = [5*1, 5*2, 5*3, 5*4, 4*5, 5*6, 5*6, 5*6, 25, 30, 40, 50, 5*6]
        reward = max_reward * (score / optimal_scores[i])
        return reward

        
    def step(self, action):
        # Implement the step function to execute a step in the environment

        # Extract dice actions and category selection from the action

        illegal_action = False
        category_selected = None
        mapping = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14]

        if action < self.NUM_DICE_SELECTIONS:
            dice_selecetion = self.dice_selection_permutations[action]
            if action != 0:
                for i, die_view_model in enumerate(self.dice_view_model):
                    if dice_selecetion[i] == True:
                        die_view_model.handle_select()
                    else:
                        die_view_model.handle_unselect()
            rolled = False
            for die_view_model in self.dice_view_model:
                res = die_view_model.handle_roll()
                rolled = res | rolled
            if rolled == True:
                self.yahtzee_game_view_model.handle_roll()
            else:
                illegal_action = True
        else:
            #print('select category')       
            category_selected = action - self.NUM_DICE_SELECTIONS
            res = self.scoreboard_view_model.handle_scoreboard_selection(mapping[category_selected])
            if res == False:
                illegal_action = True
            self.scoreboard_view_model.handle_end_turn_click()  
            self.yahtzee_game_view_model.handle_end_turn_click()


        # Update game state here based on events or other logic
        dice_values = [die_model.value for die_model in self.dice_model]
        self.scoreboard_model.calculate_scores(dice_values)
        self.yahtzee_game_view_model.yahtzee_game_model.rolling = False
        # Perform the dice actions (re-roll or keep dice)

        # Update the current_state based on the chosen actions
        user_scores = []
        for i in mapping:
            c = self.scoreboard_model.categories[i]
            score = self.scoreboard_model.scoreboard_data[c]['user_score']
            if score is None:
                user_scores.append(-1)
            else:
                user_scores.append(score)
        self.current_state = dice_values + [self.yahtzee_game_model.n_rolls] + user_scores
        # Calculate reward and done status based on the selected category
        # Update any other necessary variables

        # Update with the actual reward based on the game rules

        # Adjust rewards based on legal actions
        if not illegal_action:
            c = self.scoreboard_model.categories[-1]
            new_score = self.scoreboard_model.scoreboard_data[c]['user_score'] 
            # Reward for score improvement
            reward = 0# (new_score - self.old_score)
            self.old_score = new_score

            # Reward for completing a category
            if not category_selected is None:
                c = self.scoreboard_model.categories[mapping[category_selected]]
                score = self.scoreboard_model.scoreboard_data[c]['user_score']
                reward += self.get_category_completion_reward(category_selected, score)

            # Reward for turn completion
            #reward += 0.1  # Small reward for completing a turn without illegal moves

        elif illegal_action == True:
            reward = -1


        if self.scoreboard_view_model.is_finished():
            c = self.scoreboard_model.categories[i]
            final_score = self.scoreboard_model.scoreboard_data[c]['user_score']
            # Reward higher final scores or penalize lower scores
            reward += final_score  # Adjust this based on desired outcomes
            self.done = True


        info = {}   # Additional information (if needed)
            
        return self.current_state, reward, self.done, info
    
    def render(self):
        # Render the game
        for event in pygame.event.get():   
            if event.type == pygame.QUIT:
                pygame.quit()
        self.yahtzee_game_view.render_screen()
        self.yahtzee_game_view.draw_reset_button()

        self.scoreboard_view.draw_table(self.yahtzee_game_view.screen) 
        for die_view in self.dice_view:
            die_view.draw(self.yahtzee_game_view.screen)
        pygame.display.flip()
        # Control the frame rate
        self.clock.tick(60)  # Limit to 60 frames per second    


if __name__ == "__main__":
    env = YhatzeeEnv()
    observation = env.reset()
    # Perform a random action and observe the outcome
    env.render()
    start_time = time.time() 
    while True:
        #time.sleep(1)
        legal_actions = env.get_legal_actions()
        action = random.choice(legal_actions)
        observation, reward, done, info = env.step(action)
        env.render()
        if env.scoreboard_view_model.is_finished():
            end_time = time.time() 
            elapsed_time = end_time - start_time
            start_time = time.time() 
            print(f"Elapsed time: {elapsed_time} seconds")
            print('Reward: ', reward)
            env.yahtzee_game_view_model.handle_reset()
            

    