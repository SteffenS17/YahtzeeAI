import unittest
from yahtzee_game_env import YhatzeeEnv


class TestYhatzeeEnvRollingActions(unittest.TestCase):
    def setUp(self):
        self.env = YhatzeeEnv()
        self.observation = None

    def test_initialization_rolling_action(self):
        observation = self.env.reset()
        legal_actions = self.env.get_legal_actions()

        # Check initial state - assuming current_state is a list
        expected_state = [1] * self.env.NUM_DICE + [0] + [-1] * self.env.NUM_CATEGORIES
        self.assertEqual(self.env.current_state, expected_state,
                         msg="Array does not match the expected initial state")

        # Perform the initialization rolling action with index 0
        self.observation, reward, done, info = self.env.step(0)
        self.assertEqual(len(self.observation), self.env.observation_space)
        
        # Check if the values for dices (first self.env.NUM_DICE elements) are between 1 and 6
        for i in range(self.env.NUM_DICE):
            value = self.observation[i]
            with self.subTest(value=value):
                self.assertGreaterEqual(value, 1,
                                        msg="Value should be greater than or equal to 1")
                self.assertLessEqual(value, 6,
                                     msg="Value should be less than or equal to 6")

        # Check if the ROLL_POSITION is updated to 1 after the rolling action
        self.assertEqual(self.observation[self.env.ROLL_POSITION], 1,
                         msg="ROLL_POSITION should be updated to 1 after the rolling action")
        
        self.check_dice_models_match_observation(self.observation)

        
    def check_dice_models_match_observation(self, observation):
        # Check if the dice values match the values in the dice models
        for i, die_model in enumerate(self.env.dice_model):
            self.assertEqual(die_model.value, observation[i])

        # Check if the number of rolls matches the observation
        self.assertEqual(self.env.yahtzee_game_model.n_rolls, observation[self.env.ROLL_POSITION])

        # Check if the user scores in the observation match the actual scores in the environment
        user_scores = self.env.scoreboard_model.get_user_scores()
        for i in range(self.env.NUM_CATEGORIES):
            if observation[self.env.NUM_DICE + 1 + i] == -1:
                self.assertEqual(user_scores[i], None)
            else:
                self.assertEqual(user_scores[i], observation[self.env.NUM_DICE + 1 + i])


    def test_rolling_actions(self):
        # Perform rolling actions for indices greater than 0
        for action in range(1, self.env.NUM_DICE_SELECTIONS):
            # Additional initialization rolling action test            
            with self.subTest(action=action):
                observation = self.env.reset()
                self.test_initialization_rolling_action()
                legal_actions = self.env.get_legal_actions()

                # Perform rolling action
                self.observation, reward, done, info = self.env.step(action)

                # Validate observation, reward, and termination condition
                self.assertEqual(len(self.observation), self.env.observation_space)
                #self.assertIsInstance(reward, float)
                #self.assertIsInstance(done, bool)
                # Additional assertions specific to rolling actions


class TestYhatzeeEnvCategorySelections(TestYhatzeeEnvRollingActions):
    def setUp(self):
        self.env = YhatzeeEnv()

    def test_category_selections(self):
        for action in range(self.env.NUM_DICE_SELECTIONS, self.env.action_space):
            with self.subTest(action=action):
                # Reuse rolling test cases in the category selections
                self.test_initialization_rolling_action()

                # Perform category selection action
                self.observation, reward, done, info = self.env.step(action)                

                # Validate observation, reward, and termination condition
                self.assertEqual(len(self.observation), self.env.observation_space)

                observation_position = (action - (self.env.NUM_DICE_SELECTIONS)) + self.env.CATEGORIES_POSITION

                self.assertNotEqual(self.observation[observation_position], -1)

                self.check_dice_models_match_observation(self.observation)


if __name__ == '__main__':
    unittest.main()