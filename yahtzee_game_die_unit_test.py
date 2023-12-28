import unittest
from unittest.mock import Mock
import pygame

# Import DieModel, DieViewModel, and DieView classes here...
from yahtzee_game import DieModel, DieViewModel, DieView

class TestDieModel(unittest.TestCase):
    def setUp(self):
        self.die_model = DieModel()

    def test_initial_values(self):
        self.assertEqual(self.die_model.value, 1)
        self.assertFalse(self.die_model.selected)
        self.assertFalse(self.die_model.disabled)

    def test_roll(self):
        # Ensure die rolls between 1 and 6 after rolling
        self.die_model.value = 5
        self.die_model.roll()
        self.assertTrue(1 <= self.die_model.value <= 6)

    def test_disable(self):
        self.die_model.value = 5
        self.assertFalse(self.die_model.disabled)
        self.die_model.disabled = True
        self.assertTrue(self.die_model.disabled)
        self.die_model.roll()
        self.assertEqual(self.die_model.value, 5)

    def test_enable(self):
        self.die_model.disabled = True
        self.die_model.value = 5
        self.assertTrue(self.die_model.disabled)
        self.die_model.disabled = False
        self.assertFalse(self.die_model.disabled)
        self.die_model.roll()
        self.assertTrue(1 <= self.die_model.value <= 6)

    def test_selected(self):
        self.die_model.value = 5
        self.assertFalse(self.die_model.selected)
        self.die_model.selected = True
        self.assertTrue(self.die_model.selected)
        self.die_model.roll()
        self.assertEqual(self.die_model.value, 5)

    def test_selected(self):
        self.die_model.selected = True
        self.die_model.value = 5
        self.assertTrue(self.die_model.selected)
        self.die_model.selected = False
        self.assertFalse(self.die_model.selected)
        self.die_model.roll()
        self.assertTrue(1 <= self.die_model.value <= 6)



class TestDieViewModel(unittest.TestCase):
    def setUp(self):
        self.die_model = DieModel()
        self.die_view_model = DieViewModel(self.die_model)

    def test_handle_roll(self):
        # Mocking random.randint() to test handle_roll()
        with unittest.mock.patch('random.randint') as mock_randint:
            mock_randint.return_value = 4
            self.die_view_model.handle_roll()
            self.assertEqual(self.die_model.value, 4)

    def test_handle_disable(self):
        self.assertFalse(self.die_model.disabled)
        self.die_view_model.handle_disable()
        self.assertTrue(self.die_model.disabled)

    def test_handle_enable(self):
        self.die_model.disabled = True
        self.assertTrue(self.die_model.disabled)
        self.die_view_model.handle_enable()
        self.assertFalse(self.die_model.disabled)

    def test_handle_disable(self):
        self.assertFalse(self.die_model.disabled)
        self.die_view_model.handle_disable()
        self.assertTrue(self.die_model.disabled)

    def test_handle_enable(self):
        self.die_model.disabled = True
        self.assertTrue(self.die_model.disabled)
        self.die_view_model.handle_enable()
        self.assertFalse(self.die_model.disabled)

    def test_handle_select(self):
        self.assertFalse(self.die_model.selected)
        self.die_view_model.handle_select()
        self.assertTrue(self.die_model.selected)

    def test_handle_unselect(self):
        self.die_model.selected = True
        self.assertTrue(self.die_model.selected)
        self.die_view_model.handle_unselect()
        self.assertFalse(self.die_model.selected)


if __name__ == '__main__':
    unittest.main()
