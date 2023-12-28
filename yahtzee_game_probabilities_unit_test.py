import unittest
import numpy as np

from yahtzee_game_probabilities import YhatzeeTransitionMatrix
# Assume YhatzeeTransitionMatrix class is defined as provided in the initial code snippet

class TestTransitionMatrices(unittest.TestCase):

    def setUp(self):
        # Initialize the YhatzeeTransitionMatrix class
        self.ytm = YhatzeeTransitionMatrix()

    def test_transition_matrices_row_sum(self):
        transition_matrices = self.ytm.get_transition_matrices()

        for category, matrix in transition_matrices.items():
            # Iterate through each matrix
            rows, cols = matrix.shape

            for row in range(rows):
                # Check if the sum of each row is approximately equal to 1.0
                row_sum = np.sum(matrix[row, :])
                self.assertAlmostEqual(row_sum, 1.0, delta=1e-1, msg=f"Row sum for {category} matrix row {row} is not close to 1.0.")

if __name__ == '__main__':
    unittest.main()
