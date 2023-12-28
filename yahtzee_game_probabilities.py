import scipy.special
import math
import numpy as np


class YhatzeeTransitionMatrix:
        # Add other categories and their state functions here
    #S Dice Sides
    @staticmethod
    def prob(n, x, S):    
        comb = scipy.special.binom(n, x)*np.power(S-1,n-x)
        return comb
    #Number of Dice
    #Array of combinations e.g. AAABCDE -> p_i = (3,1,1,1,1) -> n = (4,0,1)
    @staticmethod
    def multinom(N, n, S=6):
        k = np.sum(n)
        comb = scipy.special.binom(S, k)*math.factorial(k)
        perm_num = math.factorial(N)
        perm_denom = 1
        for i, n_i in enumerate(n):
            prod = np.power(math.factorial(i + 1),n_i)*math.factorial(n_i)
            perm_denom *= prod
        return comb*perm_num/perm_denom
    
    @staticmethod
    def get_top():    
        return np.array([
        [(np.power(6,5) - YhatzeeTransitionMatrix.prob(5,2,6) - YhatzeeTransitionMatrix.prob(5,3,6) - YhatzeeTransitionMatrix.prob(5,4,6) - YhatzeeTransitionMatrix.prob(5,5,6))/np.power(6,5), 
         YhatzeeTransitionMatrix.prob(5,2,6)/np.power(6,5), 
         YhatzeeTransitionMatrix.prob(5,3,6)/np.power(6,5),
         YhatzeeTransitionMatrix.prob(5,4,6)/np.power(6,5), 
         YhatzeeTransitionMatrix.prob(5,5,6)/np.power(6,5)],

        [0, 
         (YhatzeeTransitionMatrix.prob(3,0,6) - 5)/np.power(6,3) ,
         (YhatzeeTransitionMatrix.prob(3,1,6) + 5)/np.power(6,3) ,
         YhatzeeTransitionMatrix.prob(3,2,6)/np.power(6,3), 
         YhatzeeTransitionMatrix.prob(3,3,6)/np.power(6,3)],

        [0, 
         0, 
         (np.power(6,2) - YhatzeeTransitionMatrix.prob(2,1,6) - YhatzeeTransitionMatrix.prob(2,2,6))/np.power(6,2) ,
         YhatzeeTransitionMatrix.prob(2,1,6)/np.power(6,2), 
         YhatzeeTransitionMatrix.prob(2,2,6)/np.power(6,2)],

        [0, 
         0, 
         0, 
         (np.power(6,1)- YhatzeeTransitionMatrix.prob(1,1,6))/np.power(6,1),
         YhatzeeTransitionMatrix.prob(1,1,6)/np.power(6,1)],

        [0,0,0,0,1]
        ])
        
    @staticmethod
    def get_threes():
        return np.array([
        [YhatzeeTransitionMatrix.multinom(5, [5])/np.power(6,5), 
         (YhatzeeTransitionMatrix.multinom(5, [1,2]) + YhatzeeTransitionMatrix.multinom(5, [3,1]))/np.power(6,5), 
         (YhatzeeTransitionMatrix.multinom(5, [2,0,1]) + YhatzeeTransitionMatrix.multinom(5, [0,1,1]) + YhatzeeTransitionMatrix.multinom(5, [1,0,0,1]) + YhatzeeTransitionMatrix.multinom(5, [0,0,0,0,1]))/np.power(6,5)],
        
        [0, 
         (YhatzeeTransitionMatrix.prob(3,0,6) - 5)/np.power(6,3), 
         (YhatzeeTransitionMatrix.prob(3,1,6) + YhatzeeTransitionMatrix.prob(3,2,6) + YhatzeeTransitionMatrix.prob(3,3,6) + 5)/np.power(6,3)],
        [0, 0, 1]
        ])

    @staticmethod
    def get_fours():
        return np.array([
        [YhatzeeTransitionMatrix.multinom(5, [5])/np.power(6,5), 
         (YhatzeeTransitionMatrix.multinom(5, [1,2]) + YhatzeeTransitionMatrix.multinom(5, [3,1]))/np.power(6,5), 
         (YhatzeeTransitionMatrix.multinom(5, [2,0,1]) + YhatzeeTransitionMatrix.multinom(5, [0,1,1]))/np.power(6,5), 
         (YhatzeeTransitionMatrix.multinom(5, [1,0,0,1]) + YhatzeeTransitionMatrix.multinom(5, [0,0,0,0,1]))/np.power(6,5)],

        [0, 
         (YhatzeeTransitionMatrix.prob(3,0,6) - 5)/np.power(6,3), 
         (YhatzeeTransitionMatrix.prob(3,1,6) + 5)/np.power(6,3), 
         (YhatzeeTransitionMatrix.prob(3,2,6) + YhatzeeTransitionMatrix.prob(3,3,6))/np.power(6,3)],

        [0, 
         0, 
         YhatzeeTransitionMatrix.prob(2,0,6)/np.power(6,2), 
         (YhatzeeTransitionMatrix.prob(2,1,6) + YhatzeeTransitionMatrix.prob(2,2,6))/np.power(6,2)],

        [0, 0, 0, 1]
        ])

    @staticmethod
    def get_full_house():
        #ToDo this calc is wrong
        return np.array([
        [YhatzeeTransitionMatrix.multinom(5, [5])/np.power(6,5), 
         YhatzeeTransitionMatrix.multinom(5, [3,1])/np.power(6,5), 
         (YhatzeeTransitionMatrix.multinom(5, [2,0,1]) + YhatzeeTransitionMatrix.multinom(5, [1,0,0,1]) + YhatzeeTransitionMatrix.multinom(5, [0,0,0,0,1]))/np.power(6,5), 
         YhatzeeTransitionMatrix.multinom(5, [1,2])/np.power(6,5), 
         YhatzeeTransitionMatrix.multinom(5, [0,1,1])/np.power(6,5)],

        [0,
         YhatzeeTransitionMatrix.multinom(3, [3], S=5)/np.power(6,3), 
         86/np.power(6,3), 
         YhatzeeTransitionMatrix.multinom(3, [1,1], S=5)/np.power(6,3), 
         (YhatzeeTransitionMatrix.multinom(3, [0,0,1], S=5) + YhatzeeTransitionMatrix.multinom(2, [0,1], S=5))/np.power(6,3)], #Pair

        [0,
         0,
         (np.power(6,2)-YhatzeeTransitionMatrix.multinom(2, [0,1], S=5))/np.power(6,2),
         0, 
         YhatzeeTransitionMatrix.multinom(2, [0,1], S=5)/np.power(6,2)],#Threes

        [0,
         0,
         0,
         YhatzeeTransitionMatrix.prob(1,0,5)/np.power(6,1),
         (YhatzeeTransitionMatrix.prob(1,1,6)*2)/np.power(6,1)],#Two Pairs

        [0,0,0,0,1]#Full House
        ])

    @staticmethod
    def get_small_straight():
        #http://www.datagenetics.com/blog/january42012/
        #https://issuu.com/milliemince/docs/using_markov_chains_and_probabilistic_modeling_to_
        return np.array([
            [108/1296, 525/1296, 582/1296, 108/1296],
            [0, 64/216, 122/216, 30/216],
            [0,0,25/36, 11/36],
            [0, 0, 0, 1]         
        ])

    @staticmethod
    def get_large_straight():
        return np.array([
            [16/1296, 260/1296, 660/1296, 336/1296, 24/1296],
            [0, 27/216, 111/216, 72/216, 6/216],
            [0,0,16/36, 18/36, 2/36],
            [0, 0, 0, 5/6, 1/6],
            [0, 0, 0, 0, 1]  
        ])

    @staticmethod
    def get_yahtzee():
        return np.array([
        [YhatzeeTransitionMatrix.multinom(5, [5])/np.power(6,5), 
         (YhatzeeTransitionMatrix.multinom(5, [3,1]) + YhatzeeTransitionMatrix.multinom(5, [1,2]))/np.power(6,5), 
         (YhatzeeTransitionMatrix.multinom(5, [2,0,1]) + YhatzeeTransitionMatrix.multinom(5, [0,1,1]))/np.power(6,5), 
         YhatzeeTransitionMatrix.multinom(5, [1,0,0,1])/np.power(6,5), 
         YhatzeeTransitionMatrix.multinom(5, [0,0,0,0,1])/np.power(6,5)],

        [0, 
         (YhatzeeTransitionMatrix.prob(3,0,6) - 5)/np.power(6,3), 
         (YhatzeeTransitionMatrix.prob(3,1,6) + 5)/np.power(6,3),
         YhatzeeTransitionMatrix.prob(3,2,6)/np.power(6,3), 
         YhatzeeTransitionMatrix.prob(3,3,6)/np.power(6,3)],

        [0, 
         0, 
         (np.power(6,2) - YhatzeeTransitionMatrix.prob(2,1,6) - YhatzeeTransitionMatrix.prob(2,2,6))/np.power(6,2),
         YhatzeeTransitionMatrix.prob(2,1,6)/np.power(6,2), 
         YhatzeeTransitionMatrix.prob(2,2,6)/np.power(6,2)],
        
        [0, 
         0, 
         0, 
         (np.power(6,1)- YhatzeeTransitionMatrix.prob(1,1,6))/np.power(6,1),
         YhatzeeTransitionMatrix.prob(1,1,6)/np.power(6,1)],

        [0,0,0,0,1]
        ])
    
    @staticmethod
    def get_transition_matrices():
        return {
            'Ones': YhatzeeTransitionMatrix.get_top(),
            'Twos': YhatzeeTransitionMatrix.get_top(),
            'Threes': YhatzeeTransitionMatrix.get_top(),
            'Fours': YhatzeeTransitionMatrix.get_top(),
            'Fives': YhatzeeTransitionMatrix.get_top(),
            'Sixes': YhatzeeTransitionMatrix.get_top(),
            'Three of a Kind': YhatzeeTransitionMatrix.get_threes(),
            'Four of a Kind': YhatzeeTransitionMatrix.get_fours(),
            'Full House': YhatzeeTransitionMatrix.get_full_house(),
            'Small Straight': YhatzeeTransitionMatrix.get_small_straight(),
            'Large Straight': YhatzeeTransitionMatrix.get_large_straight(),
            'Yahtzee': YhatzeeTransitionMatrix.get_yahtzee()
            # Add other categories and their matrices here
        }

    def print_transition_matrices(self):
        np.set_printoptions(suppress=True)
        print(YhatzeeTransitionMatrix.get_transition_matrices())
        np.set_printoptions(suppress=False)

class YhatzeeProbabilities():
    def __init__(self):
        self.yahtzee_transition_matrix = YhatzeeTransitionMatrix()
        #self.yahtzee_transition_matrix.print_transition_matrices()

    @staticmethod
    def get_top_state(n_throws, dice_values, dice_number):
        if n_throws == 0:
            return np.array([1,0,0,0,0])
        
        state = np.array([0,0,0,0,0])
        count = dice_values.count(dice_number)
        
        if count <= 1:
            state[0] = 1
        else:
            state[count - 1] = 1
        return state
        
    @staticmethod
    def get_ones_state(n_throws, dice_values):
        return YhatzeeProbabilities.get_top_state(n_throws, dice_values, 1)

    @staticmethod
    def get_twos_state(n_throws, dice_values):
        return YhatzeeProbabilities.get_top_state(n_throws, dice_values, 2)

    @staticmethod
    def get_threes_state(n_throws, dice_values):
        return YhatzeeProbabilities.get_top_state(n_throws, dice_values, 3)

    @staticmethod
    def get_fours_state(n_throws, dice_values):
        return YhatzeeProbabilities.get_top_state(n_throws, dice_values, 4)

    @staticmethod
    def get_fives_state(n_throws, dice_values):
        return YhatzeeProbabilities.get_top_state(n_throws, dice_values, 5)

    @staticmethod
    def get_sixes_state(n_throws, dice_values):
        return YhatzeeProbabilities.get_top_state(n_throws, dice_values, 6)

    @staticmethod
    def get_three_of_a_kind_state(n_throws, dice_values):
        if n_throws == 0:
            return np.array([1,0,0])
            #if any(dice_values.count(i) == 3 for i in dice_values):
            #    state = np.array([0,0,1])
            #else:
            #    state = np.array([1,0,0])
        else:
            state = [0, 0, 0]
            # Check if all dice have the same value (Yahtzee)
            if any(dice_values.count(i) >= 3 for i in dice_values):
                state = np.array([0,0,1])
            else:
                # Count the occurrences of each dice value    
                max_count = 0
                for i in set(dice_values):
                    count = dice_values.count(i)
                    if count > max_count:
                        max_count = count
                
                state[max_count - 1] = 1
        return state

    @staticmethod
    def get_four_of_a_kind_state(n_throws, dice_values):
        if n_throws == 0:
            return np.array([1,0,0,0])
            #if any(dice_values.count(i) == 4 for i in dice_values):
            #    state = np.array([0,0,0,1])
            #else:
            #    state = np.array([1,0,0,0])
        else:
            state = [0, 0, 0, 0]
            # Check if all dice have the same value (Yahtzee)
            if any(dice_values.count(i) >= 4 for i in dice_values):
                state = np.array([0,0,0,1])
            else:
                # Count the occurrences of each dice value    
                max_count = 0
                for i in set(dice_values):
                    count = dice_values.count(i)
                    if count > max_count:
                        max_count = count
            
                state[max_count - 1] = 1
        return state
    
    @staticmethod
    def get_full_house_state(n_throws, dice_values):
        if n_throws == 0:
            return np.array([1,0,0,0,0])
            #if any(dice_values.count(i) == 3 for i in dice_values) and any(dice_values.count(i) == 2 for i in dice_values):
            #    state = np.array([0,0,0,0,1])
            #else:
            #    state = np.array([1,0,0,0,0])

        else:
            state = [0, 0, 0, 0, 0]
            hist = [0, 0, 0, 0, 0, 0]
            for i in range(6):
                hist[i] = dice_values.count(i + 1)
            if max(hist) == 2:
                if hist.count(2) == 2:
                    state = [0, 0, 0, 1, 0] #Two Pair
                else:
                    state = [0, 1, 0, 0, 0] #Pair
            elif max(hist) >= 3:
                if 2 in hist and 3 in hist:
                    state = [0, 0, 0, 0, 1] #Full House
                else:
                    state = [0, 0, 1, 0, 0] #3 to 5 Pair
            else:
                state = [1, 0, 0, 0, 0]
        return state
    
    @staticmethod
    def get_small_straight_state(n_throws, dice_values):
        if n_throws == 0:
            return np.array([1,0,0,0])
            #if {1, 2, 3, 4}.issubset(set(dice_values)) or {2, 3, 4, 5}.issubset(set(dice_values)) or {3, 4, 5, 6}.issubset(set(dice_values)):
            #    state = np.array([0,0,0,1])
            #else:
            #    state = np.array([1,0,0,0])

        else:
            # Initialize states
            state = [0, 0, 0, 0]
            sorted_values = sorted(set(dice_values))

            if len(sorted_values) == 1: #Yahtzee
                state[0] = 1
            else:
                straights = np.array([
                    [1,2,3,4],
                    [2,3,4,5],
                    [3,4,5,6]
                ])
                max_index = 0
                for straight in straights:
                    index = 0
                    for i in sorted_values:
                        if i in straight:
                            index += 1
                        if index > max_index:
                            max_index = index

                if max_index >= len(state): #Large Straight
                    state[-1] = 1
                else:
                    state[max_index - 1] = 1
        return state
    
    @staticmethod
    def get_large_straight_state(n_throws, dice_values):
        if n_throws == 0:
            return np.array([1,0,0,0,0])
            #if {1, 2, 3, 4, 5}.issubset(set(dice_values)) or {2, 3, 4, 5, 6}.issubset(set(dice_values)):
            #    state = np.array([0,0,0,0,1])
            #else:
            #    state = np.array([1,0,0,0,0])

        else:
            # Initialize states
            state = [0, 0, 0, 0, 0]
            sorted_values = sorted(set(dice_values))

            if len(sorted_values) == 1: #Yahtzee
                state[0] = 1
            else:
                straights = np.array([
                    [1,2,3,4,5],
                    [2,3,4,5,6]
                ])
                max_index = 0
                for straight in straights:
                    index = 0
                    for i in sorted_values:
                        if i in straight:
                            index += 1
                        if index > max_index:
                            max_index = index

                if max_index >= len(state): #Large Straight
                    state[-1] = 1
                else:
                    state[max_index - 1] = 1
        return state

    @staticmethod
    def get_yahtzee_state(n_throws, dice_values):
        if n_throws == 0:
            return np.array([1,0,0,0,0])
            #if any(dice_values.count(i) == 5 for i in dice_values):
            #    state = np.array([0,0,0,0,1])
            #else:
            #    state = np.array([1,0,0,0,0])

        else:
            state = [0, 0, 0, 0, 0]
            # Check if all dice have the same value (Yahtzee)
            if len(set(dice_values)) == 1:
                state = [0, 0, 0, 0, 1]  # Yahtzee state vector

            # Count the occurrences of each dice value    
            max_count = 0
            for i in set(dice_values):
                count = dice_values.count(i)
                if count > max_count:
                    max_count = count
            
            state[max_count - 1] = 1
        return state
    
    @staticmethod
    def calculate_probabilities(n_throws, state, transition_matrix):
        probabilities = {}
        for i in range(n_throws, 3):
            state = np.dot(state, transition_matrix)
            probabilities[i] = state[-1]

        return probabilities
    
    @staticmethod
    def get_state_for_category(category, n_throws, dice_values):
        state_function_mapping = {
            'Ones': YhatzeeProbabilities.get_ones_state,
            'Twos': YhatzeeProbabilities.get_twos_state,
            'Threes': YhatzeeProbabilities.get_threes_state,
            'Fours': YhatzeeProbabilities.get_fours_state,
            'Fives': YhatzeeProbabilities.get_fives_state,
            'Sixes': YhatzeeProbabilities.get_sixes_state,
            'Three of a Kind': YhatzeeProbabilities.get_three_of_a_kind_state,
            'Four of a Kind': YhatzeeProbabilities.get_four_of_a_kind_state,
            'Full House': YhatzeeProbabilities.get_full_house_state,
            'Small Straight': YhatzeeProbabilities.get_small_straight_state,
            'Large Straight': YhatzeeProbabilities.get_large_straight_state,
            'Yahtzee': YhatzeeProbabilities.get_yahtzee_state
            # Add other categories and their state functions here
        }
        state_function = state_function_mapping.get(category)
        if state_function:
            return state_function(n_throws, dice_values)
        else:
            return None
    
    #@staticmethod
    def calculate_score_probabilities(self, n_throws, dice_values):
        transition_matrices = self.yahtzee_transition_matrix.get_transition_matrices()

        probabilities = {}
        for category, matrix in transition_matrices.items():
            state_vector = YhatzeeProbabilities.get_state_for_category(category, n_throws, dice_values)
            if state_vector is not None:
                score_probabilities = YhatzeeProbabilities.calculate_probabilities(n_throws, state_vector, matrix)
                probabilities[category] = score_probabilities

        return probabilities

if __name__ == "__main__":
    # Example for calculating probabilities for all categories
    n_throws = 1
    dices = np.array([
        [1, 2, 3, 4, 5],
        [1, 1, 1, 1, 1],
        [1, 2, 5, 6, 6],
        [1, 1, 1, 2, 2]
        ]
      )  # Example configuration
    yahtzee_probabilities = YhatzeeProbabilities()
    for dice_values in dices:
        print(dice_values)
        all_probabilities = yahtzee_probabilities.calculate_score_probabilities(n_throws, list(dice_values))
        for category, probability in all_probabilities.items():
            print(f"Probabilities for {category} category after {n_throws} throws:", probability)