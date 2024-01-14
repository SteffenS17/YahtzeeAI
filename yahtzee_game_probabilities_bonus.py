import itertools
import numpy as np
import random
import time
from scipy.sparse import lil_matrix, csr_matrix, coo_matrix, vstack, save_npz, load_npz
from tqdm import tqdm
import pickle
import gzip
import matplotlib.pyplot as plt
from tqdm import tqdm
from yahtzee_game_probabilities import YhatzeeTransitionMatrix
from memory_profiler import profile
from collections import defaultdict

class YahtzeeBonusProbabilities:

    def __init__(self):
        # Generate all possible states
        self.n_categories = 3
        self.n_dice_sides = 3
        self.max_score = np.dot(np.ones(self.n_categories) * self.n_dice_sides, range(1, self.n_categories + 1))
        self.goal_score = int(self.max_score / 2)
        self.all_states = []
        self.valid_states = []
        self.mapping = []
        self.valid_states_index = np.array([], dtype=np.uint16)
        self.invalid_index = np.array([], dtype=np.uint16) 
        self.goal_states_index = np.array([], dtype=np.uint16)
        self.loose_states_index = np.array([], dtype=np.uint16)
        self.generate_states()
        # Create the transition matrix
        #self.transition_matrix = csr_matrix((len(self.states) + 2, len(self.states) + 2), dtype=np.float16)  #np.zeros((len(self.states), len(self.states)), dtype=np.uint32)# 
        tm = YhatzeeTransitionMatrix.get_top_distribution() 
        self.probabilities = [1,0,0,0,0,0]
        for i in range(0, 3):
            self.probabilities = np.dot(self.probabilities, tm)  
        #self.probabilities = np.array(range(1,6)) #DEBUG

    def get_score_from_state(self, state):
        score = 0
        for i in range(self.n_categories):
            if state[i] != -1:
                score += state[i] * (i + 1)
        return score
    
        # Function to generate all possible states
    def generate_states(self):
        self.all_states = np.array(list(itertools.product(range(-1, self.n_dice_sides), repeat=self.n_categories))) #range(0, self.n_dice_sides)
        for i, state in enumerate(self.all_states):
            if -1 in state: #0
                #Check if state can reach goal state
                proof_state = state.copy()
                proof_state[proof_state == -1] = self.n_dice_sides -1#0
                if self.get_score_from_state(proof_state) >= self.goal_score: 
                    self.valid_states.append(np.array(state))
                    self.valid_states_index = np.hstack((self.valid_states_index, i))
                else:
                    self.loose_states_index = np.hstack((self.loose_states_index, i))
            else:
                if self.get_score_from_state(state) >= self.goal_score:
                    self.goal_states_index = np.hstack((self.goal_states_index, i))
                else:
                    self.loose_states_index = np.hstack((self.loose_states_index, i))
        self.valid_states = np.array(self.valid_states)
        self.invalid_index = np.hstack((self.loose_states_index, self.goal_states_index))
        self.mapping = np.array(range(len(self.valid_states)))


    def allowed_transitions(self, r, specific_state):
        if not -1 in specific_state: #OBSOLETE already filtered
            return np.array([]), np.array([])
        ##Filter all states, wherby the already chosen dice stay the same
        #Get all already chosen dice indices in the given state
        keep_dice_indices = np.where(np.array(specific_state) != -1)[0] #0
        if len(keep_dice_indices) > 0:
            #Filter the relevant dice values from all states
            filtered_keep_dice = self.all_states[:, keep_dice_indices]
            #Filter all states
            filtered_keep_states_indices = np.where(filtered_keep_dice == specific_state[keep_dice_indices])[0]
        else:
            filtered_keep_states_indices = np.array(range(len(self.all_states)))

        ##Filter the residual states, whereby just one dice changes
        #Get all not chosen dices indices in the given state
        variable_dice_indices = np.where(np.array(specific_state) == -1)[0] #0
        #Filter the relevant dice values from all states
        filtered_variable_dice = self.all_states[:,variable_dice_indices]
        #Find all states, which changes just by one dice to the given state from the not chosen dice
        filtered_variable_states_indices = np.where((filtered_variable_dice != -1).sum(axis=1) == 1)[0] #0

        filtered_states_indices = np.intersect1d(filtered_variable_states_indices, filtered_keep_states_indices)
        #Find out how much these values changed and get the probability
        values = self.probabilities[np.max(filtered_variable_dice[filtered_states_indices], axis = 1)]

        #Add the diagonal
        #filtered_states_indices = np.hstack((filtered_states_indices, r))
        #values = np.hstack((values, self.probabilities[0]))
        #Check if any state transition leads to the goal or loose state and add up their probabilities
        loose_state_indices = np.intersect1d(filtered_states_indices, self.loose_states_index, return_indices=True)[1]
        loose_state_prob = np.sum(values[loose_state_indices])
        goal_state_indices = np.intersect1d(filtered_states_indices, self.goal_states_index, return_indices=True)[1]
        goal_state_prob = np.sum(values[goal_state_indices])
        #Remove these columns
        invalid_indices = np.hstack((loose_state_indices, goal_state_indices))
        filtered_states_indices = np.delete(filtered_states_indices, invalid_indices)
        values = np.delete(values, invalid_indices)
        #Stack the loose and the goal state
        filtered_states_indices = np.hstack((filtered_states_indices, np.array([len(self.all_states), len(self.all_states) + 1])))
        values = np.hstack((values, np.array([loose_state_prob, goal_state_prob])))
        #Normalize the row
        values = values/np.sum(values) #DEBUG
        return filtered_states_indices, values

    def filter_transition_matrix(self, rows_to_remove, cols_to_remove):
        matrix = self.transition_matrix.tolil()

        # Remove rows efficiently in LIL format
        rows_to_remove_set = set(rows_to_remove)
        rows_to_keep = [i for i in range(matrix.shape[0]) if i not in rows_to_remove_set]
        matrix = matrix[rows_to_keep]

        # Remove columns efficiently in LIL format
        cols_to_remove_set = set(cols_to_remove)
        cols_to_keep = [j for j in range(matrix.shape[1]) if j not in cols_to_remove_set]
        matrix = matrix[:, cols_to_keep]

        return matrix.tocsr()


    @profile
    def create_probability_bonus_matrix(self, batch_size=100):
        num_valid_states = len(self.valid_states_index)
        num_all_states = len(self.all_states)
        valid_states_index = np.hstack((self.valid_states_index, np.array([num_all_states, num_all_states + 1]))) #loose state goal state

        self.transition_matrix = csr_matrix((0, num_valid_states + 2))

        for batch in tqdm(range(0, num_valid_states, batch_size), desc='Batch', total=int(num_valid_states/batch_size)):
        #for batch in range(0, num_valid_states, batch_size):
            if batch+batch_size > num_valid_states:
                valid_states_index_batch = self.valid_states_index[batch:]
            else:
                valid_states_index_batch = self.valid_states_index[batch:batch+batch_size]
            num_batch = len(valid_states_index_batch)
            data = []
            row_indices = []
            col_indices = []

            for i, row in tqdm(enumerate(valid_states_index_batch), desc='Processing rows', total=num_batch):
            #for i, row in enumerate(valid_states_index_batch):
                state = self.valid_states[i + batch]
                columns, values = self.allowed_transitions(row, state)
                #Remove invalid columns
                valid_columns = np.intersect1d(valid_states_index, columns, return_indices=True)[1] 
                valid_values = values[np.intersect1d(valid_states_index, columns, return_indices=True)[2]]

                row_indices.extend([i] * len(valid_columns))
                col_indices.extend(valid_columns)
                data.extend(valid_values)

            row_indices = np.array(row_indices)
            col_indices = np.array(col_indices)
            data = np.array(data)

            # Create the CSR matrix
            num_rows = num_batch
            num_cols = num_valid_states + 2
            transition_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(num_rows, num_cols))
            del row_indices
            del col_indices
            del data

            # Stack Transition Matrices
            self.transition_matrix = vstack([
                self.transition_matrix,
                transition_matrix
            ], format='csr')
            del transition_matrix
        # Stack loose and goal state
        self.transition_matrix = vstack([
            self.transition_matrix,
            csr_matrix(([1.0], ([0], [num_valid_states])), shape=(1, num_valid_states + 2)),
            csr_matrix(([1.0], ([0], [num_valid_states + 1])), shape=(1, num_valid_states + 2))
        ], format='csr') 
        print(self.transition_matrix.toarray())
        print('Sparsity: ', len(self.transition_matrix.data)/(self.transition_matrix.shape[0]*self.transition_matrix.shape[1])*100.0)


    def calc_initial_probabilities(self):
        states = csr_matrix(([1.0], ([0], [0])), shape=(1, self.transition_matrix.shape[1]))
        for _ in range(self.n_categories):
            start = time.time()
            states = states.dot(self.transition_matrix)
            stop = time.time()
            print(states.toarray())
            print('Matrix Vector Product Scipy Time: ', stop - start)
            print("Goal Probability: {:.2f}".format(states.toarray().flatten()[-1]*100.0))
            print("Loose Probability: {:.2f}".format(states.toarray().flatten()[-2]*100.0))


    def store_transition_matrix(self, file):
        save_npz(file, self.transition_matrix)

    def load_transition_matrix(self, file):
        self.transition_matrix = load_npz(file) 
    


if __name__ == "__main__":
    yahtzee_bonus_probabilities = YahtzeeBonusProbabilities()
    yahtzee_bonus_probabilities.create_probability_bonus_matrix()
    yahtzee_bonus_probabilities.calc_initial_probabilities()
    yahtzee_bonus_probabilities.store_transition_matrix(r'C:\develop\YahtzeeAi\tm.npz')
    #yahtzee_bonus_probabilities.load_transition_matrix(r'C:\develop\YahtzeeAi\tm.npz')
    #yahtzee_bonus_probabilities.calc_initial_probabilities()
    

        




   