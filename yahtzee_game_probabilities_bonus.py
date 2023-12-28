import itertools
import numpy as np
import random
import time
from scipy.sparse import lil_matrix, csr_matrix, coo_matrix, vstack
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from yahtzee_game_probabilities import YhatzeeTransitionMatrix
from memory_profiler import profile

class YahtzeeBonusProbabilities:

    def __init__(self):
        # Generate all possible states
        self.n_categories = 6
        self.n_score = 6
        self.max_score = np.dot(np.ones(self.n_categories) * self.n_score, range(1, self.n_categories + 1))
        self.goal_score = int(self.max_score/2)
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
        tm = YhatzeeTransitionMatrix.get_top() 
        self.probabilities = [1,0,0,0,0]
        for i in range(0, 3):
            self.probabilities = np.dot(self.probabilities, tm)        

    def get_score_from_state(self, state):
        score = 0
        for i in range(1,self.n_categories + 1):
            score += state[i - 1] * i 
        return score
    
        # Function to generate all possible states
    def generate_states(self):
        self.all_states = np.array(list(itertools.product(range(-1, self.n_score), repeat=self.n_categories)))
        for i, state in enumerate(self.all_states):
            if -1 in state:
                #Check if state can reach goal state
                proof_state = state.copy()
                proof_state[proof_state == -1] = self.n_score
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
        #Get all not used indices in the given state
        init_indices = np.where(np.array(specific_state) == -1)[0]
        #Filter all states by these indices
        filtered_states = self.all_states[:,init_indices]
        #Find all states, which changes just by one value to the given state from the not used indices
        valid_columns = np.where((filtered_states != -1).sum(axis=1) == 1)[0]
        #Find out how much these values changed and get the probability
        values = self.probabilities[np.max(filtered_states[valid_columns], axis = 1) - 1]
        #Add the diagonal
        #valid_columns = np.hstack((valid_columns, r))
        #values = np.hstack((values, self.probabilities[0]))
        #Check if any state transition leads to the goal or loose state and add up their probabilities
        loose_state_indices = np.intersect1d(valid_columns, self.loose_states_index, return_indices=True)[1]
        loose_state_prob = np.sum(values[loose_state_indices])
        goal_state_indices = np.intersect1d(valid_columns, self.goal_states_index, return_indices=True)[1]
        goal_state_prob = np.sum(values[goal_state_indices])
        #Remove these columns
        invalid_indices = np.hstack((loose_state_indices, goal_state_indices))
        valid_columns = np.delete(valid_columns, invalid_indices)
        values = np.delete(values, invalid_indices)
        #Stack the loose and the goal state
        valid_columns = np.hstack((valid_columns, np.array([len(self.all_states), len(self.all_states) + 1])))
        values = np.hstack((values, np.array([loose_state_prob, goal_state_prob])))
        #Normalize the row
        values = values/np.sum(values)
        return valid_columns, values

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
    def create_probability_bonus_matrix_coo(self, batch_size=100):
        num_valid_states = len(self.valid_states_index)
        num_all_states = len(self.all_states)
        valid_states_index = np.hstack((self.valid_states_index, np.array([num_all_states, num_all_states + 1]))) #loose state goal state

        self.transition_matrix = coo_matrix((0, num_valid_states + 2), dtype=np.float16)

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

            # Create the COO matrix
            num_rows = num_batch
            num_cols = num_valid_states + 2
            transition_matrix = coo_matrix((data, (row_indices, col_indices)), shape=(num_rows, num_cols), dtype=np.float16)
            del row_indices
            del col_indices
            del data

            # Stack Transition Matrices
            self.transition_matrix = vstack([
                self.transition_matrix,
                transition_matrix
            ], format='coo')
            del transition_matrix
        # Stack loose and goal state
        self.transition_matrix = vstack([
            self.transition_matrix,
            coo_matrix(([1.0], ([0], [num_valid_states])), shape=(1, num_valid_states + 2), dtype=np.float16),
            coo_matrix(([1.0], ([0], [num_valid_states + 1])), shape=(1, num_valid_states + 2), dtype=np.float16)
        ], format='coo') 
        pass


    def calc_initial_probabilities(self):
        states = coo_matrix(([1.0], ([0], [0])), shape=(1, self.transition_matrix.shape[1]), dtype=np.float16)
        #states =  csr_matrix(([1.0], ([0], [0])), shape=(1, self.transition_matrix.shape[1]), dtype=np.float16)
        for _ in range(self.n_categories):
            states = states.dot(self.transition_matrix)
        print("Goal Probability: {:.2f}".format(states.toarray().flatten()[-1]*100.0))
        print("Loose Probability: {:.2f}".format(states.toarray().flatten()[-2]*100.0))


    def store_transition_matrix(self, file):
        tm = open(file, 'wb')
        pickle.dump(self.transition_matrix, tm)

    def load_transition_matrix(self, file):
        tm = open(file, 'rb')
        self.transition_matrix = pickle.load(tm)

if __name__ == "__main__":
    yahtzee_bonus_probabilities = YahtzeeBonusProbabilities()
    yahtzee_bonus_probabilities.create_probability_bonus_matrix_coo()
    yahtzee_bonus_probabilities.calc_initial_probabilities()
    yahtzee_bonus_probabilities.store_transition_matrix(r'C:\develop\Yahtzee\tm.pkl')
    yahtzee_bonus_probabilities.load_transition_matrix(r'C:\develop\Yahtzee\tm.pkl')

        




   