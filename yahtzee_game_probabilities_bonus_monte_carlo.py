from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import numpy as np
import random
import time
from scipy.sparse import lil_matrix, dok_matrix
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import multiprocessing
from queue import Empty


class YahtzeeBonusMonteCarloSim:

    def __init__(self):
        # Generate all possible states
        self.states = []
        self.loose_score = (-1,-1,-1,-1,-1,-1)
        self.goal_score = (6,6,6,6,6,6)
        self.generate_states()
        # Create the transition matrix
        self.transition_matrix = dok_matrix((len(self.states), len(self.states)), dtype=np.uint32)  #np.zeros((len(self.states), len(self.states)), dtype=np.uint32)#  
        self.current_dice = [0]*5
        self.selected_dice = [False]*5
        self.current_state = [0]*6
        self.next_state = [0]*6

        # Function to generate all possible states
    def generate_states(self):
        states = list(itertools.product(range(6), repeat=6))
        for state in states:
            if self.get_score_from_state(state) < 63:
                if min(state) == 0:
                    self.states.append(state)
        self.states.append(self.loose_score)
        self.states.append(self.goal_score)


    def throw_dice(self):
        # Roll dice for unselected dice
        for i in range(len(self.current_dice)):
            if self.selected_dice[i] == False:
                self.current_dice[i] = random.randint(1,6)

    def find_retainable_dice(self):
        for i, state in enumerate(self.current_state):
            if state > 0:
                self.next_state[i] = -1
        if max(self.next_state) <= 0:
            return None, None
        else:
            max_indices_t = np.argwhere(np.array(self.next_state) == np.amax(np.array(self.next_state))).flatten().tolist()
            max_indices = []
            for index in max_indices_t:
                max_indices.append(index)
            max_rep = np.max(self.next_state)
            return max_indices, max_rep

    def simulate_turn_from_state(self):
        if all(state > 0 for state in self.current_state):
            return False     
        
        self.selected_dice = [False] * 5
    
        for _ in range(3): 
            self.throw_dice()        
            self.next_state = self.count_repetitions()
            max_indices, max_rep = self.find_retainable_dice()

            if max_indices is None:
                continue  
            dice_to_keep_index = max_indices[-1]
                
            for i, die in enumerate(self.current_dice):
                if die == dice_to_keep_index + 1:
                    self.selected_dice[i] = True
                else:
                    self.selected_dice[i] = False
        if max_indices is None:
            return False
        else:
            self.next_state = self.current_state
            self.next_state[dice_to_keep_index] = max_rep
            return True        

    # Function to count repetitions of each number
    def count_repetitions(self):
        repetitions = [0] * 6  # Representing counts for numbers 1 through 6
        for value in self.current_dice:
            repetitions[value - 1] += 1
        return repetitions

    def get_score_from_state(self, state):
        score = 0
        for i in range(1,7):
            score += state[i - 1] * i 
        return score

    # Function to update the transition matrix
    def update_transition_matrix(self, queue):             
        num_states = len(self.states) 
        states_array = np.array(self.states)
        wait_len = len(self.states)*len(self.states)*4
        for i in range(num_states):#num_states
            while queue.qsize() > wait_len:
                time.sleep(10)
            self.current_state = list(self.states[i])
            self.next_state = [0]*6
            # Check if the score is already 63 or more
            if self.get_score_from_state(self.current_state) >= 63:
                queue.put([num_states-1, num_states-1])
                continue
            if min(self.current_state) < 0:
                queue.put([num_states-2, num_states-2])
                continue

            if self.simulate_turn_from_state() == False:
                continue           
         
            # Update transition matrix using vectorized operation
            if self.get_score_from_state(self.next_state) >= 63:
                queue.put([-1, -1])
            elif min(self.next_state) != 0:
                queue.put([-2, -2])
            else:
                # Find matching indices using NumPy's vectorized comparison
                matching_indices = np.argwhere(np.all(states_array == self.next_state, axis=1)).flatten()[0]
                queue.put([i, matching_indices])
            #shared_transition_matrix[i, matching_indices] = shared_transition_matrix[i, matching_indices].toarray().flatten()[0] + 1.0
            
            # Add a stop signal to indicate the end of data transmission
        queue.put(None)
        return  

    def update_transition_continuous(self, queue):
        plt.ion()
        fig, ax = plt.subplots()
        line, = ax.plot([], [], linestyle='-')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Convergence Measure')
        ax.set_title('Convergence of Transition Matrix')
        ax.grid(True)
        convergence_values = []

        while True:
            while queue.empty():
                time.sleep(0.01)
            indices = queue.get()
            if indices == 'kill':
                #self.calculate_probabilities(self.transition_matrix)
                #self.save_transition_matrix(r'C:\develop\Yahtzee\transition_matrix.pkl', self.transition_matrix)
                return
            if indices is None:
                convergence_values.append(self.calculate_convergence())
                line.set_xdata(range(1, len(convergence_values) + 1))
                line.set_ydata(convergence_values)
                ax.relim()
                ax.autoscale_view()
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.01)
                #plt.ioff()
                #plt.show()
                continue     
            indices = np.array(indices)
            #for i, matching_indices in indices:
            self.transition_matrix[indices[0], indices[1]] += 1


    def parallel_update_transition_matrix(self, num_iterations=1000, batch_size=10, num_processes=4):     

        manager = multiprocessing.Manager()
        queue = manager.Queue()

        # Spawn a separate process for continuous update
        continuous_process = multiprocessing.Process(target=self.update_transition_continuous, args=(queue,))
        continuous_process.start()

        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = []

            for iteration in range(0, num_iterations, batch_size):
                batch_end = min(iteration + batch_size, num_iterations)
                batch_futures = [executor.submit(self.update_transition_matrix, queue) for _ in range(iteration, batch_end)]
                futures.extend(batch_futures)
                #future = executor.submit(self.update_transition_matrix, queue)
                #futures.append(future)

            for future in tqdm(as_completed(futures), total=num_iterations):
                # Wait for the future to complete
                future.result()
            queue.put('kill')
            while queue.empty() == False:
                time.sleep(10)
            time.sleep(1)

    
    def calculate_convergence(self):     
        # Now, analyze convergence or transitions within this ROI for scoring 30 or more in three rolls
        return self.transition_matrix[-1,-1]

    
    def calculate_probabilities(self, matrix):  
        num_states = len(self.states)       
        # Normalize transition matrix rows
        matrix = matrix.tocsr()  # Convert to CSR format for efficient row operations
        row_sums = np.array(matrix.sum(axis=1)).flatten()
        row_indices, col_indices = matrix.nonzero()

        for i in range(num_states):
            if row_sums[i] != 0:
                matrix[i, col_indices[row_indices == i]] /= row_sums[i]
        print(matrix)
    
    def save_transition_matrix(self, matrix, filename):
        with open(filename, 'wb') as file:
            pickle.dump(matrix, file)


if __name__ == "__main__":
    num_iterations = 100
    sim = YahtzeeBonusMonteCarloSim()
    sim.parallel_update_transition_matrix()

