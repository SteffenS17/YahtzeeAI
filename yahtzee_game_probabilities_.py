import numpy as np


def three_of_a_kind_probability(n_throws, dice_values):

    if n_throws == 0:
        if any(dice_values.count(i) == 3 for i in dice_values):
            state = np.array([0,0,1])
        else:
            state = np.array([1,0,0])

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

    transition_matrix = np.array([
        [120/1296, 900/1296, 250/1296 + 25/1296 + 1/1296],
        [0, 2/3, 1/3], #Pair
        [0,0,1]
    ])

    probability = state
    for _ in range(n_throws, 3):
        probability = np.dot(probability, transition_matrix)

    return probability[-1]


def four_of_a_kind_probability(n_throws, dice_values):

    if n_throws == 0:
        if any(dice_values.count(i) == 4 for i in dice_values):
            state = np.array([0,0,0,1])
        else:
            state = np.array([1,0,0,0])

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

    transition_matrix = np.array([
        [120/1296, 900/1296, 250/1296, 25/1296 + 1/1296],
        [0, 120/216, 80/216, 15/216 + 1/216], #Pair
        [0,0, 4/6, 2/6], #Three of a kind
        [0,0,0,1]
    ])

    probability = state
    for _ in range(n_throws, 3):
        probability = np.dot(probability, transition_matrix)

    return probability[-1]


#Full House in one Throw prob: http://www.datagenetics.com/blog/january42012/
#Two Pair in one Throw prob:

#Two Pairs in one Throw prob:
#Combos: (1-1-2-2-1)+(1-1-2-2-2)+(1-1-2-2-3)+(1-1-2-2-4)+(1-1-2-2-5)+(1-1-2-2-6) -> 6 possible combos
#Permutations: unique(set(permutations([1,1,2,2,3]))) -> 30 Permutations
#Two Pairs Combos*Permutations*Die Faces 6*30*6=1080
#Total Die Combos: 6^5 -> 7776

#Three Pair in one Throw prob: #https://www.mathcelebrity.com/yahtzee.php?pl=3+of+a+Kind
# Probability of 3 of a kind = 3 of 1 Kind and any 4th & 5th die/All Possible Rolls
# Combos:
# Three of a kind combos of 1 = (1-1-1-1-1)+(1-1-1-1-2)+(1-1-1-1-3)+(1-1-1-1-4)+(1-1-1-1-5)+(1-1-1-1-6)
# (1-1-1-2-3)+(1-1-1-2-4)+(1-1-1-2-5)+(1-1-1-2-6)+(1-1-1-3-4)(1-1-1-3-5)+(1-1-1-3-6)+(1-1-1-4-5)+(1-1-1-4-6)+(1-1-1-5-6) = 16 possible
#Total Combos of 2 non-equal die positions: unique(set(permutations([1,1,1,2,3]))) -> 20 Permutations
#Three of a kind Combos*Permutations*Die Faces: 16*20*6=1920 #By resource 1200
#Total Die Combos: 6^5 -> 7776

#Single Pair in one Throw prob:
#1-(6*5*4*3*2)/6^5
def full_house_probability(n_throws, dice_values):
    if n_throws == 0:
        if any(dice_values.count(i) == 3 for i in dice_values) and any(dice_values.count(i) == 2 for i in dice_values):
            state = np.array([0,0,0,0,1])
        else:
            state = np.array([1,0,0,0,0])

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

#ToDo: Look up this transition matrix!
    transition_matrix = np.array([     
        [720/7776, 3600/7776, 1200/7776, 1956/7776, 300/7776],
        [0, 43/216, 108/216, 60/216, 5/216], #One Pair #0,Diff,3/6,(6-1)x3x(5-1),(6-1)/216
        [0,0, 31/36, 0, 5/36], #Three of a kind
        [0,0,0,4/6, 2/6], #Two pairs
        [0,0,0,0,1] #Full House
    ])

    probability = state
    for _ in range(n_throws, 3):
        probability = np.dot(probability, transition_matrix)

    return probability[-1]
    
#4 of a kind in first throw: 150/7776
def large_straight_probability(n_throws, dice_values):
    
    if n_throws == 0:
        if {1, 2, 3, 4, 5}.issubset(set(dice_values)) or {2, 3, 4, 5, 6}.issubset(set(dice_values)):
            state = np.array([0,0,0,0,1])
        else:
            state = np.array([1,0,0,0,0])

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

    transition_matrix = np.array([
        [16/1296, 260/1296, 660/1296, 336/1296, 24/1296],
        [0, 27/216, 111/216, 72/216, 6/216],
        [0,0,16/36, 18/36, 2/36],
        [0, 0, 0, 5/6, 1/6],
        [0, 0, 0, 0, 1]      
    ])

    probability = state
    for _ in range(n_throws, 3):
        probability = np.dot(probability, transition_matrix)

    return probability[-1]

def small_straight_probability(n_throws, dice_values):
    
    if n_throws == 0:
        if {1, 2, 3, 4}.issubset(set(dice_values)) or {2, 3, 4, 5}.issubset(set(dice_values)) or {3, 4, 5, 6}.issubset(set(dice_values)):
            state = np.array([0,0,0,1])
        else:
            state = np.array([1,0,0,0])

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


    transition_matrix = np.array([
        [108/1296, 525/1296, 582/1296, 108/1296],
        [0, 64/216, 122/216, 30/216],
        [0,0,25/36, 11/36],
        [0, 0, 0, 1]        
    ])

    probability = state
    for _ in range(n_throws, 3):
        probability = np.dot(probability, transition_matrix)

    return probability[-1]

#http://www.datagenetics.com/blog/january42012/
#https://issuu.com/milliemince/docs/using_markov_chains_and_probabilistic_modeling_to_
def yahtzee_probability(n_throws, dice_values):

    if n_throws == 0:
        if any(dice_values.count(i) == 5 for i in dice_values):
            state = np.array([0,0,0,0,1])
        else:
            state = np.array([1,0,0,0,0])

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

    transition_matrix = np.array([
        [120/1296, 900/1296, 250/1296, 25/1296, 1/1296],
        [0, 120/216, 80/216, 15/216, 1/216],
        [0,0, 25/36, 10/36, 1/36],
        [0,0,0,5/6, 1/6],
        [0,0,0,0,1]
    ])

    probability = state
    for _ in range(n_throws, 3):
        probability = np.dot(probability, transition_matrix)

    return probability[-1]



if __name__ == "__main__":
    # Yahtzee Example usage:
    dices = np.array([
        [1, 2, 3, 4, 5],
        [1, 1, 1, 1, 1],
        [1, 2, 5, 6, 6],
        [1,1,1,2,2]
        ]
      )  # Example configuration
    for current_dice in dices:
        current_dice = list(current_dice)
        for n_throws in range(4):
            probability = yahtzee_probability(n_throws, current_dice)
            print(f"Yahtzee Prob: {probability:.6f}, Throws: {n_throws}, Dice: {current_dice}")
        
        for n_throws in range(4):
            probability = small_straight_probability(n_throws, current_dice)
            print(f"Small Straight Prob: {probability:.6f}, Throws: {n_throws}, Dice: {current_dice}")

        for n_throws in range(4):
            probability = large_straight_probability(n_throws, current_dice)
            print(f"Large Straight Prob: {probability:.6f}, Throws: {n_throws}, Dice: {current_dice}")

        for n_throws in range(4):
            probability = full_house_probability(n_throws, current_dice)
            print(f"Full House Prob: {probability:.6f}, Throws: {n_throws}, Dice: {current_dice}")

        for n_throws in range(4):
            probability = four_of_a_kind_probability(n_throws, current_dice)
            print(f"Four of a kind Prob: {probability:.6f}, Throws: {n_throws}, Dice: {current_dice}")

        for n_throws in range(4):
            probability = three_of_a_kind_probability(n_throws, current_dice)
            print(f"Three of a kind Prob: {probability:.6f}, Throws: {n_throws}, Dice: {current_dice}")
    