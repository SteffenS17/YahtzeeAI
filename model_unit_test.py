import unittest
import torch
from model import ReplayBuffer, SimpleQNetwork

class TestReplayBuffer(unittest.TestCase):

    def test_replay_buffer_and_network(self):
        # Define a dummy neural network
        input_dim = 4
        output_dim = 2
        dummy_network = SimpleQNetwork(input_dim, output_dim)

        # Define a replay buffer with capacity 100
        capacity = 100
        replay_buffer = ReplayBuffer(capacity)

        # Populate the replay buffer with transitions
        transitions = [
            (torch.tensor([1, 2, 3, 4], dtype=torch.float32), 0, torch.tensor([2, 3, 4, 5], dtype=torch.float32), 0.5, False),
            (torch.tensor([0, 1, 2, 3], dtype=torch.float32), 1, torch.tensor([1, 2, 3, 4], dtype=torch.float32), -0.2, True),
            (torch.tensor([3, 4, 5, 6], dtype=torch.float32), 0, torch.tensor([4, 5, 6, 7], dtype=torch.float32), 0.8, False)
        ]

        for transition in transitions:
            replay_buffer.push(*transition)

        # Sample from the replay buffer
        batch_size = 2
        sampled_transitions = replay_buffer.sample(batch_size)

        # Extract states, actions, next_states, rewards, and dones from sampled transitions
        states, actions, next_states, rewards, dones = zip(*sampled_transitions)

        # Convert states and next_states to tensors
        states_tensor = torch.stack(states)
        next_states_tensor = torch.stack(next_states)

        # Perform a forward pass through the dummy network with the states
        output = dummy_network(states_tensor)

        # Ensure the output shape matches the expected shape
        assert output.shape == (batch_size, output_dim)

        print("Test passed successfully!")


if __name__ == '__main__':
    unittest.main()
