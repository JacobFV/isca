import torch
import unittest

class TestPyTorchMPS(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"\nUsing device: {self.device}")

    def test_basic_operations(self):
        # Create tensors and move to device
        x = torch.rand(3, 3).to(self.device)
        y = torch.rand(3, 3).to(self.device)
        
        # Test addition
        z = x + y
        self.assertEqual(z.shape, (3, 3))
        self.assertEqual(z.device.type, self.device.type)
        
        # Test matrix multiplication
        result = torch.matmul(x, y)
        self.assertEqual(result.shape, (3, 3))
        self.assertEqual(result.device.type, self.device.type)

    def test_neural_network(self):
        # Create a simple neural network
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 1)
        ).to(self.device)
        
        # Create input tensor
        x = torch.rand(32, 10).to(self.device)  # batch of 32 samples
        
        # Forward pass
        output = model(x)
        self.assertEqual(output.shape, (32, 1))
        self.assertEqual(output.device.type, self.device.type)

if __name__ == '__main__':
    unittest.main() 