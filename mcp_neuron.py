"""
McCulloch-Pitts Neuron Simulation
A well-designed, extensible implementation of the McCulloch-Pitts artificial neuron model
Author: [Your Name]
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union, Callable


class MCPNeuron:
    """
    McCulloch-Pitts Neuron Model Implementation
    
    This class simulates a single artificial neuron based on the McCulloch-Pitts model.
    The neuron takes binary inputs, applies weights, and produces binary output
    based on a threshold activation function.
    """
    
    def __init__(self, num_inputs: int, weights: List[float] = None, threshold: float = 0.5):
        """
        Initialize the McCulloch-Pitts neuron.
        
        Args:
            num_inputs: Number of input connections to the neuron
            weights: Initial weights for each input (if None, random initialization)
            threshold: Activation threshold value
        """
        self.num_inputs = num_inputs
        self.threshold = threshold
        
        # Initialize weights
        if weights is None:
            # Random initialization between -1 and 1
            self.weights = np.random.uniform(-1, 1, num_inputs)
        else:
            if len(weights) != num_inputs:
                raise ValueError(f"Number of weights ({len(weights)}) must match number of inputs ({num_inputs})")
            self.weights = np.array(weights)
        
        # Store history for analysis
        self.input_history = []
        self.output_history = []
        
    def set_weights(self, weights: List[float]):
        """Update the neuron's weights."""
        if len(weights) != self.num_inputs:
            raise ValueError(f"Number of weights must be {self.num_inputs}")
        self.weights = np.array(weights)
        
    def set_threshold(self, threshold: float):
        """Update the neuron's threshold."""
        self.threshold = threshold
        
    def weighted_sum(self, inputs: List[int]) -> float:
        """
        Calculate the weighted sum of inputs.
        
        Args:
            inputs: Binary input values (0 or 1)
            
        Returns:
            The weighted sum of inputs
        """
        inputs = np.array(inputs)
        return np.dot(inputs, self.weights)
    
    def activation_function(self, weighted_input: float) -> int:
        """
        Step activation function (McCulloch-Pitts uses binary threshold).
        
        Args:
            weighted_input: The weighted sum of inputs
            
        Returns:
            1 if weighted_input >= threshold, 0 otherwise
        """
        return 1 if weighted_input >= self.threshold else 0
    
    def process(self, inputs: List[int]) -> int:
        """
        Process inputs through the neuron to generate output.
        
        Args:
            inputs: List of binary inputs (0 or 1)
            
        Returns:
            Binary output (0 or 1)
        """
        # Validate inputs
        if len(inputs) != self.num_inputs:
            raise ValueError(f"Expected {self.num_inputs} inputs, got {len(inputs)}")
        
        # Check if inputs are binary
        for inp in inputs:
            if inp not in [0, 1]:
                raise ValueError(f"Inputs must be binary (0 or 1), got {inp}")
        
        # Calculate weighted sum
        weighted_input = self.weighted_sum(inputs)
        
        # Apply activation function
        output = self.activation_function(weighted_input)
        
        # Store in history
        self.input_history.append(inputs)
        self.output_history.append(output)
        
        return output
    
    def get_state(self) -> dict:
        """Return current state of the neuron."""
        return {
            'weights': self.weights.tolist(),
            'threshold': self.threshold,
            'num_inputs': self.num_inputs,
            'history_length': len(self.input_history)
        }
    
    def reset_history(self):
        """Clear the input and output history."""
        self.input_history = []
        self.output_history = []
        
    def visualize_decision_boundary(self):
        """Visualize the decision boundary for 2-input neurons."""
        if self.num_inputs != 2:
            print("Visualization only available for 2-input neurons")
            return
        
        # Create a grid of points
        x = np.linspace(-0.5, 1.5, 100)
        y = np.linspace(-0.5, 1.5, 100)
        X, Y = np.meshgrid(x, y)
        
        # Calculate decision boundary: w1*x + w2*y = threshold
        if self.weights[1] != 0:
            decision_y = (self.threshold - self.weights[0] * X) / self.weights[1]
        else:
            decision_y = np.ones_like(X) * np.inf
        
        # Plot
        plt.figure(figsize=(8, 6))
        plt.contour(X, Y, self.weights[0]*X + self.weights[1]*Y, 
                   levels=[self.threshold], colors='red', linewidths=2)
        
        # Plot the four possible input combinations
        inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
        for inp in inputs:
            output = self.process(inp)
            color = 'green' if output == 1 else 'blue'
            plt.scatter(inp[0], inp[1], s=200, c=color, edgecolors='black', linewidth=2)
            plt.annotate(f'({inp[0]},{inp[1]})→{output}', 
                        (inp[0], inp[1]), xytext=(5, 5), textcoords='offset points')
        
        plt.xlim(-0.5, 1.5)
        plt.ylim(-0.5, 1.5)
        plt.xlabel('Input 1')
        plt.ylabel('Input 2')
        plt.title(f'Decision Boundary\nWeights: {self.weights}, Threshold: {self.threshold}')
        plt.grid(True, alpha=0.3)
        plt.legend(['Decision Boundary', 'Output = 0', 'Output = 1'])
        plt.show()


class LogicalGates:
    """
    Implementation of logical gates using McCulloch-Pitts neurons.
    Demonstrates the computational capability of the model.
    """
    
    @staticmethod
    def create_and_gate() -> MCPNeuron:
        """Create an AND gate using McCulloch-Pitts neuron."""
        neuron = MCPNeuron(num_inputs=2, weights=[1, 1], threshold=2)
        return neuron
    
    @staticmethod
    def create_or_gate() -> MCPNeuron:
        """Create an OR gate using McCulloch-Pitts neuron."""
        neuron = MCPNeuron(num_inputs=2, weights=[1, 1], threshold=1)
        return neuron
    
    @staticmethod
    def create_not_gate() -> MCPNeuron:
        """Create a NOT gate using McCulloch-Pitts neuron."""
        neuron = MCPNeuron(num_inputs=1, weights=[-1], threshold=-0.5)
        return neuron
    
    @staticmethod
    def create_nand_gate() -> MCPNeuron:
        """Create a NAND gate using McCulloch-Pitts neuron."""
        neuron = MCPNeuron(num_inputs=2, weights=[-1, -1], threshold=-1)
        return neuron
    
    @staticmethod
    def test_gate(gate: MCPNeuron, gate_name: str, truth_table: dict):
        """Test a logical gate implementation."""
        print(f"\nTesting {gate_name} Gate:")
        print("-" * 30)
        
        all_correct = True
        for inputs, expected in truth_table.items():
            output = gate.process(list(inputs))
            status = "✓" if output == expected else "✗"
            print(f"Input: {inputs} → Output: {output} (Expected: {expected}) {status}")
            if output != expected:
                all_correct = False
        
        print(f"Test Result: {'PASSED' if all_correct else 'FAILED'}")
        return all_correct


class NeuronNetwork:
    """
    Extended functionality for creating networks of McCulloch-Pitts neurons.
    This demonstrates the extensibility of the design.
    """
    
    def __init__(self):
        self.neurons = []
        self.connections = []
    
    def add_neuron(self, neuron: MCPNeuron):
        """Add a neuron to the network."""
        self.neurons.append(neuron)
        
    def connect(self, from_idx: int, to_idx: int, weight: float):
        """Create a connection between two neurons."""
        self.connections.append((from_idx, to_idx, weight))
    
    def process_layer(self, inputs: List[int]) -> List[int]:
        """Process inputs through a layer of neurons."""
        outputs = []
        for neuron in self.neurons:
            output = neuron.process(inputs)
            outputs.append(output)
        return outputs


def main():
    """
    Main function to demonstrate the McCulloch-Pitts neuron simulation.
    """
    print("=" * 50)
    print("McCulloch-Pitts Neuron Simulation")
    print("=" * 50)
    
    # 1. Create and test a basic neuron
    print("\n1. Basic Neuron Test")
    print("-" * 30)
    neuron = MCPNeuron(num_inputs=3, weights=[0.5, 0.3, 0.2], threshold=0.6)
    
    test_inputs = [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [1, 1, 1]
    ]
    
    for inputs in test_inputs:
        output = neuron.process(inputs)
        weighted_sum = neuron.weighted_sum(inputs)
        print(f"Inputs: {inputs} → Weighted Sum: {weighted_sum:.2f} → Output: {output}")
    
    # 2. Test Logical Gates
    print("\n2. Logical Gates Implementation")
    
    # AND Gate
    and_gate = LogicalGates.create_and_gate()
    and_truth_table = {
        (0, 0): 0,
        (0, 1): 0,
        (1, 0): 0,
        (1, 1): 1
    }
    LogicalGates.test_gate(and_gate, "AND", and_truth_table)
    
    # OR Gate
    or_gate = LogicalGates.create_or_gate()
    or_truth_table = {
        (0, 0): 0,
        (0, 1): 1,
        (1, 0): 1,
        (1, 1): 1
    }
    LogicalGates.test_gate(or_gate, "OR", or_truth_table)
    
    # NOT Gate
    not_gate = LogicalGates.create_not_gate()
    not_truth_table = {
        (0,): 1,
        (1,): 0
    }
    LogicalGates.test_gate(not_gate, "NOT", not_truth_table)
    
    # 3. Visualize decision boundaries
    print("\n3. Visualizing Decision Boundaries")
    print("-" * 30)
    print("Generating visualization for AND gate...")
    and_gate.visualize_decision_boundary()
    
    print("Generating visualization for OR gate...")
    or_gate.visualize_decision_boundary()
    
    # 4. Display neuron state
    print("\n4. Neuron State Information")
    print("-" * 30)
    print(f"AND Gate State: {and_gate.get_state()}")
    print(f"OR Gate State: {or_gate.get_state()}")
    
    # 5. Custom neuron configuration
    print("\n5. Custom Neuron Configuration")
    print("-" * 30)
    custom_neuron = MCPNeuron(num_inputs=2)
    print("Created custom neuron with random weights")
    print(f"Random weights: {custom_neuron.weights}")
    
    # Test XOR (will fail with single neuron - demonstrates limitation)
    print("\n6. XOR Gate Test (Demonstrating Limitation)")
    print("-" * 30)
    print("Note: XOR cannot be implemented with a single McCulloch-Pitts neuron")
    print("This demonstrates the linear separability limitation")
    
    # Try to approximate XOR (will fail for at least one case)
    xor_attempt = MCPNeuron(num_inputs=2, weights=[1, 1], threshold=1.5)
    xor_truth_table = {
        (0, 0): 0,
        (0, 1): 1,
        (1, 0): 1,
        (1, 1): 0
    }
    LogicalGates.test_gate(xor_attempt, "XOR (Attempt)", xor_truth_table)
    
    print("\n" + "=" * 50)
    print("Simulation Complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()