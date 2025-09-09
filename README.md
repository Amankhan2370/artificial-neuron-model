# Artificial Neuron Model - McCulloch-Pitts Implementation

A Python implementation of the foundational McCulloch-Pitts neuron model that started the journey of artificial neural networks in 1943.

## About The Project

This simulator brings to life the McCulloch-Pitts neuron - the first mathematical model of how neurons compute. It's essentially a simplified brain cell that makes binary decisions based on weighted inputs. Built for my Neural Networks course, this project helped me grasp how modern AI started with such a simple yet powerful concept.

## Getting Started

### What You'll Need

- Python 3.7 or newer
- Basic understanding of neural networks (helpful but not required)

### Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/Amankhan2370/artificial-neuron-model.git
cd artificial-neuron-model
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the simulation:
```bash
python mcp_neuron.py
```

## Features & Capabilities

### Core Functionality
- **Binary Neuron Simulation**: Processes binary inputs (0/1) through weighted connections
- **Threshold Activation**: Fires when weighted sum exceeds the threshold value
- **Logic Gate Implementation**: Successfully implements AND, OR, NOT, and NAND gates
- **Decision Boundary Visualization**: Visual representation of how the neuron classifies inputs
- **Extensible Design**: Object-oriented architecture ready for network expansion

### What It Demonstrates

The simulation shows several key concepts:
1. How neurons process information through weights and thresholds
2. Computational completeness via logical operations
3. Linear separability limitations (XOR problem)
4. The foundation of modern deep learning

## How The Neuron Works

The McCulloch-Pitts neuron operates on a simple principle:

```
Input(0/1) → Weight → Sum → Threshold Check → Output(0/1)
```

**Step-by-step process:**
1. Receives binary inputs (either 0 or 1)
2. Multiplies each input by its corresponding weight
3. Sums all weighted inputs
4. Compares sum to threshold value
5. Outputs 1 if sum ≥ threshold, otherwise outputs 0

This simple mechanism can compute any logical function that's linearly separable!

## Code Structure

```
artificial-neuron-model/
│
├── mcp_neuron.py          # Main simulation with all classes
├── requirements.txt       # Python dependencies
├── README.md             # Documentation (you're here!)
└── .gitignore            # Git ignore rules
```

### Key Classes

- **MCPNeuron**: Core neuron implementation with configurable weights and threshold
- **LogicalGates**: Pre-configured neurons demonstrating logic gates
- **NeuronNetwork**: Framework for connecting multiple neurons (extensible)

## Example Output

```
==================================================
McCulloch-Pitts Neuron Simulation
==================================================

Testing AND Gate:
------------------------------
Input: (0, 0) → Output: 0 (Expected: 0) ✓
Input: (0, 1) → Output: 0 (Expected: 0) ✓
Input: (1, 0) → Output: 0 (Expected: 0) ✓
Input: (1, 1) → Output: 1 (Expected: 1) ✓
Test Result: PASSED
```

## Visualizations

The program generates decision boundary plots showing:
- How the neuron divides input space
- Which input combinations trigger firing
- The geometric interpretation of weights and threshold

## Learning Outcomes

Through this implementation, I understood:
- The biological inspiration behind artificial neurons
- How simple units can perform complex computations
- Why modern neural networks need multiple layers
- The historical significance of the McCulloch-Pitts model
- Linear separability and its implications

## Potential Enhancements

Future development ideas:
- Implement learning algorithms (Perceptron rule)
- Build multi-layer networks for XOR solution
- Add continuous activation functions
- Create interactive GUI for real-time experimentation
- Extend to pattern recognition tasks

## Technologies Used

- **Python 3.7+** - Primary programming language
- **NumPy** - Efficient numerical computations
- **Matplotlib** - Visualization and plotting

## Installation Requirements

```txt
numpy>=1.21.0
matplotlib>=3.3.0
```

## How to Contribute

Interested in extending this project? Here's how:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Contribution Ideas
- Implement additional activation functions
- Add gradient descent learning
- Build a simple pattern classifier
- Improve visualization aesthetics

## References & Acknowledgments

- Original Paper: "A Logical Calculus of Ideas Immanent in Nervous Activity" by McCulloch & Pitts (1943)
- Course: Neural Networks and Deep Learning
- Inspired by the pioneers who saw computation in biology

## Contact

**Aman Khan**  
GitHub: [@Amankhan2370](https://github.com/Amankhan2370)  
Project Link: [https://github.com/Amankhan2370/artificial-neuron-model](https://github.com/Amankhan2370/artificial-neuron-model)

## License

Distributed under the MIT License. Use freely for educational purposes!

---

<p align="center">
<i>"From simple beginnings come great things - the McCulloch-Pitts neuron proved that artificial intelligence was possible."</i>
</p>
