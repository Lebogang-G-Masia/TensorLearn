# TensorLearn

TensorLearn is a lightweight, modern C++ machine learning library designed for educational purposes and efficient neural network experimentation. It leverages standard C++20 features to provide a clean and intuitive API for building, training, and deploying deep learning models.

## Features

- **Modular Architecture:** Easily define layers and networks with flexible activation functions.
- **Automatic Differentiation:** Powered by `unitgrad` for seamless backpropagation and gradient computation.
- **Linear Algebra Support:** Built on top of `lingebra` for optimized matrix operations.
- **MNIST Loader:** Built-in support for loading and processing the MNIST dataset.
- **Header-Only Core:** The core library is designed to be easily integrated into existing projects.

## Dependencies

TensorLearn automatically manages its core dependencies via CMake's `FetchContent`:

- [lingebra](https://github.com/Lebogang-G-Masia/lingebra): A modern C++ linear algebra library.
- [unitgrad](https://github.com/Lebogang-G-Masia/unitgrad): A lightweight automatic differentiation engine.

## Getting Started

### Prerequisites

- C++20 compatible compiler (e.g., GCC 10+, Clang 10+, MSVC 19.28+)
- CMake 3.14 or higher

### Building the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/TensorLearn.git
   cd TensorLearn
   ```

2. Create a build directory and compile:
   ```bash
   mkdir build && cd build
   cmake ..
   make
   ```

### Running the MNIST Example

To run the provided MNIST classification example, ensure you have the MNIST dataset files in the `data/` directory:

- `data/train-images-idx3-ubyte`
- `data/train-labels-idx1-ubyte`
- `data/t10k-images-idx3-ubyte`
- `data/t10k-labels-idx1-ubyte`

Then execute the training binary:
```bash
./train
```

## Example Usage

```cpp
#include "tensorlearn/network.h"

using namespace TensorLearn;

// Define a simple 2-layer network
Layer l1(784, 128, Activation::ReLU);
Layer l2(128, 10, Activation::Sigmoid);
Network network({l1, l2});

// Training loop (Simplified)
// ... load data ...
// network.train(dataset, epochs, learning_rate);
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
