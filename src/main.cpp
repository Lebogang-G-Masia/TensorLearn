#include <iostream>
#include <vector>
#include "../include/network.h"

using namespace TensorLearn;

int main() {
    // Mini-batch XOR: 4 samples in one batch
    // Shape: (input_features=2, batch_size=4)
    mat_f32 batch_x({
        {0.0f, 0.0f, 1.0f, 1.0f},
        {0.0f, 1.0f, 0.0f, 1.0f}
    });
    
    // Shape: (output_features=1, batch_size=4)
    mat_f32 batch_y({
        {0.0f, 1.0f, 1.0f, 0.0f}
    });

    tensor::Ptr X = tensor::make(batch_x);
    tensor::Ptr Y = tensor::make(batch_y);

    Layer hidden(2, 8, true);
    Layer output(8, 1);

    Network network({hidden, output});
    SGD optimizer(0.1f); // Using a slightly higher learning rate for batch

    std::size_t epochs = 5000;
    for (std::size_t i = 0; i < epochs; i++) {
        // Forward (Network currently takes vector of pointers, let's adapt)
        // We can just call output(hidden(X)) directly or wrap X in a vector
        auto outputs = network({X});
        tensor::Ptr pred = outputs[0];
        
        // Loss
        tensor::Ptr loss = mse_loss(pred, Y);

        // Backward
        network.zero_grad();
        backward(loss);

        // Update
        optimizer.step(network.parameters());

        if (i % 500 == 0) {
            std::cout << "Epoch " << i << " Loss: " << loss->data(0, 0) << std::endl;
        }
    }

    // Test
    std::cout << "\nTesting Trained XOR Network (Batch):" << std::endl;
    auto final_outputs = network({X});
    tensor::Ptr final_pred = final_outputs[0];
    
    for (std::size_t j = 0; j < 4; j++) {
        std::cout << "Input: [" << X->data(0, j) << ", " << X->data(1, j) << "] "
                  << "Pred: " << (final_pred->data(0, j) > 0.5 ? 1 : 0) 
                  << " (Raw: " << final_pred->data(0, j) << ") "
                  << "Target: " << Y->data(0, j) << std::endl;
    }

    return 0;
}
