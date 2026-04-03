#include <iostream>
#include <vector>
#include "../include/network.h"

using namespace TensorLearn;

int main() {
    // 4 samples for XOR gate (Non-linearly separable)
    std::vector<tensor::Ptr> X = {
        tensor::make(mat_f32({{0.0f}, {0.0f}})),
        tensor::make(mat_f32({{0.0f}, {1.0f}})),
        tensor::make(mat_f32({{1.0f}, {0.0f}})),
        tensor::make(mat_f32({{1.0f}, {1.0f}}))
    };
    std::vector<tensor::Ptr> Y = {
        tensor::make(mat_f32({{0.0f}})),
        tensor::make(mat_f32({{1.0f}})),
        tensor::make(mat_f32({{1.0f}})),
        tensor::make(mat_f32({{0.0f}}))
    };

    // XOR requires at least one hidden layer with non-linearity
    Layer hidden(2, 8, true); // 2 in, 8 out, relu=true
    Layer output(8, 1);       // 8 in, 1 out

    Network network({hidden, output});
    SGD optimizer(0.05f); // Increased learning rate for XOR

    std::size_t epochs = 5000; // XOR can be harder to converge
    for (std::size_t i = 0; i < epochs; i++) {
        float epoch_loss = 0.0f;
        for (std::size_t j = 0; j < X.size(); j++) {
            // Forward
            auto outputs = network({X[j]});
            tensor::Ptr pred = outputs[0];
            
            // Loss
            tensor::Ptr loss = mse_loss(pred, Y[j]);
            epoch_loss += loss->data(0, 0);

            // Backward
            network.zero_grad();
            backward(loss);

            // Update
            optimizer.step(network.parameters());
        }
        if (i % 500 == 0) {
            std::cout << "Epoch " << i << " Loss: " << epoch_loss / X.size() << std::endl;
        }
    }

    // Test
    std::cout << "\nTesting Trained XOR Network:" << std::endl;
    auto final_outputs = network(X);
    for (std::size_t j = 0; j < X.size(); j++) {
        std::cout << "Input: [" << X[j]->data(0,0) << ", " << X[j]->data(1,0) << "] "
                  << "Pred: " << (final_outputs[j]->data(0,0) > 0.5 ? 1 : 0) 
                  << " (Raw: " << final_outputs[j]->data(0,0) << ") "
                  << "Target: " << Y[j]->data(0,0) << std::endl;
    }

    return 0;
}
