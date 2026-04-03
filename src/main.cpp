#include <iostream>
#include <vector>
#include <iomanip>
#include "../include/network.h"

using namespace TensorLearn;

void run_stress_test() {
    std::cout << "=== TensorLearn Deep Stress Test (XOR with Mixed Activations) ===" << std::endl;

    // XOR Data
    mat_f32 batch_x({{0.0f, 0.0f, 1.0f, 1.0f}, {0.0f, 1.0f, 0.0f, 1.0f}});
    mat_f32 batch_y({{0.0f, 1.0f, 1.0f, 0.0f}});

    tensor::Ptr X = tensor::make(batch_x);
    tensor::Ptr Y = tensor::make(batch_y);

    // Deep Network: 2 -> 8 (Tanh) -> 4 (ReLU) -> 1 (None/Linear)
    Layer l1(2, 8, Activation::Tanh);
    Layer l2(8, 4, Activation::ReLU);
    Layer l3(4, 1, Activation::None);

    Network network({l1, l2, l3});
    SGD optimizer(0.1f);

    std::size_t epochs = 3000;
    for (std::size_t i = 0; i < epochs; i++) {
        auto outputs = network({X});
        tensor::Ptr pred = outputs[0];
        tensor::Ptr loss = mse_loss(pred, Y);

        network.zero_grad();
        backward(loss);
        optimizer.step(network.parameters());

        if (i % 500 == 0) {
            std::cout << "Epoch " << std::setw(4) << i << " | Loss: " << std::fixed << std::setprecision(8) << loss->data(0, 0) << std::endl;
        }
    }

    std::cout << "\nFinal Results:" << std::endl;
    auto final_outputs = network({X});
    tensor::Ptr final_pred = final_outputs[0];
    
    bool all_passed = true;
    for (std::size_t j = 0; j < 4; j++) {
        float val = final_pred->data(0, j);
        int pred_bin = val > 0.5f ? 1 : 0;
        int target_bin = Y->data(0, j) > 0.5f ? 1 : 0;
        
        std::cout << "Input: [" << X->data(0, j) << ", " << X->data(1, j) << "] "
                  << "Raw: " << std::setw(10) << val 
                  << " | Pred: " << pred_bin 
                  << " | Target: " << target_bin;
        
        if (pred_bin == target_bin) {
            std::cout << " [PASS]" << std::endl;
        } else {
            std::cout << " [FAIL]" << std::endl;
            all_passed = false;
        }
    }

    if (all_passed) {
        std::cout << "\nSTRESS TEST PASSED: Network converged with mixed activations." << std::endl;
    } else {
        std::cout << "\nSTRESS TEST FAILED: Network failed to converge." << std::endl;
    }
}

int main() {
    try {
        run_stress_test();
    } catch (const std::exception& e) {
        std::cerr << "Runtime Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
