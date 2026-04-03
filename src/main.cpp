#include <iostream>
#include <vector>
#include <iomanip>
#include "../include/network.h"

using namespace TensorLearn;

void run_serialization_test() {
    std::cout << "=== TensorLearn Serialization Test ===" << std::endl;

    // XOR Data
    mat_f32 batch_x({{0.0f, 0.0f, 1.0f, 1.0f}, {0.0f, 1.0f, 0.0f, 1.0f}});
    mat_f32 batch_y({{0.0f, 1.0f, 1.0f, 0.0f}});

    tensor::Ptr X = tensor::make(batch_x);
    tensor::Ptr Y = tensor::make(batch_y);

    Layer l1(2, 8, Activation::Tanh);
    Layer l2(8, 1, Activation::None);
    Network network({l1, l2});

    // 1. Train
    std::cout << "Training model..." << std::endl;
    SGD optimizer(0.1f);
    for (std::size_t i = 0; i < 2000; i++) {
        auto outputs = network({X});
        backward(mse_loss(outputs[0], Y));
        optimizer.step(network.parameters());
        network.zero_grad();
    }

    auto pred_before = network({X})[0]->data;
    std::cout << "Loss before saving: " << mse_loss(network({X})[0], Y)->data(0,0) << std::endl;

    // 2. Save
    std::cout << "Saving model to 'xor_model.bin'..." << std::endl;
    network.save("xor_model.bin");

    // 3. Create a fresh network and Load
    std::cout << "Loading model into a fresh network..." << std::endl;
    Layer l1_new(2, 8, Activation::Tanh);
    Layer l2_new(8, 1, Activation::None);
    Network fresh_network({l1_new, l2_new});

    fresh_network.load("xor_model.bin");

    // 4. Verify
    auto pred_after = fresh_network({X})[0]->data;
    std::cout << "Loss after loading: " << mse_loss(fresh_network({X})[0], Y)->data(0,0) << std::endl;

    bool match = true;
    for(std::size_t i=0; i<4; i++) {
        if (std::abs(pred_before(0, i) - pred_after(0, i)) > 1e-6) match = false;
    }

    if (match) {
        std::cout << "\nSERIALIZATION TEST PASSED: Loaded weights produce identical results." << std::endl;
    } else {
        std::cout << "\nSERIALIZATION TEST FAILED: Results mismatch after loading." << std::endl;
    }
}

int main() {
    try {
        run_serialization_test();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
