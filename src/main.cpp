#include <iostream>
#include <vector>
#include <iomanip>
#include "../include/network.h"
#include "../include/mnist_loader.h"

using namespace TensorLearn;

void train_mnist() {
    std::cout << "=== TensorLearn MNIST Classification ===" << std::endl;

    // Load MNIST (subset for speed but enough to see learning)
    std::cout << "Loading MNIST..." << std::endl;
    auto ds = MNIST::load("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte", 5000);
    auto test_ds = MNIST::load("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte", 1000);

    // Network: 784 -> 128 (ReLU) -> 10 (Sigmoid)
    Layer l1(784, 128, Activation::ReLU);
    Layer l2(128, 10, Activation::Sigmoid);
    Network network({l1, l2});
    SGD optimizer(0.5f); // Increased learning rate

    int batch_size = 64;
    int epochs = 20;

    std::cout << "Starting training on " << ds.images.size() << " samples..." << std::endl;

    for (int epoch = 0; epoch < epochs; epoch++) {
        float epoch_loss = 0;
        int correct = 0;

        for (std::size_t i = 0; i < ds.images.size(); i += batch_size) {
            int current_batch = std::min(batch_size, (int)(ds.images.size() - i));
            
            mat_f32 bx(784, current_batch);
            mat_f32 by(10, current_batch);

            for (int b = 0; b < current_batch; b++) {
                for (int f = 0; f < 784; f++) bx(f, b) = ds.images[i + b][f];
                for (int c = 0; c < 10; c++) by(c, b) = (ds.labels[i + b] == c) ? 1.0f : 0.0f;
            }

            tensor::Ptr X = tensor::make(bx);
            tensor::Ptr Y = tensor::make(by);

            auto outputs = network({X});
            tensor::Ptr pred = outputs[0];
            tensor::Ptr loss = mse_loss(pred, Y);

            network.zero_grad();
            backward(loss);
            optimizer.step(network.parameters());

            epoch_loss += loss->data(0, 0);
            
            for (int b = 0; b < current_batch; b++) {
                int max_idx = 0;
                float max_val = -1.0f;
                for (int c = 0; c < 10; c++) {
                    if (pred->data(c, b) > max_val) {
                        max_val = pred->data(c, b);
                        max_idx = c;
                    }
                }
                if (max_idx == ds.labels[i + b]) correct++;
            }
        }
        std::cout << "Epoch " << std::setw(2) << epoch << " | Loss: " << std::fixed << std::setprecision(6) << epoch_loss / (ds.images.size() / batch_size)
                  << " | Accuracy: " << (float)correct / ds.images.size() * 100 << "%" << std::endl;
    }

    std::cout << "\nSaving trained model to 'mnist_model.bin'..." << std::endl;
    network.save("mnist_model.bin");

    // Evaluation
    std::cout << "Evaluating on test set..." << std::endl;
    int test_correct = 0;
    for (std::size_t i = 0; i < test_ds.images.size(); i++) {
        mat_f32 tx(784, 1);
        for (int f = 0; f < 784; f++) tx(f, 0) = test_ds.images[i][f];
        
        auto outputs = network({tensor::make(tx)});
        auto pred = outputs[0];

        int max_idx = 0;
        float max_val = -1.0f;
        for (int c = 0; c < 10; c++) {
            if (pred->data(c, 0) > max_val) {
                max_val = pred->data(c, 0);
                max_idx = c;
            }
        }
        if (max_idx == test_ds.labels[i]) test_correct++;
    }
    std::cout << "Test Accuracy: " << (float)test_correct / test_ds.images.size() * 100 << "%" << std::endl;
}

int main() {
    try {
        train_mnist();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
