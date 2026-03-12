#include <iostream>
#include <vector>
#include "network.h"
#include <cmath>

using namespace TensorLearn;
using namespace UnitGrad;

using ut_f32 = UnitTensor<float>;

int main() {
    nn model(2, {4, 1});

    std::vector<std::vector<ut_f32::Ptr>> X = {
        {ut_f32::make(0.0f), ut_f32::make(0.0f)},
        {ut_f32::make(0.0f), ut_f32::make(1.0f)},
        {ut_f32::make(1.0f), ut_f32::make(0.0f)},
        {ut_f32::make(1.0f), ut_f32::make(1.0f)}
    };

    std::vector<ut_f32::Ptr> Y {
        ut_f32::make(0.0f),
        ut_f32::make(1.0f),
        ut_f32::make(1.0f),
        ut_f32::make(0.0f)
    };

    float learning_rate = 0.05f;
    int epochs = 1000;

    std::cout << "Starting training loop..." << std::endl;

    for (int epoch = 0; epoch < epochs; epoch++) {
        ut_f32::Ptr total_loss = ut_f32::make(0.0f);
        for (std::size_t i = 0; i < X.size(); i++) {
            std::vector<ut_f32::Ptr> y_pred = model(X[i]);
            ut_f32::Ptr diff = y_pred[0] - Y[i]; 
            ut_f32::Ptr sq_diff = diff * diff;
            
            total_loss = total_loss + sq_diff;
        }
        model.zero_grad();
        backward(total_loss);
        for (ut_f32::Ptr p : model.parameters()) {
            p->data = p->data - (learning_rate * p->grad);
        }
        if (epoch % 100 == 0 || epoch == epochs - 1) {
            std::cout << "Epoch " << epoch << " | Loss: " << total_loss->data << "\n";
        }
    }
    std::cout << "\nTraining complete!" << std::endl;
    for (std::size_t i = 0; i < X.size(); i++) {
        std::vector<ut_f32::Ptr> pred = model(X[i]);
        std::cout << "Input: [" << X[i][0]->data << ", " << X[i][1]->data 
                  << "] -> Prediction: " << std::round(pred[0]->data)
                  << " (Target: " << Y[i]->data << ")\n";
    }

    return 0;
}
