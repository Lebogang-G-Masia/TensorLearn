#include "neuron.h"

#ifndef TENSOR_LEARN_LAYER_H
#define TENSOR_LEARN_LAYER_H

using namespace UnitGrad;

namespace TensorLearn {
    class Layer : public Module {
        public:
            std::vector<Neuron> neurons {};
            Layer(std::size_t input_features, std::size_t output_features, bool nonLinear = true) {
                for (std::size_t i = 0; i < output_features; i++) {
                    neurons.push_back(Neuron(input_features, nonLinear));
                }
            }

            std::vector<ut_f32::Ptr> operator()(const std::vector<ut_f32::Ptr>& x) {
                std::vector<ut_f32::Ptr> output {};
                output.reserve(x.size());

                for (std::size_t i = 0; i < x.size(); i++) {
                    output.push_back(neurons[i](x));
                }
                return output;
            }

            std::vector<ut_f32::Ptr> parameters() const override {
                std::vector<ut_f32::Ptr> params {};
                for (std::size_t i = 0; i < neurons.size(); i++) {
                    std::vector<ut_f32::Ptr> n_params = neurons[i].parameters();
                    params.insert(params.end(), n_params.begin(), n_params.end());
                }
                return params;
            }
    };
}

#endif // TENSOR_LEARN_LAYER_H
