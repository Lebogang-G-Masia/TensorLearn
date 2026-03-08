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

                for (std::size_t i = 0; i < neurons.size(); i++) {
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

    class nn : public Module {
        public:
            std::vector<Layer> layers {};
            nn(std::size_t input_features, const std::vector<std::size_t>& output_features) {
                std::vector<std::size_t> sizes = {input_features};
                sizes.insert(sizes.end(), output_features.begin(), output_features.end());

                for (std::size_t i = 0; i < output_features.size(); i++) {
                    bool last = (i == output_features.size() - 1);
                    layers.push_back(Layer(sizes[i], sizes[i+1], !last));
                }
            }

            std::vector<ut_f32::Ptr> operator()(std::vector<ut_f32::Ptr> x) {
                for (std::size_t i = 0; i < layers.size(); i++) {
                    x = layers[i](x);
                }
                return x;
            }

            std::vector<ut_f32::Ptr> parameters() const override {
                std::vector<ut_f32::Ptr> params {};
                for (std::size_t i = 0; i < layers.size(); i++) {
                    std::vector<ut_f32::Ptr> l_params = layers[i].parameters();
                    params.insert(params.end(), l_params.begin(), l_params.end());
                }
                return params;
            }
    };
}

#endif // TENSOR_LEARN_LAYER_H
