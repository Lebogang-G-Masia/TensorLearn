#include <vector>
#include <random>
#include <stdexcept>
#include "include/matrix.hpp"
#include "engine/unit.h"

using namespace UnitGrad;
using namespace Lingebra;

using mat_f32 = Matrix<float>; 
using tensor = UnitTensor<mat_f32>;


namespace TensorLearn {
    class Module {
        public:
            virtual ~Module() = default;
            void zero_grad() {
                for (tensor::Ptr p : parameters()) {
                    std::size_t total_elements = p->grad.shape()[0] * p->grad.shape()[1];
                    for (std::size_t i = 0; i < total_elements; i++)
                        p->grad.data_ptr()[i] = 0.0f;
                }
            }

            virtual std::vector<tensor::Ptr> parameters() const { return {}; }
    };

    class Layer : public Module {
        public:
            tensor::Ptr w;
            tensor::Ptr b;
            bool nonLinear;

            Layer (std::size_t in_features, std::size_t out_features, bool nonLinear_=true) : nonLinear(nonLinear_) {
                static std::random_device rd;
                static std::mt19937 gen(rd());
                std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

                mat_f32 weights(in_features, out_features);
                mat_f32 bias(1, out_features);

                for (std::size_t i = 0; i < in_features; i++)
                    for (std::size_t j = 0; j < out_features; j++)
                        weights(i, j) = dist(gen);

                w = tensor::make(weights);
                b = tensor::make(bias);
            }

            tensor::Ptr operator()(const tensor::Ptr& x) {
                tensor::Ptr out = tensor::make(x->data.matmul(w->data) + b->data);
                return nonLinear ? relu(out) : out;
            }

            std::vector<tensor::Ptr> parameters() const override {
                return {w, b};
            }
    };

    class nn : public Module {
        public:
            std::vector<Layer> layers {};

            nn (std::size_t in_features, const std::vector<std::size_t>& out_features) {
                std::vector<std::size_t> sizes = {in_features};
                sizes.insert(sizes.end(), out_features.begin(), out_features.end());

                for (std::size_t i = 0; i < out_features.size(); i++) {
                    bool last = (i == out_features.size() - 1);
                    layers.push_back(Layer(sizes[i], sizes[i+1], !last));
                }
            }

            tensor::Ptr operator()(tensor::Ptr x) {
                for (std::size_t i = 0; i < layers.size(); i++)
                    x = layers[i](x);
                return x;
            }

            std::vector<tensor::Ptr> parameters() const override {
                std::vector<tensor::Ptr> params {};
                for (std::size_t i = 0; i < layers.size(); i++) {
                    std::vector<tensor::Ptr> l_params = layers[i].parameters();
                    params.insert(params.end(), l_params.begin(), l_params.end());
                }
                return params;
            }
    };

}
