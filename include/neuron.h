#include "unit.h"
#include <vector>
#include <random>

#ifndef TENSOR_LEARN_NEURON_H
#define TENSOR_LEARN_NEURON_H

using namespace UnitGrad;
using ut_f32 = UnitTensor<float>;

namespace TensorLearn {
    class Module {
        public:
            virtual ~Module() = default;
            void zero_grad() {
                for (ut_f32::Ptr p: parameters()) p->grad = 0.0;
            }
    
            virtual std::vector<ut_f32::Ptr> parameters() const {
                return  {};
            }
    };

    class Neuron : public Module {
        public:
            std::vector<ut_f32::Ptr> w;
            ut_f32::Ptr b;
            bool nonLinear;
    
            Neuron(std::size_t input_features, bool nonLinear_) : nonLinear(nonLinear_) {
                static std::random_device rd;
                static std::mt19937 gen(rd());
                std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
                for (std::size_t i = 0; i < input_features; i++) {
                    w.push_back(ut_f32::make(dist(gen)));
                }
                b = ut_f32::make(0.0);
            }
    
            ut_f32::Ptr operator()(const std::vector<ut_f32::Ptr>& x) {
                ut_f32::Ptr act = b;
    
                for (std::size_t i = 0; i < w.size(); i++) {
                    act = act + (w[i] * x[i]);
                }
                if (nonLinear) return relu(act);
                return act;
            }
    
            std::vector<ut_f32::Ptr> parameters() const override {
                std::vector<ut_f32::Ptr> params = w;
                params.push_back(b);
                return params;
            }
    };
}

#endif // TENSOR_LEARN_NEURON_H

