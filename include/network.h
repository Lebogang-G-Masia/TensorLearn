#include <stdexcept>
#include <vector>
#include <initializer_list>
#include "engine/unit.h"
#include "include/matrix.hpp"

using mat_f32 = Lingebra::Matrix<float>;

// Overload operator* for Lingebra::Matrix to support matrix multiplication via matmul
namespace Lingebra {
    template <typename F>
    Matrix<F> operator*(const Matrix<F>& l, const Matrix<F>& r) {
        return l.matmul(r);
    }
}

using tensor = UnitGrad::UnitTensor<mat_f32>;

namespace TensorLearn {
    inline tensor::Ptr transpose(const tensor::Ptr& u) {
        tensor::Ptr out = tensor::make(u->data.transpose());
        out->prev = {u};
        out->op = "transpose";
        out->_backward = [u, out]() {
            u->grad += out->grad.transpose();
        };
        return out;
    }

    class Module {
        public:
            void zero_grad() {
                for (tensor::Ptr p : parameters()) {
                    std::size_t total = p->grad.shape()[1] * p->grad.shape()[0];
                    for (std::size_t i = 0; i < total; i++)
                        p->grad.data_ptr()[i] = 0.0f;
                }
            }

            virtual std::vector<tensor::Ptr> parameters() const { return {}; }
    };

    class Layer : public Module {
        public:
            tensor::Ptr w;
            tensor::Ptr b;
            std::size_t ins;
            std::size_t outs;
            bool use_relu;

            Layer(std::size_t in_features, std::size_t out_features, bool relu_act = false) 
                : ins(in_features), outs(out_features), use_relu(relu_act) {
                mat_f32 weights(out_features, in_features, true);
                mat_f32 biases(out_features, 1);
                w = tensor::make(weights);
                b = tensor::make(biases);
            }

            tensor::Ptr operator()(const tensor::Ptr& x) {
                tensor::Ptr res = w*x + b;
                if (use_relu) return relu(res);
                return res;
            }

            std::vector<tensor::Ptr> parameters() const override {
                std::vector<tensor::Ptr> params { w, b };
                return params;
            }
    };

    class Network : public Module {
        public:
            std::vector<Layer> layers {};
            Network(std::initializer_list<Layer> seq) {
                for (std::size_t i = 0; i < seq.size(); i++) {
                    if (i != 0) {
                        std::size_t last_out = (std::data(seq) + i - 1)->outs;
                        if ((std::data(seq) + i)->ins != last_out) throw std::invalid_argument("Layer mismatch");
                    }
                    layers.push_back(*(std::data(seq) + i));
                }
            }

            std::vector<tensor::Ptr> operator()(const std::vector<tensor::Ptr>& X) {
                std::vector<tensor::Ptr> output {};
                for (tensor::Ptr x : X) {
                    tensor::Ptr out = x;
                    for (Layer& layer : layers) {
                        out = layer(out);
                    }
                    output.push_back(out);
                }
                return output;
            }

            std::vector<tensor::Ptr> parameters() const override {
                std::vector<tensor::Ptr> params {};
                for (Layer& layer : layers) {
                    std::vector<tensor::Ptr> l_params = layer.parameters();
                    params.insert(params.end(), l_params.begin(), l_params.end());
                }
                return params;
            }
    };

    class SGD {
        public:
            float lr;
            SGD(float learning_rate) : lr(learning_rate) {}
            void step(const std::vector<tensor::Ptr>& params) {
                for (auto& p : params) {
                    std::size_t total = p->data.shape()[0] * p->data.shape()[1];
                    for (std::size_t i = 0; i < total; i++) {
                        p->data.data_ptr()[i] -= lr * p->grad.data_ptr()[i];
                    }
                }
            }
    };

    inline tensor::Ptr mse_loss(const tensor::Ptr& pred, const tensor::Ptr& target) {
        tensor::Ptr diff = pred - target;
        return transpose(diff) * diff;
    }
}

