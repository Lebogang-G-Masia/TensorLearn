#include <stdexcept>
#include <vector>
#include <initializer_list>
#include <fstream>
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

    inline tensor::Ptr element_wise_mul(const tensor::Ptr& a, const tensor::Ptr& b) {
        tensor::Ptr out = tensor::make(a->data.element_wise(b->data));
        out->prev = {a, b};
        out->op = "element_wise_mul";
        out->_backward = [a, b, out]() {
            a->grad += out->grad.element_wise(b->data);
            b->grad += out->grad.element_wise(a->data);
        };
        return out;
    }

    inline tensor::Ptr sum_all(const tensor::Ptr& u) {
        mat_f32 s(1, 1);
        s(0, 0) = u->data.sum();
        tensor::Ptr out = tensor::make(s);
        out->prev = {u};
        out->op = "sum_all";
        out->_backward = [u, out]() {
            float g = out->grad(0, 0);
            auto shape = u->data.shape();
            for (std::size_t i = 0; i < shape[0] * shape[1]; i++) {
                u->grad.data_ptr()[i] += g;
            }
        };
        return out;
    }

    inline tensor::Ptr scale(const tensor::Ptr& u, float s) {
        tensor::Ptr out = tensor::make(u->data * (double)s);
        out->prev = {u};
        out->op = "scale";
        out->_backward = [u, s, out]() {
            u->grad += out->grad * (double)s;
        };
        return out;
    }

    inline tensor::Ptr broadcast_add(const tensor::Ptr& x, const tensor::Ptr& b) {
        auto x_shape = x->data.shape();
        auto b_shape = b->data.shape();

        if (x_shape[0] != b_shape[0]) {
            throw std::invalid_argument("Broadcast add shape mismatch: row dimensions must match.");
        }

        if (b_shape[1] == x_shape[1]) {
            return x + b; // Normal addition
        }

        if (b_shape[1] != 1) {
            throw std::invalid_argument("Broadcast add shape mismatch: bias must have 1 column for broadcasting.");
        }

        mat_f32 res_data(x_shape[0], x_shape[1]);
        for (std::size_t i = 0; i < x_shape[0]; i++) {
            for (std::size_t j = 0; j < x_shape[1]; j++) {
                res_data(i, j) = x->data(i, j) + b->data(i, 0);
            }
        }

        tensor::Ptr out = tensor::make(res_data);
        out->prev = {x, b};
        out->op = "broadcast_add";
        out->_backward = [x, b, out]() {
            x->grad += out->grad;
            // Sum across batch dimension (columns)
            mat_f32 b_grad_sum = out->grad.transpose().sum_rows().transpose();
            b->grad += b_grad_sum;
        };
        return out;
    }

    inline tensor::Ptr sigmoid(const tensor::Ptr& u) {
        auto sig_func = [](float x) -> float { return 1.0f / (1.0f + std::exp(-x)); };
        tensor::Ptr out = tensor::make(u->data.map(sig_func));
        out->prev = {u};
        out->op = "sigmoid";
        out->_backward = [u, out]() {
            auto d_sig = [](float x) -> float { 
                float s = 1.0f / (1.0f + std::exp(-x));
                return s * (1.0f - s); 
            };
            u->grad += out->grad.element_wise(u->data.map(d_sig));
        };
        return out;
    }

    inline tensor::Ptr tanh(const tensor::Ptr& u) {
        auto tanh_func = [](float x) -> float { return std::tanh(x); };
        tensor::Ptr out = tensor::make(u->data.map(tanh_func));
        out->prev = {u};
        out->op = "tanh";
        out->_backward = [u, out]() {
            auto d_tanh = [](float x) -> float { 
                float t = std::tanh(x);
                return 1.0f - t * t; 
            };
            u->grad += out->grad.element_wise(u->data.map(d_tanh));
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

            void save(const std::string& filename) const {
                std::ofstream ofs(filename, std::ios::binary);
                if (!ofs.is_open()) throw std::runtime_error("Could not open file for saving: " + filename);

                auto params = parameters();
                std::size_t num_params = params.size();
                ofs.write(reinterpret_cast<const char*>(&num_params), sizeof(num_params));

                for (const auto& p : params) {
                    auto shape = p->data.shape();
                    std::size_t r = shape[0];
                    std::size_t c = shape[1];
                    ofs.write(reinterpret_cast<const char*>(&r), sizeof(r));
                    ofs.write(reinterpret_cast<const char*>(&c), sizeof(c));
                    ofs.write(reinterpret_cast<const char*>(p->data.data_ptr()), r * c * sizeof(float));
                }
                ofs.close();
            }

            void load(const std::string& filename) {
                std::ifstream ifs(filename, std::ios::binary);
                if (!ifs.is_open()) throw std::runtime_error("Could not open file for loading: " + filename);

                std::size_t num_params;
                ifs.read(reinterpret_cast<char*>(&num_params), sizeof(num_params));

                auto params = parameters();
                if (num_params != params.size()) {
                    throw std::runtime_error("Parameter count mismatch: file contains " + std::to_string(num_params) + 
                                             ", but module expects " + std::to_string(params.size()));
                }

                for (auto& p : params) {
                    std::size_t r, c;
                    ifs.read(reinterpret_cast<char*>(&r), sizeof(r));
                    ifs.read(reinterpret_cast<char*>(&c), sizeof(c));

                    auto shape = p->data.shape();
                    if (r != shape[0] || c != shape[1]) {
                        throw std::runtime_error("Parameter dimension mismatch for one of the tensors.");
                    }

                    ifs.read(reinterpret_cast<char*>(p->data.data_ptr()), r * c * sizeof(float));
                }
                ifs.close();
            }
    };


    enum class Activation {
        None,
        ReLU,
        Sigmoid,
        Tanh
    };

    class Layer : public Module {
        public:
            tensor::Ptr w;
            tensor::Ptr b;
            std::size_t ins;
            std::size_t outs;
            Activation activation;

            Layer(std::size_t in_features, std::size_t out_features, Activation act = Activation::None) 
                : ins(in_features), outs(out_features), activation(act) {
                mat_f32 weights(out_features, in_features, true);
                mat_f32 biases(out_features, 1);
                w = tensor::make(weights);
                b = tensor::make(biases);
            }

            tensor::Ptr operator()(const tensor::Ptr& x) {
                tensor::Ptr res = broadcast_add(w*x, b);
                switch(activation) {
                    case Activation::ReLU: return relu(res);
                    case Activation::Sigmoid: return sigmoid(res);
                    case Activation::Tanh: return tanh(res);
                    default: return res;
                }
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
                for (const Layer& layer : layers) {
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
        tensor::Ptr sq = element_wise_mul(diff, diff);
        tensor::Ptr total_sum = sum_all(sq);
        auto shape = pred->data.shape();
        return scale(total_sum, 1.0f / (shape[0] * shape[1]));
    }
}
