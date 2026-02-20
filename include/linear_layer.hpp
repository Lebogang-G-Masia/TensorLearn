#include "matrix.hpp"

#ifndef TENSOR_LEARN_LINEAR_LAYER
#define TENSOR_LEARN_LINEAR_LAYER

using namespace Lingebra;

namespace TensorLearn {
    class LinearLayer {
        private:
            // Activations
            Matrix a;
            // Weights
            Matrix w;
            // Biases
            Matrix b;
        public:
            LinearLayer(std::size_t, std::size_t);

            ~LinearLayer();
            LinearLayer(const LinearLayer&);
            LinearLayer(LinearLayer&&) noexcept;
            LinearLayer& operator=(const LinearLayer&);
            LinearLayer& operator=(LinearLayer&&) noexcept;
    };
}

#endif // TENSOR_LEARN_LINEAR_LAYER
