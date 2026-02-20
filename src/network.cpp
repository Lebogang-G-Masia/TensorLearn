#include "../include/linear_layer.hpp"
#include <cstddef>

using namespace TensorLearn;

LinearLayer::LinearLayer(std::size_t num_input, std::size_t num_output) :
    a(num_input, num_output),
    w(num_input, num_output),
    b(num_input, num_output)
{}

LinearLayer::~LinearLayer() {}

LinearLayer::LinearLayer(const LinearLayer& other) :
    a(other.a),
    w(other.w),
    b(other.b)
{}

LinearLayer::LinearLayer(LinearLayer&& other) noexcept :
    a(std::move(other.a)),
    w(std::move(other.w)),
    b(std::move(other.b))
{}


LinearLayer& LinearLayer::operator=(const LinearLayer& other) {
    if (this == &other) return *this;
    a = other.a;
    w = other.w;
    b = other.b;
    return *this;
}

LinearLayer& LinearLayer::operator=(LinearLayer&& other) noexcept {
    if (this == &other) return *this;
    a = std::move(other.a);
    w = std::move(other.w);
    b = std::move(other.b);
    return *this;
}


