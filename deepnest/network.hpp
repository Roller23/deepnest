#if !defined(__NETWORK_)
#define __NETWORK_

#include <vector>
#include <string>
#include "network_types.hpp"

class Network;
class Layer;

class Neuron {
  friend class Layer;
  friend class Network;
  protected:
    Data weights;
  public:
    Neuron(const Data &__weights) : weights(__weights) {}
    double get_output(const Data &inputs) const;
};

class Layer {
  friend class Network;
  protected:
    std::vector<Neuron> neurons;
    Data2d output;
    Data2d delta;
    Data2d weight_delta;
    Activ activation_type = Activ::NONE;
    Activation activation_func = nullptr;
    Activation activation_deriv = nullptr;
    double __dropout = 1.0f;
    bool has_dropout = false;
    Data2d dropout_mask;

    void activate(void);
    void softmax(void);
    void dropout(void);
    void update_weight_delta(const Data2d &delta, const Data2d &input);
    void update_delta(const Data2d &next_delta, const Data2d &next_weights);
    void update_weights(const Data2d &new_weights);
    void set_delta(const Data2d &__delta);
    const Data2d &compute_output(const Data2d &batch);

  public:
    Layer(const std::vector<Neuron> &__neurons) : neurons(__neurons) {}
    const Data2d &get_delta() const;
    const Data2d &get_output() const;
    const Data2d &get_weight_delta() const;
    int size(void) const;
    Data2d to_matrix(void) const;
    Layer &set_activation(Activ activation);
    Layer &set_dropout(double dropout);
};

class Network {
  private:
    friend class Layer;
    std::vector<Layer> layers;
    int network_input_size;

    void update_hidden_layers_deltas(const Data2d &network_input);
    void update_hidden_layers_weights(double alpha);

    static double relu(double x);
    static double relu_deriv(double x);

    static double tanh(double x);
    static double tanh_deriv(double x);

    static double sigmoid(double x);
    static double sigmoid_deriv(double x);

    static double softmax(double x);
    static double softmax_deriv(double x);

    static Data2d &activate(Data2d &data, Activation activation);

    Data2d __predict(const Data2d &batch, bool training);

  public:
    Network(int __size) : network_input_size(__size) {}
    Network(int __size, const std::vector<Layer> &__layers)
      : network_input_size(__size), layers(__layers) {}

    Layer &add_layer(const std::vector<Neuron> &__neurons, Activ activation = Activ::NONE);
    Layer &add_layer(const Layer &__layer, Activ activation = Activ::NONE);
    Layer &add_layer(int n, double min_weight = 0.0, double max_weight = 1.0, Activ activation = Activ::NONE);
    const Layer &get_layer(size_t idx) const;
    void train(size_t epochs, size_t batch_size, const Data2d &inputs, const Data2d &expected, double alpha = 0.000001);
    Data2d predict(const Data2d &batch);

    void save_weights(const std::string &path) const;
    void load_weights(const std::string &path);
};

#endif // __NETWORK_