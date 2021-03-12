#if !defined(__NETWORK_)
#define __NETWORK_

#include <vector>

typedef std::vector<double> Data;
typedef std::vector<Data> Data2d;
typedef double (*Activation)(const double &x);

typedef enum {
  NONE, RELU
} Activ;

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
    Data output;
    Data delta;
    Data2d weight_delta;
    Activation activation_func = nullptr;
    Activation activation_deriv = nullptr;
    double __dropout = 1.0f;
    bool has_dropout = false;
    Data dropout_mask;

    void activate(void);
    void dropout(void);
    void update_weight_delta(const Data &delta, const Data &input);
    void update_delta(const Data &next_delta, const Data2d &next_weights);
    void update_weights(const Data2d &new_weights);
    void set_delta(const Data &__delta);

  public:
    Layer(const std::vector<Neuron> &__neurons) : neurons(__neurons) {}

    const Data &get_delta() const;
    const Data &get_output() const;
    const Data2d &get_weight_delta() const;
    const Data &compute_output(const Data &input);
    int size(void) const;
    Data2d to_matrix(void) const;
    Layer &set_activation(Activ activation);
    Layer &set_dropout(double dropout);
};

class Network {
  friend class Layer;
  std::vector<Layer> layers;
  int input_size;

  void update_hidden_layers_deltas(const Data &network_input);
  void update_hidden_layers_weights(double alpha);

  static double relu_deriv(const double &x);

  static Data &activate(Data &data, Activation activation);

  public:
    Network(int __size) : input_size(__size) {}
    Network(int __size, const std::vector<Layer> &__layers)
      : input_size(__size), layers(__layers) {}

    Layer &add_layer(const std::vector<Neuron> &__neurons, Activ activation = Activ::NONE);
    Layer &add_layer(const Layer &__layer, Activ activation =  Activ::NONE);
    Layer &add_layer(int n, double min_weight = 0.0, double max_weight = 1.0, Activ activation =  Activ::NONE);
    void train(size_t epochs, double alpha, size_t batch_size, const Data2d &inputs, const Data2d &expected);
    Data predict(const Data &input);

    static double relu(const double &x);
};

Data vec_sub(const Data &v1, const Data &v2);
Data vec_mul(const Data &v1, const Data &v2);
Data vec_mul(const Data &v, double scalar);
Data vec_sum(const Data &v1, const Data &v2);
double vec_reduce_sum(const Data &v);
Data2d vec_outer(const Data &v1, const Data &v2);
Data vec_mat_mul(const Data &v, const Data2d &m);
Data2d mat_mul(const Data2d &m1, double scalar);
Data2d mat_mul(const Data2d &m1, const Data2d &m2);
Data2d mat_sub(const Data2d &m1, const Data2d &m2);
std::string to_str(const Data2d &m);
std::string to_str(const Data &v);

#endif // __NETWORK_