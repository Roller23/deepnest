#include <vector>
#include <random>
#include <assert.h>
#include <iostream>
#include <sstream>

#include "network.h"

static double rand(double min, double max) {
  std::random_device rd;
  std::mt19937 e2(rd());
  std::uniform_real_distribution<double> dist(min, max);
  return dist(e2);
}

double Neuron::get_output(const Data &inputs) const {
  assert(inputs.size() == weights.size() && "Inputs and weights sizes must match");
  double result = 0;
  int idx = 0;
  for (auto &input : inputs) {
    result += weights[idx++] * input;
  }
  return result;
}

double Network::relu(const double &x) {
  if (x < 0.0f) return 0.0f;
  return x;
}

double Network::relu_deriv(const double &x) {
  if (x > 0.0f) return 1.0f;
  return 0.0f;
}

Data &Network::activate(Data &data, Activation activation) {
  for (auto &val : data) {
    val = activation(val);
  }
  return data;
}

const Data &Layer::compute_output(const Data &input) {
  output.clear();
  output.reserve(input.size());
  for (const Neuron &neuron : neurons) {
    output.push_back(neuron.get_output(input));
  }
  return output;
}

const Data &Layer::get_output(void) const {
  return output;
}

const Data &Layer::get_delta(void) const {
  return delta;
}

const Data2d &Layer::get_weight_delta(void) const {
  return weight_delta;
}

int Layer::size(void) const {
  return neurons.size();
}

Data2d Layer::to_matrix(void) const {
  Data2d result;
  for (const Neuron &neuron: neurons) {
    result.push_back(Data());
    Data &row = result.back();
    for (const double &val : neuron.weights) {
      row.push_back(val);
    }
  }
  return result;
}

void Layer::update_weights(const Data2d &new_weights) {
  int col_size = new_weights.size();
  for (int i = 0; i < col_size; i++) {
    const Data &row = new_weights[i];
    int row_size = row.size();
    for (int j = 0; j < row_size; j++) {
      this->neurons[i].weights[j] = new_weights[i][j];
    }
  }
}

void Layer::update_weight_delta(const Data &input, const Data &delta) {
  weight_delta = vec_outer(delta, input);
}

void Layer::update_delta(const Data &next_delta, const Data2d &next_weights) {
  delta = vec_mat_mul(next_delta, next_weights);
  if (activation_deriv != nullptr) {
    delta = vec_mul(delta, Network::activate(output, activation_deriv));
  }
}

void Layer::set_delta(const Data &__delta) {
  delta = __delta;
}

Layer &Layer::set_activation(Activ __activation) {
  if (__activation == RELU) {
    activation_func = Network::relu;
    activation_deriv = Network::relu_deriv;
  }
  return *this;
}

Layer &Layer::set_dropout(double dropout) {
  __dropout = dropout;
  dropout_mask.resize(size());
  has_dropout = true;
  return *this;
}

void Layer::activate(void) {
  assert(activation_func != nullptr);
  Network::activate(output, activation_func);  
}

void Layer::dropout(void) {
  assert(has_dropout == true);
  for (double &x : dropout_mask) {
    x = (double)(rand(0.0f, 1.0f) < __dropout);
  }
  output = vec_mul(vec_mul(output, dropout_mask), 1.0f / __dropout);
}

Layer &Network::add_layer(const std::vector<Neuron> &__neurons, Activ activation) {
  layers.emplace_back(__neurons);
  layers.back().set_activation(activation);
  return layers.back();
}

Layer &Network::add_layer(const Layer &__layer, Activ activation) {
  layers.push_back(__layer);
  layers.back().set_activation(activation);
  return layers.back();
}

Layer &Network::add_layer(int n, double min_weight, double max_weight, Activ activation) {
  int new_layer_size = this->input_size;
  if (layers.size() != 0) {
    new_layer_size = layers.back().size();
  }
  std::vector<Neuron> neurons;
  neurons.reserve(n);
  for (int i = 0; i < n; i++) {
    Data weights;
    weights.reserve(new_layer_size);
    for (int j = 0; j < new_layer_size; j++) {
      weights.push_back(rand(min_weight, max_weight));
    }
    neurons.emplace_back(weights);
  }
  layers.emplace_back(neurons);
  layers.back().set_activation(activation);
  return layers.back();
}

void Network::update_hidden_layers_deltas(const Data &network_input) {
  for (int i = layers.size() - 2; i >= 0; i--) {
    const Layer &next_layer = layers[i + 1];
    layers[i].update_delta(next_layer.get_delta(), next_layer.to_matrix());
    const Data &prev_output = i ? layers[i - 1].get_output() : network_input;
    layers[i].update_weight_delta(prev_output, layers[i].get_delta());
  }
}

void Network::update_hidden_layers_weights(double alpha) {
  for (int i = layers.size() - 2; i >= 0; i--) {
    Layer &layer = layers[i];
    Data2d new_weights = mat_sub(layer.to_matrix(), mat_mul(layer.get_weight_delta(), alpha));
    layer.update_weights(new_weights);
  }
}

void Network::train(size_t epochs, double alpha, size_t batch_size, const Data2d &inputs, const Data2d &expected) {
  if (layers.size() == 0) return;
  assert(inputs.size() == expected.size() && "Inputs vector size must match expected vector size");
  if (batch_size > inputs.size()) {
    batch_size = inputs.size();
  }
  size_t batches = (inputs.size() / batch_size) + !!(inputs.size() % batch_size);
  Data2d input_batch;
  Data2d output_batch;
  input_batch.resize(batch_size);
  output_batch.resize(batch_size);
  for (size_t epoch = 0; epoch < epochs; epoch++) {
    double error_sum = 0;
    for (size_t i = 0; i < batches; i++) {
      size_t batch_start = i * batch_size;
      size_t batch_end = (i + 1) * batch_size;
      for (size_t j = batch_start, idx = 0; j < batch_end; j++, idx++) {
        input_batch[idx] = inputs[j];
        output_batch[idx] = expected[j];
      }
      const Data2d &prev_output = layers.size() == 1 ? input_batch : input_batch;
    }
    // int inputs_size = inputs.size();
    // for (int i = 0; i < inputs_size; i++) {
    //   const Data &input = inputs[i];
    //   const Data &prev_output = layers.size() == 1 ? input : layers[layers.size() - 2].get_output();
    //   const Data &goal = expected[i];
    //   Data prediction = this->predict(input);
    //   this->layers.back().set_delta(vec_sub(prediction, goal));
    //   const Data &delta = this->layers.back().get_delta();
    //   Data error = vec_mul(delta, delta);
    //   error_sum += vec_reduce_sum(error);
    //   Data2d weight_delta = vec_outer(delta, prev_output);
    //   Data2d weights = this->layers.back().to_matrix();
    //   Data2d new_weights = mat_sub(weights, mat_mul(weight_delta, alpha));
    //   this->update_hidden_layers_deltas(input);
    //   this->layers.back().update_weights(new_weights);
    //   this->update_hidden_layers_weights(alpha);
    //   int progress = (i / (double)inputs_size) * 100.0f;
    //   std::cout << "progress: " << progress << "%\r";
    // }
    std::cout << "epoch " << (epoch + 1) << "/" << epochs << " done\n";
    std::cout << "error sum: " << error_sum << std::endl;
  }
}

Data Network::predict(const Data &input) {
  Data current_input = input;
  for (Layer &layer : layers) {
    current_input = layer.compute_output(current_input);
    if (layer.activation_func != nullptr) {
      layer.activate();
      current_input = layer.get_output();
    }
    if (layer.has_dropout) {
      layer.dropout();
      current_input = layer.get_output();
    }
  }
  return current_input;
}

Data vec_sub(const Data &v1, const Data &v2) {
  assert(v1.size() == v2.size() && "Vector sizes must match");
  Data result = v1;
  int size = v2.size();
  for (int i = 0; i < size; i++) {
    result[i] -= v2[i];
  }
  return result;
}

Data vec_mul(const Data &v1, const Data &v2) {
  assert(v1.size() == v2.size() && "Vector sizes must match");
  Data result = v1;
  int size = v2.size();
  for (int i = 0; i < size; i++) {
    result[i] *= v2[i];
  }
  return result;
}

Data vec_mul(const Data &v, double scalar) {
  Data result = v;
  int size = v.size();
  for (int i = 0; i < size; i++) {
    result[i] *= scalar;
  }
  return result;
}

Data vec_sum(const Data &v1, const Data &v2) {
  assert(v1.size() == v2.size() && "Vector sizes must match");
  Data result = v1;
  int size = v2.size();
  for (int i = 0; i < size; i++) {
    result[i] += v2[i];
  }
  return result;
}

double vec_reduce_sum(const Data &v) {
  double sum = 0;
  for (auto &val : v) {
    sum += val;
  }
  return sum;
}

Data2d vec_outer(const Data &v1, const Data &v2) {
  Data2d result;
  result.reserve(v1.size());
  for (auto &val1 : v1) {
    result.push_back(Data());
    Data &row = result.back();
    for (auto &val2 : v2) {
      row.push_back(val1 * val2);
    }
  }
  return result;
}

Data vec_mat_mul(const Data &v, const Data2d &m) {
  assert(v.size() == m.size() && "Vector size must match matrix row count");
  Data result;
  if (m.size() == 0) return result;
  const size_t columns = m[0].size();
  result.resize(columns);
  std::fill(result.begin(), result.end(), 0.0f);
  for (size_t i = 0; i < v.size(); i++) {
    for (size_t j = 0; j < columns; j++) {
      result[j] += v[i] * m[i][j];
    }
  }
  return result;
}

Data2d mat_mul(const Data2d &m1, double scalar) {
  Data2d result = m1;
  for (Data &row : result) {
    for (double &val : row) {
      val *= scalar;
    }
  }
  return result;
}

Data2d mat_mul(const Data2d &m1, const Data2d &m2) {
  assert(m1.size() == m2.size() && "Matrices have to be the same size");
  Data2d result;
  result.reserve(m1.size());
  int size = m1.size();
  for (int i = 0; i < size; i++) {
    result.emplace_back(vec_mul(m1[i], m2[i]));
  }
  return result;
}

Data2d mat_sub(const Data2d &m1, const Data2d &m2) {
  assert(m1.size() == m2.size() && "Matrices have to be the same size");
  Data2d result;
  result.reserve(m1.size());
  int size = m1.size();
  for (int i = 0; i < size; i++) {
    result.emplace_back(vec_sub(m1[i], m2[i]));
  }
  return result;
}

std::string to_str(const Data2d &m) {
  std::string result = "[\n";
  for (int i = 0; i < m.size(); i++) {
    result += "  " + to_str(m[i]);
    if (i < m.size() - 1) result += ",";
    result += "\n";
  }
  return result + "]";
}

std::string to_str(const Data &v) {
  std::stringstream result;
  result << "[";
  int size = v.size();
  for (int i = 0; i < size; i++) {
    result << v[i];
    if (i + 1 < size) {
      result << ", ";
    }
  }
  result << "]";
  return result.str();
}