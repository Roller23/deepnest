#include <vector>
#include <random>
#include <assert.h>
#include <iostream>
#include <sstream>
#include <cmath>

#include "network.hpp"

static std::random_device rd;
static std::mt19937 e2(rd());

static double rand(double min, double max) {
  return std::uniform_real_distribution<double>(min, max)(e2);
}

static void __softmax(Data2d &data) {
  for (Data &x : data) {
    double exp_sum = 0.0f;
    for (double &val : x) {
      val = std::exp(val);
      exp_sum += val;
    }
    x = vec_div(x, exp_sum);
  }
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

double Network::relu(double x) {
  if (x < 0.0f) return 0.0f;
  return x;
}

double Network::relu_deriv(double x) {
  if (x > 0.0f) return 1.0f;
  return 0.0f;
}

double Network::tanh(double x) {
  return std::tanh(x);
}

double Network::tanh_deriv(double x) {
  return 1 - x * x;
}

double Network::softmax(double x) {
  assert(false && "softmax should not be used directly");
  return 0.0f;
}

double Network::softmax_deriv(double x) {
  assert(false && "softmax_deriv should not be used directly");
  return 0.0f;
}

double Network::sigmoid(double x) {
  return 1.0f / (1.0f + std::exp(-x));
}

double Network::sigmoid_deriv(double x) {
  return x * (1.0f - x);
}

Data2d &Network::activate(Data2d &data, Activation activation) {
  for (auto &row : data) {
    for (auto &val : row) {
      val = activation(val);
    }
  }
  return data;
}

const Data2d &Layer::compute_output(const Data2d &batch) {
  output.clear();
  output.reserve(batch.size());
  for (const Neuron &neuron : neurons) {
    Data batch_output;
    batch_output.reserve(batch.size());
    for (auto &input : batch) {
      batch_output.push_back(neuron.get_output(input));
    }
    output.push_back(batch_output);
  }
  return output;
}

const Data2d &Layer::get_output(void) const {
  return output;
}

const Data2d &Layer::get_delta(void) const {
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

void Layer::update_weight_delta(const Data2d &input, const Data2d &delta) {
  weight_delta = mat_multiply(transpose(delta), input);
}

void Layer::update_delta(const Data2d &next_delta, const Data2d &next_weights) {
  delta = mat_multiply(next_delta, next_weights);
  if (activation_deriv != nullptr && activation_type != SOFTMAX) {
    // TODO - might need to transpose it permanently
    Data2d output_T = transpose(output);
    delta = mat_mul(delta, Network::activate(output_T, activation_deriv));
  }
}

void Layer::set_delta(const Data2d &__delta) {
  delta = __delta;
}

Layer &Layer::set_activation(Activ __activation) {
  if (__activation == RELU) {
    activation_func = Network::relu;
    activation_deriv = Network::relu_deriv;
  } else if (__activation == SIGMOID) {
    activation_func = Network::sigmoid;
    activation_deriv = Network::sigmoid_deriv;
  } else if (__activation == SOFTMAX) {
    activation_func = Network::softmax;
    activation_deriv = Network::softmax_deriv;
  } else if (__activation == TANH) {
    activation_func = Network::tanh;
    activation_deriv = Network::tanh_deriv;
  }
  activation_type = __activation;
  return *this;
}

Layer &Layer::set_dropout(double dropout) {
  __dropout = dropout;
  dropout_mask.resize(this->size());
  for (Data &row : dropout_mask) {
    row.resize(neurons[0].weights.size());
  }
  has_dropout = true;
  return *this;
}

void Layer::activate(void) {
  assert(activation_func != nullptr);
  Network::activate(output, activation_func);
}

void Layer::dropout(void) {
  assert(has_dropout == true);
  for (Data &row : dropout_mask) {
    for (double &x : row) {
      x = (double)(rand(0.0f, 1.0f) < __dropout);
    }
  }
  output = mat_mul(mat_mul(output, dropout_mask), 1.0f / __dropout);
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
  int new_layer_size = network_input_size;
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

const Layer &Network::get_layer(size_t idx) const {
  return layers[idx];
}

void Network::update_hidden_layers_deltas(const Data2d &network_input) {
  for (int i = layers.size() - 2; i >= 0; i--) {
    const Layer &next_layer = layers[i + 1];
    layers[i].update_delta(next_layer.get_delta(), next_layer.to_matrix());
    const Data2d &prev_output = i ? layers[i - 1].get_output() : network_input;
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
  assert(batch_size != 0 && "Batch size cannot be 0");
  int layers_size = layers.size();
  if (layers_size == 0) return;
  assert(inputs.size() == expected.size() && "Inputs vector size must match expected vector size");
  if (batch_size > inputs.size()) {
    batch_size = inputs.size();
  }
  size_t batches = inputs.size() / batch_size;
  Data2d input_batch;
  Data2d expected_batch;
  input_batch.resize(batch_size);
  expected_batch.resize(batch_size);
  for (size_t epoch = 0; epoch < epochs; epoch++) {
    double error_sum = 0;
    for (size_t i = 0; i < batches; i++) {
      size_t batch_start = i * batch_size;
      size_t batch_end = (i + 1) * batch_size;
      for (size_t j = batch_start, idx = 0; j < batch_end; j++, idx++) {
        input_batch[idx] = inputs[j];
        expected_batch[idx] = expected[j];
      }
      const Data2d &prev_output = layers.size() == 1 ? input_batch : layers[layers_size - 2].get_output();
      const Data2d &prediction = __predict(input_batch, true);
      const Data2d &delta = mat_sub(prediction, expected_batch);
      layers.back().set_delta(delta);
      const Data2d &error = mat_mul(delta, delta);
      error_sum += mat_reduce_sum(mat_mul(error, error));
      const Data2d &weight_delta = mat_multiply(transpose(delta), transpose(prev_output));
      const Data2d &weights = layers.back().to_matrix();
      const Data2d &new_weights = mat_sub(weights, mat_mul(weight_delta, alpha));
      update_hidden_layers_deltas(input_batch);
      layers.back().update_weights(new_weights);
      update_hidden_layers_weights(alpha);
      int progress = (i / (double)batches) * 100.0f;
      std::cout << "progress: " << progress << "% (" << i << "/" << batches << ")\r" << std::flush;
    }
    std::cout << "                                          \r";
    std::cout << "epoch " << (epoch + 1) << "/" << epochs << " done\n";
    std::cout << "error sum: " << error_sum << std::endl;
  }
}

Data2d Network::predict(const Data2d &batch) {
  return __predict(batch, false);
}

Data2d Network::__predict(const Data2d &batch, bool training) {
  Data2d current_input = batch;
  for (Layer &layer : layers) {
    layer.compute_output(current_input);
    if (layer.activation_func != nullptr && layer.activation_type != Activ::SOFTMAX) {
      layer.activate();
    }
    if (training && layer.has_dropout) {
      layer.dropout();
    }
    current_input = transpose(layer.get_output());
    if (layer.activation_type == Activ::SOFTMAX) {
      __softmax(current_input);
    }
  }
  return current_input;
}

void Network::save_weights(const std::string &path) const {
  std::FILE *file = std::fopen(path.c_str(), "wb");
  assert(file != NULL && "Could not open file");
  std::uint32_t layers_count = layers.size();
  std::fwrite(&layers_count, sizeof(layers_count), 1, file);
  for (const Layer &layer : layers) {
    std::uint32_t rows = layer.neurons.size();
    std::uint32_t cols = layer.neurons[0].weights.size();
    std::fwrite(&rows, sizeof(rows), 1, file);
    std::fwrite(&cols, sizeof(cols), 1, file);
    for (const Neuron &neuron : layer.neurons) {
      for (const double &weight : neuron.weights) {
        std::fwrite(&weight, sizeof(weight), 1, file);
      }
    }
  }
  std::fclose(file);
}

void Network::load_weights(const std::string &path) {
  std::FILE *file = std::fopen(path.c_str(), "rb");
  assert(file != NULL && "Could not open file");
  std::uint32_t layers_count = 0;
  assert(std::fread(&layers_count, sizeof(layers_count), 1, file) == 1);
  assert(layers_count == layers.size());
  for (Layer &layer : layers) {
    std::uint32_t rows = 0;
    std::uint32_t cols = 0;
    assert(std::fread(&rows, sizeof(rows), 1, file) == 1);
    assert(std::fread(&cols, sizeof(cols), 1, file) == 1);
    assert(rows == layer.neurons.size());
    assert(cols == layer.neurons[0].weights.size());
    for (Neuron &neuron : layer.neurons) {
      for (double &weight : neuron.weights) {
        assert(std::fread(&weight, sizeof(weight), 1, file));
      }
    }
  }
  std::fclose(file);
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

Data vec_div(const Data &v, double scalar) {
  Data result = v;
  int size = v.size();
  for (int i = 0; i < size; i++) {
    result[i] /= scalar;
  }
  return result;
}

Data2d transpose(const Data2d &m) {
  if (m.size() == 0) return m;
  Data2d result(m[0].size(), Data());
  for (int i = 0; i < m.size(); i++) {
    for (int j = 0; j < m[i].size(); j++) {
      result[j].push_back(m[i][j]);
    }
  }
  return result;
}

Data2d mat_multiply(const Data2d &m1, const Data2d &m2) {
  int n = m1.size();
  int m = m1[0].size();
  int p = m2[0].size();

  Data2d result(n, Data(p, 0.0f));
  for (int j = 0; j < p; j++) {
    for (int k = 0; k < m; k++) {
      for (int i = 0; i < n; i++) {
        result[i][j] += m1[i][k] * m2[k][j];
      }
    }
  }
  return result;
}

Data vec_add(const Data &v1, const Data &v2) {
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
    row.reserve(v2.size());
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
  result.resize(m1.size());
  int size = m1.size();
  for (int i = 0; i < size; i++) {
    int cols = m1[i].size();
    result[i].resize(cols);
    for (int j = 0; j < cols; j++) {
      result[i][j] = m1[i][j] * m2[i][j];
    }
  }
  return result;
}

Data2d mat_sub(const Data2d &m1, const Data2d &m2) {
  assert(m1.size() == m2.size() && "Matrices have to be the same size");
  Data2d result;
  result.resize(m1.size());
  int size = m1.size();
  for (int i = 0; i < size; i++) {
    int cols = m1[i].size();
    result[i].resize(cols);
    for (int j = 0; j < cols; j++) {
      result[i][j] = m1[i][j] - m2[i][j];
    }
  }
  return result;
}

Data2d mat_add(const Data2d &m1, const Data2d &m2) {
  assert(m1.size() == m2.size() && "Matrices have to be the same size");
  Data2d result;
  result.resize(m1.size());
  int size = m1.size();
  for (int i = 0; i < size; i++) {
    int cols = m1[i].size();
    result[i].resize(cols);
    for (int j = 0; j < cols; j++) {
      result[i][j] = m1[i][j] + m2[i][j];
    }
  }
  return result;
}

double mat_reduce_sum(const Data2d &m) {
  double sum = 0;
  for (auto &row : m) {
    for (auto &val : row) {
      sum += val;
    }
  }
  return sum;
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