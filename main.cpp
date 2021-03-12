#include <cstdio>
#include <iostream>

#include "network/network.h"

#define ROW_SIZE 28
#define COL_SIZE 28
#define IMAGE_SIZE (ROW_SIZE * COL_SIZE)

static int find_max_index(const Data &data) {
  if (data.size() == 0) return -1;
  int result = 0;
  double max_val = data[0];
  for (int i = 0; i < data.size(); i++) {
    if (data[i] > max_val) {
      result = i;
      max_val = data[i];
    }
  }
  return result;
}

static std::uint32_t bswap(std::uint32_t x) {
  x = (x & 0x0000FFFF) << 16 | (x & 0xFFFF0000) >> 16;
  return (x & 0x00FF00FF) << 8 | (x & 0xFF00FF00) >> 8;
}

static void load_data(const char *labels_path, const char *images_path, Data2d &inputs, Data2d &outputs, const int count) {
  std::FILE *train_labels = std::fopen(labels_path, "rb");
  std::FILE *train_images = std::fopen(images_path, "rb");
  assert(train_labels != NULL);
  assert(train_images != NULL);
  std::uint32_t magic = 0;
  assert(std::fread(&magic, sizeof(std::uint32_t), 1, train_labels) == 1);
  assert(bswap(magic) == 2049);
  assert(std::fread(&magic, sizeof(std::uint32_t), 1, train_images) == 1);
  assert(bswap(magic) == 2051);
  std::uint32_t train_labels_count = 0;
  assert(std::fread(&train_labels_count, sizeof(std::uint32_t), 1, train_labels) == 1);
  train_labels_count = bswap(train_labels_count);
  assert(train_labels_count == count);
  std::uint32_t train_images_count = 0;
  assert(std::fread(&train_images_count, sizeof(std::uint32_t), 1, train_images) == 1);
  assert(train_labels_count == bswap(train_images_count));
  std::uint32_t rows = 0;
  std::uint32_t columns = 0;
  assert(std::fread(&rows, sizeof(std::uint32_t), 1, train_images) == 1);
  assert(std::fread(&columns, sizeof(std::uint32_t), 1, train_images) == 1);
  assert(bswap(rows) == ROW_SIZE);
  assert(bswap(columns) == COL_SIZE);
  std::uint8_t label = 0;
  std::uint8_t image[IMAGE_SIZE];
  inputs.reserve(train_labels_count);
  outputs.reserve(train_labels_count);
  for (std::uint32_t i = 0; i < train_labels_count; i++) {
    assert(std::fread(&label, sizeof(std::uint8_t), 1, train_labels) == 1);
    assert(label < 10);
    assert(std::fread(image, IMAGE_SIZE, 1, train_images) == 1);
    Data input;
    input.reserve(IMAGE_SIZE);
    for (int i = 0; i < IMAGE_SIZE; i++) {
      input.push_back((double)image[i] / 255.0f);
    }
    inputs.push_back(input);
    Data output = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    output[label] = 1.0f;
    outputs.push_back(output);
  }

  std::fclose(train_labels);
  std::fclose(train_images);
}

int main(void) {
  // Network n(3);
  // n.add_layer({
  //   Neuron({0.1, 0.2, -0.1}),
  //   Neuron({-0.1, 0.1, 0.9}),
  //   Neuron({0.1, 0.4, 0.1})
  // }, Layer::RELU);
  // n.add_layer({
  //   Neuron({0.3, 1.1, -0.3}),
  //   Neuron({0.1, 0.2, 0.0}),
  //   Neuron({0.0, 1.3, 0.1})
  // });
  // n.train(1, 0.01, {
  //   {8.5, 0.65, 1.2},
  //   {9.5, 0.8, 1.3},
  //   {9.9, 0.8, 0.5},
  //   {9.0, 0.9, 1.0}
  // }, {
  //   {0.1, 1.0, 0.1},
  //   {0.0, 1.0, 0.0},
  //   {0.0, 0.0, 0.1},
  //   {0.1, 1.0, 0.2}
  // });

  Network mnist(IMAGE_SIZE);
  mnist.add_layer(100, -0.1f, 0.1f).set_activation(Activ::RELU).set_dropout(0.5f);
  mnist.add_layer(10, -0.1f, 0.1f);

  Data2d train_inputs;
  Data2d train_outputs;
  load_data("./MNIST/train-labels.idx1-ubyte", "./MNIST/train-images.idx3-ubyte", train_inputs, train_outputs, 60 * 1000);
  Data2d test_inputs;
  Data2d test_outputs;
  load_data("./MNIST/t10k-labels.idx1-ubyte", "./MNIST/t10k-images.idx3-ubyte", test_inputs, test_outputs, 10 * 1000);

  int previous_guessed = 0;
  int min_improvement = 100;
  while (true) {
    mnist.train(10, 0.005, 1, train_inputs, train_outputs);
    int guessed_right = 0;
    for (int i = 0; i < test_inputs.size(); i++) {
      Data prediction = mnist.predict(test_inputs[i]);
      int guess = find_max_index(prediction);
      guessed_right += test_outputs[i][guess] == 1.0f;
    }
    int improvement = guessed_right - previous_guessed;
    std::cout << "images guessed right: " << guessed_right << "/10000" << std::endl;
    std::cout << "improvement over last time: " << improvement << std::endl;
    if (improvement < min_improvement) {
      // only continue training if the network has guessed 100 or more
      // images than previously
      break;
    }
    previous_guessed = guessed_right;
  }
  return 0;
}