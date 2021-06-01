#include <vector>
#include <iostream>
#include <algorithm>

#include "deepnest/network.hpp"

static const char *colors_lut[] = {
  "red", "green", "blue"
};

typedef std::vector<std::vector<double>> Matrix;

int main(void) {
  // init a network with 3 inputs
  Network network(3);
  // a hidden layer with 10 neurons and ReLU activation function
  network.add_layer(10, Activ::RELU);
  // a layer with 3 neurons (outputs)
  network.add_layer(3);

  const Matrix inputs = {
    {255, 0, 0},
    {0, 255, 0},
    {0, 0, 255},
    {250, 30, 0},
    {20, 245, 0},
    {0, 20, 255},
    {253, 20, 30},
    {20, 244, 40},
    {50, 20, 240},
  };

  const Matrix expected_outputs = {
    {1, 0, 0},
    {0, 1, 0},
    {0, 0, 1},
    {1, 0, 0},
    {0, 1, 0},
    {0, 0, 1},
    {1, 0, 0},
    {0, 1, 0},
    {0, 0, 1},
  };

  int epochs = 30;
  int batch_size = 3;
  network.train(epochs, batch_size, inputs, expected_outputs); // start fitting the network

  const Matrix colors = {
    {250, 15, 20}, // red
    {11, 254, 40}, // green
    {36, 24, 245}, // blue
  };

  const Matrix &prediction = network.predict(colors);
  for (const auto &row : prediction) {
    int index = std::distance(row.begin(), std::max_element(row.begin(), row.end()));
    std::cout << "Network predicted: " << colors_lut[index] << std::endl;
  }
  return 0;
}