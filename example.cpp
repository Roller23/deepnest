#include <vector>
#include <iostream>
#include <algorithm>

#include "deepnest/network.hpp"

const char *colors_lut[] = {
  "red", "green", "blue"
};

int main(void) {
  // init a network with 3 inputs
  Network n(3);
  // a hidden layer with 10 neurons with weights between 0 and 1
  n.add_layer(10, 0, 1, Activ::RELU);
  // a layer with 3 outputs
  n.add_layer(3);

  const std::vector<std::vector<double>> inputs = {
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

  const std::vector<std::vector<double>> outputs = {
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

  // train for 100 epochs, alpha = 0.000001, batch size = 3
  n.train(100, 0.000001, 3, inputs, outputs);

  const std::vector<std::vector<double>> colors = {
    {250, 15, 20}, // red
    {11, 254, 40}, // green
    {36, 24, 245}, // blue
  };

  const std::vector<std::vector<double>> &prediction = n.predict(colors);
  for (const auto &row : prediction) {
    int index = std::distance(row.begin(), std::max_element(row.begin(), row.end()));
    std::cout << "predicted: " << colors_lut[index] << std::endl;
  }
  return 0;
}