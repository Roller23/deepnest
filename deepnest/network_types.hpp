#if !defined(__NETWORK_TYPES_)
#define __NETWORK_TYPES_

#include <vector>

typedef std::vector<double> Data;
typedef std::vector<Data> Data2d;
typedef double (*Activation)(double x);

typedef enum {
  NONE, RELU, TANH, SIGMOID, SOFTMAX
} Activ;

#endif // __NETWORK_TYPES_