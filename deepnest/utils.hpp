#if !defined(__NETWORK_UTILS_)
#define __NETWORK_UTILS_

#include <string>
#include "network_types.hpp"

Data vec_sub(const Data &v1, const Data &v2);
Data vec_mul(const Data &v1, const Data &v2);
Data vec_mul(const Data &v, double scalar);
Data vec_div(const Data &v, double scalar);
Data vec_add(const Data &v1, const Data &v2);
double vec_reduce_sum(const Data &v);
Data2d vec_outer(const Data &v1, const Data &v2);
Data vec_mat_mul(const Data &v, const Data2d &m);
Data2d mat_mul(const Data2d &m1, double scalar);
Data2d mat_mul(const Data2d &m1, const Data2d &m2);
Data2d transpose(const Data2d &m);
Data2d mat_multiply(const Data2d &m1, const Data2d &m2);
Data2d mat_sub(const Data2d &m1, const Data2d &m2);
Data2d mat_add(const Data2d &m1, const Data2d &m2);
double mat_reduce_sum(const Data2d &m);
std::string to_str(const Data2d &m);
std::string to_str(const Data &v);

#endif // __NETWORK_UTILS_