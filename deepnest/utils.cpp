#include <cassert>
#include <sstream>
#include <string>

#include "utils.hpp"
#include "network_types.hpp"

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