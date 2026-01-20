#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/ones.h>
#include <gtest/gtest.h>
#if !USE_PADDLE_API
#include <torch/all.h>
#endif

#include <vector>
#if USE_PADDLE_API
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/memory/malloc.h"
namespace phi {
inline std::ostream& operator<<(std::ostream& os, AllocationType type) {
  return os << static_cast<int>(type);
}
}  // namespace phi
#endif

namespace at {
namespace test {

class TensorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::vector<int64_t> shape = {2, 3, 4};

    tensor = at::ones(shape, at::kFloat);
    // std::cout << "tensor dim: " << tensor.dim() << std::endl;
  }

  at::Tensor tensor;
};

TEST_F(TensorTest, ConstructFromPaddleTensor) {
  EXPECT_EQ(tensor.dim(), 3);
  EXPECT_EQ(tensor.numel(), 24);  // 2*3*4
}

// 测试 data_ptr
TEST_F(TensorTest, DataPtr) {
  // Tensor tensor(paddle_tensor_);

  void* ptr = tensor.data_ptr();
  EXPECT_NE(ptr, nullptr);

  float* float_ptr = tensor.data_ptr<float>();
  EXPECT_NE(float_ptr, nullptr);
}

// 测试 strides
TEST_F(TensorTest, Strides) {
  // Tensor tensor(paddle_tensor_);

  c10::IntArrayRef strides = tensor.strides();
  EXPECT_GT(strides.size(), 0U);  // 使用无符号字面量
}

// 测试 sizes
TEST_F(TensorTest, Sizes) {
  // Tensor tensor(paddle_tensor_);

  c10::IntArrayRef sizes = tensor.sizes();
  EXPECT_EQ(sizes.size(), 3U);
  EXPECT_EQ(sizes[0], 2U);
  EXPECT_EQ(sizes[1], 3U);
  EXPECT_EQ(sizes[2], 4U);
}

// 测试 toType
TEST_F(TensorTest, ToType) {
  // Tensor tensor(paddle_tensor_);

  Tensor double_tensor = tensor.toType(c10::ScalarType::Double);
  EXPECT_EQ(double_tensor.dtype(), c10::ScalarType::Double);
}

// 测试 numel
TEST_F(TensorTest, Numel) {
  // Tensor tensor(paddle_tensor_);

  EXPECT_EQ(tensor.numel(), 24U);  // 2*3*4
}

// 测试 device
TEST_F(TensorTest, Device) {
  // Tensor tensor(paddle_tensor_);

  c10::Device device = tensor.device();
  EXPECT_EQ(device.type(), c10::DeviceType::CPU);
}

// 测试 get_device
TEST_F(TensorTest, GetDevice) {
  // Tensor tensor(paddle_tensor_);

  c10::DeviceIndex device_idx = tensor.get_device();
  EXPECT_GE(device_idx, -1);
}

// 测试 dim 和 ndimension
TEST_F(TensorTest, DimAndNdimension) {
  // Tensor tensor(paddle_tensor_);

  EXPECT_EQ(tensor.dim(), 3);
  EXPECT_EQ(tensor.ndimension(), 3);
  EXPECT_EQ(tensor.dim(), tensor.ndimension());
}

// 测试 contiguous
TEST_F(TensorTest, Contiguous) {
  // Tensor tensor(paddle_tensor_);

  at::Tensor cont_tensor = tensor.contiguous();
  EXPECT_TRUE(cont_tensor.is_contiguous());
}

// 测试 is_contiguous
TEST_F(TensorTest, IsContiguous) {
  // Tensor tensor(paddle_tensor_);

  EXPECT_TRUE(tensor.is_contiguous());
}

// 测试 scalar_type
TEST_F(TensorTest, ScalarType) {
  // Tensor tensor(paddle_tensor_);

  c10::ScalarType stype = tensor.scalar_type();
  EXPECT_EQ(stype, c10::ScalarType::Float);
}

// 测试 fill_
TEST_F(TensorTest, Fill) {
  // Tensor tensor(paddle_tensor_);

  tensor.fill_(5.0);
  float* data = tensor.data_ptr<float>();
  EXPECT_FLOAT_EQ(data[0], 5.0f);
}

// 测试 zero_
TEST_F(TensorTest, Zero) {
  // Tensor tensor(paddle_tensor_);

  tensor.zero_();
  float* data = tensor.data_ptr<float>();
  EXPECT_FLOAT_EQ(data[0], 0.0f);
}

// 测试 is_cpu
TEST_F(TensorTest, IsCpu) {
  // Tensor tensor(paddle_tensor_);

  EXPECT_TRUE(tensor.is_cpu());
}

// 测试 is_cuda (在 CPU tensor 上应该返回 false)
TEST_F(TensorTest, IsCuda) {
  // Tensor tensor(paddle_tensor_);

  EXPECT_FALSE(tensor.is_cuda());
}

// 测试 reshape
TEST_F(TensorTest, Reshape) {
  // Tensor tensor(paddle_tensor_);

  at::Tensor reshaped = tensor.reshape({6, 4});
  EXPECT_EQ(reshaped.sizes()[0], 6);
  EXPECT_EQ(reshaped.sizes()[1], 4);
  EXPECT_EQ(reshaped.numel(), 24);
}

// 测试 transpose
TEST_F(TensorTest, Transpose) {
  // Tensor tensor(paddle_tensor_);

  at::Tensor transposed = tensor.transpose(0, 2);
  EXPECT_EQ(transposed.sizes()[0], 4);
  EXPECT_EQ(transposed.sizes()[2], 2);
}

// 测试 var(bool unbiased)
TEST_F(TensorTest, VarUnbiased) {
  std::vector<int64_t> shape = {2, 3};
  at::Tensor test_tensor = at::ones(shape, at::kFloat);
  // 设置一些不同的值以便计算方差
  test_tensor.data_ptr<float>()[0] = 1.0f;
  test_tensor.data_ptr<float>()[1] = 2.0f;
  test_tensor.data_ptr<float>()[2] = 3.0f;
  test_tensor.data_ptr<float>()[3] = 4.0f;
  test_tensor.data_ptr<float>()[4] = 5.0f;
  test_tensor.data_ptr<float>()[5] = 6.0f;

  // 测试 unbiased=True (默认)
  at::Tensor var_result = test_tensor.var(true);
  EXPECT_TRUE(var_result.defined());
  EXPECT_EQ(var_result.dim(), 0);  // 标量结果

  // 测试 unbiased=False
  at::Tensor var_result_biased = test_tensor.var(false);
  EXPECT_TRUE(var_result_biased.defined());
  EXPECT_EQ(var_result_biased.dim(), 0);
}

// 测试 var(OptionalIntArrayRef dim, bool unbiased, bool keepdim)
TEST_F(TensorTest, VarDim) {
  std::vector<int64_t> shape = {2, 3};
  at::Tensor test_tensor = at::ones(shape, at::kFloat);

  for (int i = 0; i < 6; ++i) {
    test_tensor.data_ptr<float>()[i] = static_cast<float>(i + 1);
  }

  // 测试在维度 0 上计算方差
  at::Tensor var_result = test_tensor.var({0}, true, false);
  EXPECT_TRUE(var_result.defined());
  EXPECT_EQ(var_result.dim(), 1);
  EXPECT_EQ(var_result.size(0), 3);

  // 测试在维度 1 上计算方差，keepdim=true
  at::Tensor var_result_keepdim = test_tensor.var({1}, true, true);
  EXPECT_TRUE(var_result_keepdim.defined());
  EXPECT_EQ(var_result_keepdim.dim(), 2);
  EXPECT_EQ(var_result_keepdim.size(0), 2);
  EXPECT_EQ(var_result_keepdim.size(1), 1);
}

// 测试 var(OptionalIntArrayRef dim, optional<Scalar> correction, bool keepdim)
TEST_F(TensorTest, VarCorrection) {
  std::vector<int64_t> shape = {2, 3};
  at::Tensor test_tensor = at::ones(shape, at::kFloat);
  for (int i = 0; i < 6; ++i) {
    test_tensor.data_ptr<float>()[i] = static_cast<float>(i + 1);
  }

  // 测试使用 correction=1.0 (Bessel's correction)
  at::Tensor var_result = test_tensor.var({0}, at::Scalar(1.0), false);
  EXPECT_TRUE(var_result.defined());
  EXPECT_EQ(var_result.dim(), 1);
  EXPECT_EQ(var_result.size(0), 3);

  // 测试使用 correction=0.0 (population variance)
  at::Tensor var_result_pop = test_tensor.var({0}, at::Scalar(0.0), false);
  EXPECT_TRUE(var_result_pop.defined());
  EXPECT_EQ(var_result_pop.dim(), 1);
  EXPECT_EQ(var_result_pop.size(0), 3);
}

}  // namespace test
}  // namespace at
