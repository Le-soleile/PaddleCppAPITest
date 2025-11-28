#include "gtest/gtest.h"
#if USE_PADDLE_API
#include "paddle/extension.h"
#endif

int main(int argc, char** argv) {  // NOLINT
  testing::InitGoogleTest(&argc, argv);

  int ret = RUN_ALL_TESTS();

  return ret;
}
