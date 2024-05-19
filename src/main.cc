#include "renderer.h"
#include <fmt/core.h>

int main() {
  try {
    Renderer renderer;
    
  } catch (const std::exception& e) {
    fmt::println("Exception Occured: {}", e.what());
  }

  return 0;
}