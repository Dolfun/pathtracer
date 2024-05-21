#include "renderer.h"
#include <fmt/core.h>

int main() {
  try {
    Renderer renderer;

    RenderConfig config {
      .image_width = 16,
      .image_height = 16
    };
    auto image_data = renderer.render(config);
    
  } catch (const std::exception& e) {
    fmt::println("Exception Occured: {}", e.what());
  }

  return 0;
}