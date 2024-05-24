#include "renderer.h"
#include <fmt/core.h>
#include <stb_image_write.h>

int main() {
  try {
    Renderer renderer;

    RenderConfig config {
      .image_width = 100,
      .image_height = 100
    };
    auto [data, size] = renderer.render(config);

    std::vector<uint8_t> image(size);
    for (auto i = 0z; i < size; ++i) {
      image[i] = static_cast<uint8_t>(data[i] * 255.999f);
    }
    
    stbi_write_bmp("output.bmp", config.image_width, config.image_height, NR_CHANNELS, image.data());

    
  } catch (const std::exception& e) {
    fmt::println("Exception Occured: {}", e.what());
  }

  return 0;
}