#include "renderer.h"
#include <fmt/core.h>
#include <stb_image_write.h>

template <typename T>
void save_image(int width, int height, const std::vector<T>& image) {
  stbi_write_bmp("output.bmp", width, height, NR_CHANNELS, static_cast<const void*>(image.data()));
}

int main() {
  try {
    Renderer renderer;

    RenderConfig config {
      .image_width = 16,
      .image_height = 16
    };
    auto image_data = renderer.render(config);

    save_image(config.image_width, config.image_height, image_data);
    
  } catch (const std::exception& e) {
    fmt::println("Exception Occured: {}", e.what());
  }

  return 0;
}