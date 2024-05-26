#include "renderer.h"
#include <fmt/core.h>
#include <stb_image_write.h>
#include "timeit.h"

int main() {
  try {
    std::unique_ptr<Renderer> renderer;

    timeit("Initialization", [&] { 
      renderer = std::make_unique<Renderer>(); 
    });

    RenderConfig config {
      .image_width = 1920,
      .image_height = 1080
    };

    const float* data;
    std::size_t size;
    timeit("Rendering", [&] { 
      std::tie(data, size) = renderer->render(config);
    });

    std::vector<uint8_t> image(size);
    timeit("Image format conversion", [&] { 
      for (auto i = 0z; i < size; ++i) {
        image[i] = static_cast<uint8_t>(data[i] * 255.999f);
      }
    });
    
    timeit("Image saving", [&] { 
      stbi_write_bmp("output.bmp", config.image_width, config.image_height, NR_CHANNELS, image.data());
    });

    
  } catch (const std::exception& e) {
    fmt::println("Exception Occured: {}", e.what());
  }

  return 0;
}