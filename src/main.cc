#include <random>
#include <fmt/core.h>
#include <stb_image_write.h>
#include "renderer.h"
#include "timeit.h"

int main() {
  try {
    std::unique_ptr<Renderer> renderer;

    timeit("Initialization", [&] { 
      renderer = std::make_unique<Renderer>(); 
    });

    std::random_device rd;
    std::mt19937 engine { rd() };
    std::uniform_int_distribution<std::uint32_t> dist;

    RenderConfig config {
      .image_width = 1920,
      .image_height = 1080,
      .seed = dist(engine),
      .nr_samples = 100,
      .camera {
        .center = { 0.0f, 0.6f, 1.75f },
        .lookat = { 0.0f, 0.0f, 0.0f },
        .up = { 0.0f, 1.0f, 0.0f },
        .vertical_fov = 90.0f,
      },
    };

    const float* data;
    std::size_t size;
    timeit("Rendering", [&] { 
      std::tie(data, size) = renderer->render(config);
    });

    std::vector<uint8_t> image(size);
    timeit("Image format conversion", [&] { 
      for (std::size_t i = 0; i < size; ++i) {
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