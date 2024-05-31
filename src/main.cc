#include <random>
#include <algorithm>
#include <execution>
#include <fmt/core.h>
#include <stb_image_write.h>
#include "renderer.h"
#include "timeit.h"
#include "gltf_loader.h"

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
      .image_width = 3840,
      .image_height = 2160,
      .seed = dist(engine),
      .nr_samples = 100,
      .camera {
        .center = { 0.0f, 0.6f, 1.75f },
        .lookat = { 0.0f, 0.0f, 0.0f },
        .up = { 0.0f, 1.0f, 0.0f },
        .vertical_fov = 90.0f,
      },
    };

    Scene scene = load_gltf("cube.glb");
    for (auto i = 0z; i < scene.vertices.size(); ++i) {
      fmt::println("Vertex No.: {}", i);
      auto position = scene.vertices[i].position;
      auto normal = scene.vertices[i].position;
      fmt::println("Position: {} {} {}", position.x, position.y, position.z);
      fmt::println("Normal: {} {} {}", normal.x, normal.y, normal.z);
      fmt::println("");
    }

    const float* data;
    std::size_t size;
    timeit("Rendering", [&] { 
      std::tie(data, size) = renderer->render(config);
    });

    std::vector<uint8_t> image(size);
    std::transform(std::execution::par, data, data + size, image.begin(), [] (const float x) {
      return static_cast<std::uint8_t>(x * 255.999f);
    });
    
    timeit("Saving", [&] { 
      stbi_write_bmp("output.bmp", config.image_width, config.image_height, NR_CHANNELS, image.data());
    });

    
  } catch (const std::exception& e) {
    fmt::println("Exception Occured: {}", e.what());
  }

  return 0;
}