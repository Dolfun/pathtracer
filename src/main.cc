#include <random>
#include <algorithm>
#include <execution>
#include <fmt/core.h>
#include <stb_image_write.h>
#include "renderer.h"
#include "timeit.h"
#include "gltf_loader.h"
#include "scene.h"

int main() {
  try {
    std::unique_ptr<Renderer> renderer;

    timeit("Renderer::Renderer", [&] {
      renderer = std::make_unique<Renderer>(); 
    });

    std::random_device rd;
    std::mt19937 engine { rd() };
    std::uniform_int_distribution<std::uint32_t> dist;

    RenderConfig config {
      .image_width = 1080,
      .image_height = 1920,
      .seed = dist(engine),
      .sample_count = 128,
      .bg_color = { 0.1, 0.1, 0.1 },
      .camera {
        .position = { 4.2f, 4.0f, 2.8f },
        .lookat = { 0.0f, 0.0f, 0.0f },
        .up = { 0.0f, 1.0f, 0.0f },
        .vertical_fov = 40.0f,
      },
    };

    Scene scene;
    timeit("load_gltf", [&] {
      scene = load_gltf("sorceress.glb");
    });

    auto [data, size] = renderer->render(scene, config);

    std::vector<uint8_t> image(size);
    timeit("std::transform", [&] {
      std::transform(std::execution::par, data, data + size, image.begin(), [] (const float x) {
        return static_cast<std::uint8_t>(x * 255.999f);
      });
    });
    
    timeit("stbi_write_bmp", [&] {
      stbi_write_bmp("output.bmp", config.image_width, config.image_height, NR_CHANNELS, image.data());
    });

    fmt::println("\nTriangle Count: {}", scene.triangle_indices.size());
    fmt::println("Material Count: {}", scene.materials.size());
    fmt::println("Image Count: {}", scene.images.size());
    fmt::println("Sampler Count: {}", scene.samplers.size());
    fmt::println("Texture Count: {}", scene.textures.size());
    
  } catch (const std::exception& e) {
    fmt::println("Exception Occured: {}", e.what());
  }

  return 0;
}