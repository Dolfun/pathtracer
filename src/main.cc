#include <random>
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
      .resolution_x = 1080,
      .resolution_y = 1920,
      .seed = dist(engine),
      .sample_count = 128,
      .bg_color = { 0.239, 0.239, 0.239 },
    };

    Scene scene;
    timeit("load_gltf", [&] {
      scene = load_gltf("sorceress.glb");
    });

    auto [image, size] = renderer->render(scene, config);
    
    timeit("stbi_write_bmp", [&] {
      stbi_write_bmp("output.bmp", config.resolution_x, config.resolution_y, NR_CHANNELS, image);
    });

    fmt::println("\nTriangle Count: {}", scene.triangle_indices.size());
    fmt::println("Material Count: {}", scene.materials.size());
    fmt::println("Image Count: {}", scene.images.size());
    fmt::println("Sampler Count: {}", scene.samplers.size());
    fmt::println("Texture Count: {}", scene.textures.size());
    fmt::println("Directional Light Count: {}", scene.directional_lights.size() - 1);
    fmt::println("Point Light Count: {}", scene.point_lights.size() - 1);
    
  } catch (const std::exception& e) {
    fmt::println("Exception Occured: {}", e.what());
  }

  return 0;
}