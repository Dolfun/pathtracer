#include <fstream>
#include <filesystem>
#include <fmt/core.h>
#include <nlohmann/json.hpp>
#include <stb_image_write.h>
#include "timeit.h"
#include "renderer.h"
#include "gltf_loader.h"

using json = nlohmann::json;

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string_view(argv[1]) == "list_devices") {
      list_devices();
      return 0;
    }
    
    std::string config_json_path = "config.json";
    if (argc == 2) {
      config_json_path = argv[1];
    }

    std::ifstream config_file { config_json_path };
    json config = json::parse(config_file);

    RenderConfig render_config {
      .resolution_x = config["resolution_x"],
      .resolution_y = config["resolution_y"],
      .sample_count = config["sample_count"],
      .bg_color = {
        config["bg_color"][0], 
        config["bg_color"][1], 
        config["bg_color"][2]
      },
    };

    std::uint32_t device_id = -1;
    if (config.contains("device_id")) {
      device_id = config["device_id"];
    }

    Renderer renderer { device_id };

    std::string gltf_path = config["gltf_path"];
    Scene scene;
    timeit("GLTF Parsing", [&] {
      scene = load_gltf(gltf_path);
    });

    auto [image, size] = renderer.render(scene, render_config);
    
    timeit("Saving PNG", [&] {
      std::filesystem::path path { gltf_path };
      std::string result_path = path.stem().string() + ".png";

      std::uint32_t stride = render_config.resolution_x * 4;
      stbi_write_png(
        result_path.c_str(), 
        render_config.resolution_x, 
        render_config.resolution_y, 
        NR_CHANNELS, image, stride
      );
    });
    
  } catch (const std::exception& e) {
    fmt::println("Exception Occured: {}", e.what());
  }

  return 0;
}