#pragma once
#include <cstdint>
#include <glm/vec3.hpp>

#define NR_CHANNELS 4

struct RenderConfig {
  std::uint32_t image_width, image_height;
  std::uint32_t seed;
  std::uint32_t nr_samples;

  struct Camera {
    glm::vec3 center;
    glm::vec3 lookat;
    glm::vec3 up;
    float vertical_fov;
  } camera;
};