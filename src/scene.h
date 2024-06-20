#pragma once
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <vector>
#include <array>

struct Scene {
  struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec3 tangent;
    glm::vec3 bitangnet;
    glm::vec2 texcoord;
    std::int32_t material_index;
  };

  using VertexIndices = std::array<std::uint32_t, 3>;

  struct Material {
    glm::vec4 base_color_factor;
    std::int32_t base_color_texture_index;
    float metallic_factor;
    float roughness_factor;
    std::int32_t metallic_roughness_texture_index;
    float normal_scale;
    std::int32_t normal_texture_index;
    float occlusion_strength;
    std::int32_t occlusion_texture_index;
    glm::vec3 emissive_factor;
    std::int32_t emissive_texture_index;
  };

  enum class SamplerFilter_t {
    nearest, linear
  };

  enum class SamplerWarp_t {
    clamp_to_edge, mirrored_repeat, repeat
  };

  struct Sampler {
    SamplerFilter_t mag_filter, min_filter;
    SamplerWarp_t wrap_s, wrap_t;
  };

  struct Image {
    std::uint32_t width, height, component_count;
    std::vector<unsigned char> data;
  };

  struct Texture {
    std::int32_t sampler_index;
    std::int32_t image_index;
  };

  struct alignas(16) DirectionalLight {
    glm::vec3 color;
    float intensity;
    glm::vec3 direction;
  };

  struct alignas(16) PointLight {
    glm::vec3 color;
    float intensity;
    glm::vec3 position;
  };

  struct Camera {
    glm::vec3 position;
    glm::vec3 lookat;
    float vertical_fov;
  };
  
  std::vector<Vertex> vertices;
  std::vector<VertexIndices> triangle_indices;
  std::vector<Material> materials;
  std::vector<Image> images;
  std::vector<Sampler> samplers;
  std::vector<Texture> textures;
  std::vector<DirectionalLight> directional_lights;
  std::vector<PointLight> point_lights;
  Camera camera;
};