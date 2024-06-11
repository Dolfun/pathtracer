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
  
  using VertexIndices = std::array<std::uint32_t, 3>;

  std::vector<Vertex> vertices;
  std::vector<VertexIndices> triangle_indices;
  std::vector<Material> materials;
};