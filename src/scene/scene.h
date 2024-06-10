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
    std::uint32_t material_index;
  };
  
  using VertexIndices = std::array<std::uint32_t, 3>;

  std::vector<Vertex> vertices;
  std::vector<VertexIndices> triangle_indices;
};

struct OptimizedScene {
  struct alignas(16) VertexData {
    glm::vec4 normal_and_texcoord_u;
    glm::vec4 tangent_and_texcoord_v;
    glm::vec3 bitangent;
    std::uint32_t material_index;
  };
  static_assert(alignof(VertexData) == 16);
  static_assert(sizeof(VertexData)  == 48);

  struct alignas(16) OptimizedBVHNode {
    glm::vec3 aabb_min;
    std::uint32_t left_or_begin;
    glm::vec3 aabb_max;
    std::uint32_t triangle_count;
  };
  static_assert(alignof(OptimizedBVHNode) == 16);
  static_assert(sizeof(OptimizedBVHNode)  == 32);

  std::vector<glm::vec4> vertex_positions;
  std::vector<VertexData> vertex_data;
  std::vector<OptimizedBVHNode> bvh_nodes;
};

struct BVHNode;
OptimizedScene optimize_scene(const Scene&, const std::vector<BVHNode>&);