#pragma once
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <vector>
#include <array>

struct Scene {
  struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    std::uint32_t material_index;
  };

  using VertexIndices = std::array<std::uint32_t, 3>;

  std::vector<Vertex> vertices;
  std::vector<VertexIndices> triangle_indices;
};

struct OptimizedScene {
  struct VertexData {
    glm::vec3 normal;
    std::uint32_t material_index;
  };

  struct OptimizedBVHNode {
    glm::vec3 aabb_min;
    std::uint32_t left_or_begin;
    glm::vec3 aabb_max;
    std::uint32_t triangle_count;
  };

  std::vector<glm::vec4> vertex_positions;
  std::vector<VertexData> vertex_data;
  std::vector<OptimizedBVHNode> bvh_nodes;
};

struct BVHNode;
OptimizedScene optimize_scene(const Scene&, const std::vector<BVHNode>&);