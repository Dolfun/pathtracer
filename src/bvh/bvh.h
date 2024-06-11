#pragma once
#include "../scene.h"
#include <glm/glm.hpp>

struct AABB {
  glm::vec3 box_min;
  glm::vec3 box_max;

  AABB() : 
    box_min {  std::numeric_limits<float>::infinity() }, 
    box_max { -std::numeric_limits<float>::infinity() } {}

  void expand(const glm::vec3& v) {
    box_max = glm::max(box_max, v);
    box_min = glm::min(box_min, v);
  }

  void expand(const std::vector<Scene::Vertex>& vertices, 
              const Scene::VertexIndices& vertex_indices) {
    for (auto i : vertex_indices) {
      expand(vertices[i].position);
    }
  }

  void expand(const AABB& other) {
    box_max = glm::max(box_max, other.box_max);
    box_min = glm::min(box_min, other.box_min);
  }

  float area() const {
    glm::vec3 e = box_max - box_min;
    return e.x * e.y + e.y * e.z + e.z * e.x;
  }
};

struct BVHNode {
  AABB aabb;
  std::uint32_t left_child_index;
  std::uint32_t begin_index;
  std::uint32_t triangle_count;
};

auto build_bvh(Scene&, std::uint32_t) -> std::vector<BVHNode>;