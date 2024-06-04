#pragma once
#include "../Scene.h"
#include <glm/vec3.hpp>

struct BVHNode {
  glm::vec3 aabb_min;
  std::uint32_t left_or_begin;
  glm::vec3 aabb_max;
  std::uint32_t triangle_count;
};

auto build_bvh(Scene&) -> std::vector<BVHNode>;