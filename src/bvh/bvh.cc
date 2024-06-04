#include "bvh.h"
#include <limits>
#include <algorithm>
#include <glm/glm.hpp>

#define MAX_TRIANGLES_IN_BVH_NODE 2

struct BVHNode2 {
  glm::vec3 aabb_min;
  glm::vec3 aabb_max;
  std::uint32_t left_child_index;
  std::uint32_t begin_index;
  std::uint32_t triangle_count;
};

using Triangle = Scene::Triangle;

class BVHBuilder {
public:
  BVHBuilder(std::vector<Triangle>& _triangles, std::vector<BVHNode2>& _nodes)
    : triangles { _triangles }, nodes { _nodes } {}

  void build() {
    std::uint32_t triangle_count = triangles.size();
    nodes.resize(2 * triangle_count - 1);
    node_count = 1;

    BVHNode2& root = nodes[0];
    root.left_child_index = 0;
    root.begin_index = 0;
    root.triangle_count = triangle_count;

    subdivide(root);
    nodes.resize(node_count);
  }

  void subdivide(BVHNode2& node) {
    update_bounds(node);
    if (node.triangle_count <= MAX_TRIANGLES_IN_BVH_NODE) return;

    glm::vec3 extent = node.aabb_max - node.aabb_min;
    std::uint32_t axis = 0;
    if (extent.y > extent.x) axis = 1;
    if (extent.z > extent[axis]) axis = 2;
    float e = extent[axis];
    float split_pos = node.aabb_min[axis] + extent[axis] / 2.0f;

    std::uint32_t i = node.begin_index;
    std::uint32_t j = i + node.triangle_count - 1;
    while (i <= j) {
      const auto& triangle = triangles[i];
      glm::vec3 centroid = (triangle[0].position + 
                            triangle[1].position + 
                            triangle[2].position) / 3.0f;
      if (centroid[axis] < split_pos) {
        ++i;
      } else {
        std::swap(triangles[i], triangles[j--]);
      }
    }

    std::uint32_t left_count = i - node.begin_index;
    if (left_count == 0 || left_count == node.triangle_count) {
      return;
    }

    node.left_child_index = node_count;

    auto& left_child = nodes[node_count++];
    left_child.begin_index = node.begin_index;
    left_child.triangle_count = left_count;

    auto& right_child = nodes[node_count++];
    right_child.begin_index = i;
    right_child.triangle_count = node.triangle_count - left_count;

    node.triangle_count = 0;

    subdivide(left_child);
    subdivide(right_child);
  }

  void update_bounds(BVHNode2& node) {
    node.aabb_min = glm::vec3(std::numeric_limits<float>::infinity());
    node.aabb_max = glm::vec3(-std::numeric_limits<float>::infinity());
    for (std::uint32_t i = 0; i < node.triangle_count; ++i) {
      for (int j = 0; j < 3; ++j) {
        const auto& vertex = triangles[node.begin_index + i][j];
        node.aabb_min = glm::min(node.aabb_min, glm::vec3(vertex.position));
        node.aabb_max = glm::max(node.aabb_max, glm::vec3(vertex.position));
      }
    }
  }

private:
  std::vector<Triangle>& triangles;
  std::vector<BVHNode2>& nodes;
  std::uint32_t node_count;
};

auto build_bvh(Scene& scene) -> std::vector<BVHNode> {
  std::vector<BVHNode2> nodes2;
  BVHBuilder builder { scene.triangles, nodes2 };
  builder.build();

  std::vector<BVHNode> nodes(nodes2.size());
  std::transform(
    nodes2.begin(), nodes2.end(), nodes.begin(), [] (const BVHNode2& node2) {
      BVHNode node {
        .aabb_min       = node2.aabb_min,
        .left_or_begin  = node2.triangle_count > 0 ? node2.begin_index : node2.left_child_index,
        .aabb_max       = node2.aabb_max,
        .triangle_count = node2.triangle_count
      };
      return node;
  });

  return nodes;
}