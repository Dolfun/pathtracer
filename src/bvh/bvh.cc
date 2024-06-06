#include "bvh.h"
#include <tuple>
#include <limits>
#include <algorithm>
#include <execution>
#include <glm/glm.hpp>

using Triangle = Scene::Triangle;
glm::vec3 centroid(const Triangle& t) {
  return (t[0].position + t[1].position + t[2].position) / 3.0f;
}

constexpr float infinity = std::numeric_limits<float>::infinity();

struct AABB {
  glm::vec3 box_min;
  glm::vec3 box_max;

  AABB() : box_min { infinity }, box_max { -infinity } {}

  void expand(const glm::vec3& v) {
    box_max = glm::max(box_max, v);
    box_min = glm::min(box_min, v);
  }

  void expand(const Triangle& triangle) {
    expand(triangle[0].position);
    expand(triangle[1].position);
    expand(triangle[2].position);
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

struct BVHNode2 {
  AABB aabb;
  std::uint32_t left_child_index;
  std::uint32_t begin_index;
  std::uint32_t triangle_count;
};

struct Bin {
  AABB bounds;
  std::uint32_t triangle_count;
};
constexpr std::uint32_t bin_count = 8;

class BVHBuilder {
public:
  BVHBuilder(std::vector<Triangle>& _triangles, std::vector<BVHNode2>& _nodes)
    : triangles { _triangles }, nodes { _nodes } {}

  void build() {
    std::uint32_t triangle_count = triangles.size();
    nodes.resize(2 * triangle_count);
    node_count = 2;

    BVHNode2& root = nodes[0];
    root.left_child_index = 0;
    root.begin_index = 0;
    root.triangle_count = triangle_count;

    subdivide(root);
    nodes.resize(node_count);
  }

  void subdivide(BVHNode2& node) {
    update_bounds(node);

    auto [split_axis, split_pos, split_cost] = find_split(node);
    float nosplit_cost = node.aabb.area() * node.triangle_count;
    if (split_cost >= nosplit_cost) return;

    auto begin_it = triangles.begin() + node.begin_index;
    auto end_it   = begin_it + node.triangle_count;
    auto pred = [&] (const Triangle& triangle) {
      return centroid(triangle)[split_axis] < split_pos;
    };
    auto it = std::partition(begin_it, end_it, pred);
    
    auto i = static_cast<std::uint32_t>(it - triangles.begin());
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

  void update_bounds(BVHNode2& node) const {
    for (std::uint32_t i = 0; i < node.triangle_count; ++i) {
      node.aabb.expand(triangles[node.begin_index + i]);
    }
  }

  auto find_split(BVHNode2& node) const -> std::tuple<std::uint32_t, float, float> {
    std::uint32_t split_axis;
    float split_pos;
    float split_cost = std::numeric_limits<float>::infinity();

    glm::vec3 bound_min {  infinity };
    glm::vec3 bound_max { -infinity };
    for (std::uint32_t i = 0; i < node.triangle_count; ++i) {
      glm::vec3 v = centroid(triangles[node.begin_index + i]);
      bound_min = glm::min(bound_min, v);
      bound_max = glm::max(bound_max, v);
    }

    for (std::uint32_t axis = 0; axis < 3; ++axis) {
      std::array<Bin, bin_count> bins{};
      float scale = static_cast<float>(bin_count) / (bound_max[axis] - bound_min[axis]);
      for (std::uint32_t i = 0; i < node.triangle_count; ++i) {
        const auto& triangle = triangles[node.begin_index + i];
        std::uint32_t index = std::min(
          bin_count - 1,
          static_cast<std::uint32_t>((centroid(triangle)[axis] - bound_min[axis]) * scale)
        );
        ++bins[index].triangle_count;
        bins[index].bounds.expand(triangle);
      }

      std::array<float, bin_count> left_area{}, right_area{};
      std::array<std::uint32_t, bin_count> left_count{}, right_count{};
      AABB left_box, right_box;
      std::uint32_t left_sum = 0, right_sum = 0;
      for (std::uint32_t i = 0; i < bin_count - 1; ++i) {
        left_sum += bins[i].triangle_count;
        left_count[i] = left_sum;
        left_box.expand(bins[i].bounds);
        left_area[i] = left_box.area();

        right_sum += bins[bin_count - i - 1].triangle_count;
        right_count[bin_count - i - 2] = right_sum;
        right_box.expand(bins[bin_count - i - 1].bounds);
        right_area[bin_count - i - 2] = right_box.area();
      }

      scale = (bound_max[axis] - bound_min[axis]) / static_cast<float>(bin_count);
      for (std::uint32_t i = 0; i < bin_count - 1; ++i) {
        float cost = left_count[i] * left_area[i] + right_count[i] * right_area[i];
        if (cost < split_cost) {
          split_axis = axis;
          split_pos = bound_min[axis] + scale * (i + 1);
          split_cost = cost;
        }
      }
    }

    return { split_axis, split_pos, split_cost };
  }

  float calculate_cost(BVHNode2& node, std::uint32_t axis, float pos) const {
    AABB left_box, right_box;
    std::uint32_t left_count = 0, right_count = 0;
    for (std::uint32_t i = 0; i < node.triangle_count; ++i) {
      const auto& triangle = triangles[node.begin_index + i];
      if (centroid(triangle)[axis] < pos) {
        ++left_count;
        left_box.expand(triangle);
      } else {
        ++right_count;
        right_box.expand(triangle);
      }
    }
    float cost = left_count * left_box.area() + right_count * right_box.area();
    return cost;
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
    std::execution::par,
    nodes2.begin(), nodes2.end(), nodes.begin(), [] (const BVHNode2& node2) {
      BVHNode node {
        .aabb_min       = node2.aabb.box_min,
        .left_or_begin  = node2.triangle_count > 0 ? node2.begin_index : node2.left_child_index,
        .aabb_max       = node2.aabb.box_max,
        .triangle_count = node2.triangle_count
      };
      return node;
  });

  return nodes;
}