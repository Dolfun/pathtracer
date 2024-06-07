#include "bvh.h"
#include <tuple>
#include <limits>
#include <algorithm>
#include <glm/glm.hpp>

using Vertex = Scene::Vertex;
using VertexIndices = Scene::VertexIndices;

struct Bin {
  AABB bounds;
  std::uint32_t triangle_count;
};

class BVHBuilder {
public:
  BVHBuilder(Scene& scene, std::vector<BVHNode>& _nodes, std::uint32_t _bin_count)
    : vertices { scene.vertices }, triangle_indices { scene.triangle_indices }, 
      nodes { _nodes }, node_count { 2 }, bin_count { _bin_count } {}

  void build();
  void subdivide(BVHNode&);
  void update_bounds(BVHNode&) const;
  auto find_split(BVHNode&) const -> std::tuple<std::uint32_t, float, float>;
  glm::vec3 centroid(const VertexIndices&) const;

private:
  const std::vector<Vertex>& vertices;
  std::vector<VertexIndices>& triangle_indices;
  std::vector<BVHNode>& nodes;
  std::uint32_t node_count;
  std::uint32_t bin_count;
};

void BVHBuilder::build() {
  std::uint32_t triangle_count = triangle_indices.size();
  nodes.resize(2 * triangle_count);

  BVHNode& root = nodes[0];
  root.left_child_index = 0;
  root.begin_index = 0;
  root.triangle_count = triangle_count;

  subdivide(root);
  nodes.resize(node_count);
}

void BVHBuilder::subdivide(BVHNode& node) {
  update_bounds(node);

  auto [split_axis, split_pos, split_cost] = find_split(node);
  float nosplit_cost = node.aabb.area() * node.triangle_count;
  if (split_cost >= nosplit_cost) return;

  auto begin_it = triangle_indices.begin() + node.begin_index;
  auto end_it   = begin_it + node.triangle_count;
  auto pred = [&] (const VertexIndices& indices) {
    return centroid(indices)[split_axis] < split_pos;
  };
  auto it = std::partition(begin_it, end_it, pred);
  
  auto i = static_cast<std::uint32_t>(it - triangle_indices.begin());
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

void BVHBuilder::update_bounds(BVHNode& node) const {
  for (std::uint32_t i = 0; i < node.triangle_count; ++i) {
    node.aabb.expand(vertices, triangle_indices[node.begin_index + i]);
  }
}

auto BVHBuilder::find_split(BVHNode& node) const -> std::tuple<std::uint32_t, float, float> {
  std::uint32_t split_axis;
  float split_pos;
  float split_cost = std::numeric_limits<float>::infinity();

  glm::vec3 bound_min {  std::numeric_limits<float>::infinity() };
  glm::vec3 bound_max { -std::numeric_limits<float>::infinity() };
  for (std::uint32_t i = 0; i < node.triangle_count; ++i) {
    glm::vec3 v = centroid(triangle_indices[node.begin_index + i]);
    bound_min = glm::min(bound_min, v);
    bound_max = glm::max(bound_max, v);
  }

  for (std::uint32_t axis = 0; axis < 3; ++axis) {
    std::vector<Bin> bins(bin_count);
    float scale = static_cast<float>(bin_count) / (bound_max[axis] - bound_min[axis]);
    for (std::uint32_t i = 0; i < node.triangle_count; ++i) {
      const auto& vertex_indices = triangle_indices[node.begin_index + i];
      std::uint32_t index = std::min(
        bin_count - 1,
        static_cast<std::uint32_t>((centroid(vertex_indices)[axis] - bound_min[axis]) * scale)
      );
      ++bins[index].triangle_count;
      bins[index].bounds.expand(vertices, vertex_indices);
    }

    std::vector<float> left_area(bin_count), right_area(bin_count);
    std::vector<std::uint32_t> left_count(bin_count), right_count(bin_count);
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

glm::vec3 BVHBuilder::centroid(const VertexIndices& indices) const {
  glm::vec3 c = (
    vertices[indices[0]].position +
    vertices[indices[1]].position +
    vertices[indices[2]].position
  ) / 3.0f;
  return c;
}

auto build_bvh(Scene& scene, std::uint32_t bin_count) -> std::vector<BVHNode> {
  std::vector<BVHNode> nodes;

  BVHBuilder builder { scene, nodes, bin_count };
  builder.build();

  return nodes;
}