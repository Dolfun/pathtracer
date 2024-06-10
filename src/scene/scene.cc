#include "scene.h"
#include "../bvh/bvh.h"
#include <algorithm>

OptimizedScene optimize_scene(const Scene& scene, const std::vector<BVHNode>& nodes) {
  OptimizedScene optimized_scene;

  std::size_t vertex_count = scene.triangle_indices.size() * 3;
  optimized_scene.vertex_positions.reserve(vertex_count);
  optimized_scene.vertex_data.reserve(vertex_count);
  for (const auto& vertex_indices : scene.triangle_indices) {
    for (auto i : vertex_indices) {
      const auto& vertex = scene.vertices[i];

      optimized_scene.vertex_positions.push_back(
        glm::vec4(vertex.position, 0.0f)
      );

      OptimizedScene::VertexData vertex_data {
        .normal_and_texcoord_u = glm::vec4(vertex.normal, vertex.texcoord.x),
        .tangent_and_texcoord_v = glm::vec4(vertex.tangent, vertex.texcoord.y),
        .bitangent = vertex.bitangnet,
        .material_index = vertex.material_index
      };
      optimized_scene.vertex_data.push_back(vertex_data);
    }
  }

  optimized_scene.bvh_nodes.resize(nodes.size());
  std::transform(
    nodes.begin(), nodes.end(), optimized_scene.bvh_nodes.begin(),
    [] (const BVHNode& node) {
      OptimizedScene::OptimizedBVHNode optimized_node {
        .aabb_min       = node.aabb.box_min,
        .left_or_begin  = node.triangle_count > 0 ? node.begin_index : node.left_child_index,
        .aabb_max       = node.aabb.box_max,
        .triangle_count = node.triangle_count
      };

      return optimized_node;
    }
  );

  optimized_scene.materials = scene.materials;

  return optimized_scene;
}