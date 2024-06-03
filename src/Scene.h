#pragma once
#include <glm/vec3.hpp>
#include <vector>

struct Scene {
  struct Vertex {
    glm::vec3 position;
    alignas(16) glm::vec3 normal;
  };

  struct Triangle {
    Vertex vertices[3];

    const Vertex& operator[] (std::size_t i) const noexcept {
      return vertices[i];
    };

    Vertex& operator[] (std::size_t i) noexcept {
      return const_cast<Vertex&>(const_cast<const Triangle&>(*this)[i]);
    };
  };

  std::vector<Triangle> triangles;
};