#pragma once
#include <glm/vec4.hpp>
#include <vector>

struct Scene {
  struct Vertex {
    glm::vec4 position;
    glm::vec4 normal;
  };

  std::vector<Vertex> vertices;
};