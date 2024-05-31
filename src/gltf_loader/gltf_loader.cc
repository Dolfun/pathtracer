#include "gltf_loader.h"
#include <filesystem>
#include <fmt/core.h>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp>

#define TINYGLTF_USE_CPP14
#define TINYGLTF_IMPLEMENTATION
#include <tiny_gltf.h>

template <typename T>
class AccessorHelper {
public:
  AccessorHelper() : ptr { nullptr }, offset { 0 }, stride { 0 } {}
 
  AccessorHelper(const tinygltf::Model& model, const tinygltf::Accessor& accessor) {
    assign(model, accessor);
  }

  void assign(const tinygltf::Model& model, const tinygltf::Accessor& accessor) {
    const auto& buffer_view = model.bufferViews[accessor.bufferView];
    offset = accessor.byteOffset + buffer_view.byteOffset;
    stride = accessor.ByteStride(buffer_view);
    byte_size = buffer_view.byteLength;
    ptr = model.buffers[buffer_view.buffer].data.data() + offset;
  }

  bool is_valid() const {
    if (stride == 0) {
      return false;
    } else {
      return sizeof(T) % stride == 0;
    }
  }

  std::size_t size() const {
    return byte_size / sizeof(T);
  }

  T get(std::size_t i) const {
    return static_cast<const T*>(ptr)[i];
  }

private:
  const void* ptr;
  std::size_t offset, stride, byte_size;
};

class Loader {
public:
  Loader(const std::string&);
  Scene process() const;

private:
  void process_node(const tinygltf::Node&, glm::mat4&, Scene&) const;
  void process_mesh(const tinygltf::Mesh&, const glm::mat4&, Scene&) const;

  tinygltf::Model model;
};

Loader::Loader(const std::string& path) {
  tinygltf::TinyGLTF loader;
  std::string error, warning;
  bool return_code;
  if (std::filesystem::path(path).extension() == ".glb") {
    return_code = loader.LoadBinaryFromFile(&model, &error, &warning, path);
  } else {
    return_code = loader.LoadASCIIFromFile(&model, &error, &warning, path);
  }

  if (!warning.empty()) {
    fmt::println("glTF Loader warning: {}", warning);
  }

  if (!error.empty()) {
    throw std::runtime_error(error);
  }

  if (!return_code) {
    throw std::runtime_error("Failed to parse glTF\n");
  }
}

Scene Loader::process() const {
  Scene result;
  const auto& gltf_scene = model.scenes[model.defaultScene];
  for (auto i : gltf_scene.nodes) {
    glm::mat4 transform { 1.0f };
    process_node(model.nodes[i], transform, result);
  }
  return result;
}

void Loader::process_node(const tinygltf::Node& node, glm::mat4& transform, Scene& result) const {
  if (!node.matrix.empty()) {
    glm::dmat4 mat;
    std::memcpy(&mat, node.matrix.data(), sizeof(glm::dmat4));
  } else {
    if (!node.translation.empty()) {
      glm::vec3 translate { node.translation[0], node.translation[1], node.translation[2] };
      transform = glm::translate(transform, translate);
    }

    if (!node.rotation.empty()) {
      glm::quat rotate(node.rotation[3], node.rotation[0], node.rotation[1], node.rotation[2]);
      transform *= glm::toMat4(rotate);
    }

    if (!node.scale.empty()) {
      glm::vec3 scale { node.scale[0], node.scale[1], node.scale[2] };
      transform = glm::scale(transform, scale);
    }
  }

  if (node.mesh != -1) {
    process_mesh(model.meshes[node.mesh], transform, result);
  }

  for (auto i : node.children) {
    process_node(model.nodes[i], transform, result);
  }
}

void Loader::process_mesh(const tinygltf::Mesh& mesh, const glm::mat4& transform, Scene& result) const {
  for (const auto& primitive : mesh.primitives) {
    if (primitive.mode != TINYGLTF_MODE_TRIANGLES) continue;

    AccessorHelper<glm::vec3> positions;
    {
      auto it = primitive.attributes.find("POSITION");
      assert(it != primitive.attributes.end());
      const auto& accessor = model.accessors[it->second];
      assert(!accessor.sparse.isSparse);
      assert(accessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT);
      assert(accessor.type == TINYGLTF_TYPE_VEC3);
      positions.assign(model, accessor);
      assert(positions.is_valid());
    }

    AccessorHelper<glm::vec3> normals;
    {
      auto it = primitive.attributes.find("NORMAL");
      assert(it != primitive.attributes.end());
      const auto& accessor = model.accessors[it->second];
      assert(!accessor.sparse.isSparse);
      assert(accessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT);
      assert(accessor.type == TINYGLTF_TYPE_VEC3);
      normals.assign(model, accessor);
      assert(normals.is_valid());
    }

    AccessorHelper<std::uint16_t> indices;
    {
      const auto& accessor = model.accessors[primitive.indices];
      assert(!accessor.sparse.isSparse);
      assert(accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT);
      assert(accessor.type == TINYGLTF_TYPE_SCALAR);
      indices.assign(model, accessor);
      assert(indices.is_valid());
    }

    auto normal_transform = glm::mat3(glm::transpose(glm::inverse(transform)));
    for (std::size_t i = 0; i < indices.size(); ++i) {
      auto index = indices.get(i);
      auto position = glm::vec3(transform * glm::vec4(positions.get(i), 1.0f));
      auto normal   = glm::vec3(normal_transform * normals.get(i));
      result.vertices.emplace_back(glm::vec4(position, 0.0f), glm::vec4(normal, 0.0f));
    }
  }
}

Scene load_gltf(const std::string& path) {
  Loader loader { path };
  return loader.process();
}