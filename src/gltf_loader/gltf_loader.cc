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
  AccessorHelper() : ptr { nullptr }, stride { 0 } {}

  AccessorHelper(const tinygltf::Model& model, const tinygltf::Accessor& accessor) {
    auto component_size = tinygltf::GetComponentSizeInBytes(accessor.componentType);
    auto nr_components  = tinygltf::GetNumComponentsInType(static_cast<std::uint32_t>(accessor.type));
    assert(sizeof(T) == static_cast<std::size_t>(component_size * nr_components));

    const auto& buffer_view = model.bufferViews[accessor.bufferView];
    stride = accessor.ByteStride(buffer_view);
    element_count = accessor.count;

    std::size_t offset = accessor.byteOffset + buffer_view.byteOffset;
    ptr = model.buffers[buffer_view.buffer].data.data() + offset;
  }

  T operator[] (std::size_t i) const {
    return *reinterpret_cast<const T*>(ptr + i * stride);
  }

  std::size_t count() const {
    return element_count;
  }

  bool empty() const {
    return ptr == nullptr;
  }

private:
  const unsigned char* ptr;
  std::size_t stride;
  std::size_t element_count;
};

class Loader {
public:
  Loader(const std::string&, Scene&);
  void process();

private:
  void process_node(const tinygltf::Node&, glm::mat4&);

  template <typename T>
  auto get_attribute_accessor(const tinygltf::Primitive&, const std::string&) const
    -> AccessorHelper<T>;

  template <typename T>
  void process_indices(const AccessorHelper<T>&);

  void process_primitive(const tinygltf::Primitive&, const glm::mat4&);

  Scene& scene;
  tinygltf::Model model;
  std::uint32_t index_offset;
};

Loader::Loader(const std::string& path, Scene& _scene) 
    : scene { _scene }, index_offset { 0 } {
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

void Loader::process() {
  const auto& gltf_scene = model.scenes[model.defaultScene];
  for (auto i : gltf_scene.nodes) {
    glm::mat4 transform { 1.0f };
    process_node(model.nodes[i], transform);
  }
}

void Loader::process_node(const tinygltf::Node& node, glm::mat4& transform) {
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
    const auto& mesh = model.meshes[node.mesh];
    for (const auto& primitive : mesh.primitives) {
      if (primitive.mode == TINYGLTF_MODE_TRIANGLES) {
        process_primitive(primitive, transform);
      }
    }
  }

  for (auto i : node.children) {
    process_node(model.nodes[i], transform);
  }
}

template <typename T>
auto Loader::get_attribute_accessor(
  const tinygltf::Primitive& primitive, 
  const std::string& attribute) const
    -> AccessorHelper<T> {
  auto it = primitive.attributes.find(attribute);
  assert(it != primitive.attributes.end());
  const auto& accessor = model.accessors[it->second];
  assert(!accessor.sparse.isSparse);
  assert(accessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT);
  constexpr std::size_t nr_compnents = sizeof(T) / sizeof(float);
  if constexpr (nr_compnents == 2) {
    assert(accessor.type == TINYGLTF_TYPE_VEC2);
  } else if constexpr (nr_compnents == 3) {
    assert(accessor.type == TINYGLTF_TYPE_VEC3);
  } else if constexpr (nr_compnents == 4) {
    assert(accessor.type == TINYGLTF_TYPE_VEC4);
  } else {
    static_assert(false, "Invalid type for attribute accessor.");
  }
  return AccessorHelper<T> { model, accessor };
}

template <typename T>
void Loader::process_indices(const AccessorHelper<T>& indices) {
  for (std::size_t i = 0; i < indices.count(); i += 3) {
    Scene::VertexIndices vertex_indices = {
      index_offset + indices[i],
      index_offset + indices[i + 1],
      index_offset + indices[i + 2],
    };
    scene.triangle_indices.push_back(vertex_indices);
  }
}

void Loader::process_primitive(const tinygltf::Primitive& primitive, const glm::mat4& transform) {
  auto positions = get_attribute_accessor<glm::vec3>(primitive, "POSITION");
  auto normals = get_attribute_accessor<glm::vec3>(primitive, "NORMAL");
  auto tex_coords = get_attribute_accessor<glm::vec2>(primitive, "TEXCOORD_0");
  AccessorHelper<glm::vec4> tangents;
  if (primitive.attributes.contains("TANGENT")) {
    tangents = get_attribute_accessor<glm::vec4>(primitive, "TANGENT");
  }

  std::size_t vertex_count = positions.count();
  assert(vertex_count == normals.count());
  assert(vertex_count == tex_coords.count());
  auto normal_transform = glm::mat3(glm::transpose(glm::inverse(transform)));
  for (std::size_t i = 0; i < vertex_count; ++i) {
    Scene::Vertex vertex {
      .position = glm::vec3(transform * glm::vec4(positions[i], 1.0f)),
      .normal = glm::normalize(glm::vec3(normal_transform * normals[i])),
      .texcoord = tex_coords[i]
    };

    if (!tangents.empty()) {
      vertex.tangent = glm::vec3(tangents[i]);
      vertex.bitangnet = glm::cross(normals[i], vertex.tangent) * tangents[i].w;
      vertex.tangent = glm::normalize(glm::vec3(normal_transform * vertex.tangent));
      vertex.bitangnet = glm::normalize(glm::vec3(normal_transform * vertex.bitangnet));
    }

    scene.vertices.push_back(vertex);
  }

  const auto& accessor = model.accessors[primitive.indices];
  if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
    AccessorHelper<std::uint16_t> indices { model, accessor };
    process_indices(indices);

  } else if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
    AccessorHelper<std::uint32_t> indices { model, accessor };
    process_indices(indices);

  } else {
    throw std::runtime_error("Unknown index type in gltf file.");
  }

  index_offset += vertex_count;
}

Scene load_gltf(const std::string& path) {
  Scene scene;

  Loader loader { path, scene };
  loader.process();

  return scene;
}