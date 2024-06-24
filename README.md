# Pathtracer

A pathtracer made using C++ and Vulkan.

## Features

- Unidirectional path tracing with multiple importance sampling
- Supports directional, point and area lights
- Supports textures and PBR materials
- Uses Vulkan compute shaders for GPU acceleration
- Uses binned BVH with surface area heuristics
- Cross platform support: Windows and Linux
- Cross GPU Vendor support: Nvidia and AMD

## Building

### Requirements

- C++ compiler with C++20 support
- CMake
- Vulkan SDK
- [glslc](https://github.com/google/shaderc.git) executable in PATH

### Commands

```console
> git clone https://github.com/Dolfun/pathtracer.git --recurse-submodules
> cd pathtracer
> mkdir build
> cd build
> cmake ..
> cmake --build . --config Release
```

## Usage

### List devices

```console
> pathtracer list_devices
```

### Rendering

``` console
> pathtracer config.json
```

or

 ```console
> pathtracer
```

This will use config.json as default config file.

### Config File

```json
{
  "resolution_x": 1920,
  "resolution_y": 1080,
  "sample_count": 128,
  "bg_color": [ 0.239, 0.239, 0.239 ],
  "gltf_path": "cube.glb",
  "device_id": 0
}
```

Set device_id to -1 to auto-select device.

## Samples

### Diffuse

![diffuse](https://i.imgur.com/VqUf08Q.png)

### Metal

![metal](https://i.imgur.com/FOu7iwf.png)

### Reflection

![reflection](https://i.imgur.com/jYzjRD5.png)

### Cornell Box

![cornell box](https://i.imgur.com/wmEt1h2.png)

### [Western Pacific](https://sketchfab.com/3d-models/emd-gp7-western-pacific-713-1c89cb9f2c224b78b6fea50f82e042c3)

![western pacific](https://i.imgur.com/k8zUboP.png)

### [Forest House](https://sketchfab.com/3d-models/forest-house-52429e4ef7bf4deda1309364a2cda86f)

![forest house](https://i.imgur.com/ZGMxMi4.png)
