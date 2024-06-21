# Pathtracer

A pathtracer created using C++ and Vulkan.

## Building

### Requirements

- C++ compiler with C++20 support
- CMake
- Vulkan SDK

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
