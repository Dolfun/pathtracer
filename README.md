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

## Samples

### Suzanne

![suzanne](https://i.imgur.com/J5N4zR3.png)

### Cornell Box

![cornell box](https://i.imgur.com/SCCgPDY.png)

### [Forest House](https://sketchfab.com/3d-models/forest-house-52429e4ef7bf4deda1309364a2cda86f)

![forest house](https://i.imgur.com/DW66g5C.png)

### [Western Pacific](https://sketchfab.com/3d-models/emd-gp7-western-pacific-713-1c89cb9f2c224b78b6fea50f82e042c3)

![western pacific](https://i.imgur.com/5VnCCIY.png)
