#include "render_job.h"
#include "renderer.h"

RenderJob::RenderJob(const Renderer& _renderer, const RenderConfig& _config)
  : renderer{ _renderer }, config { _config }, 
    allocator { renderer.device, renderer.vk_manager.get_physical_device().getMemoryProperties() } {
  
  image_size = config.image_width * config.image_height * NR_CHANNELS;
  allocator.allocate_and_bind();
}

auto RenderJob::render() const -> std::vector<std::byte> {
  std::vector<std::byte> result(image_size);

  return result;
}