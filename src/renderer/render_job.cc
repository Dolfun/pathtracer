#include "render_job.h"

RenderJob::RenderJob(const VkManager& vk_manager, const RenderConfig& _config)
  : device { vk_manager.get_device() }, vk_allocator { vk_manager }, config { _config } {
  
  image_size = config.image_width * config.image_height * NR_CHANNELS;
  vk_allocator.allocate_and_bind();
}

auto RenderJob::render() const -> std::vector<std::byte> {
  std::vector<std::byte> result(image_size);

  return result;
}