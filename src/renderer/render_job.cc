#include "render_job.h"

RenderJob::RenderJob(const VkManager& _vk_manager, const RenderConfig& _config)
    : vk_manager { _vk_manager }, config { _config } {

}

auto RenderJob::render() const -> std::vector<std::byte> {
  return {};
}