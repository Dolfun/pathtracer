#include "renderer.h"
#include "render_job.h"

auto Renderer::render(const RenderConfig& config) const -> std::vector<std::byte> {
  RenderJob render_job { vk_manager, config };

  auto result = render_job.render();

  return result;
}