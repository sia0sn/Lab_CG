#include "application.hpp"

#include <imgui.h>
#include <vulkan/vulkan.h>

namespace application {

    bool initialize()
    {
        return true;
    }

    void shutdown()
    {
        auto& context = graphics::internal::context;
        vkQueueWaitIdle(context.graphics_queue);
    }

    void update([[maybe_unused]] double time)
    {
        ImGui::ShowDemoWindow();
    }

    void render(const graphics::internal::FrameData& fd)
    {
        auto& context = graphics::internal::context;

        vkResetCommandBuffer(fd.command_buffer, 0);

        const VkCommandBufferBeginInfo command_buffer_begin = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };

        vkBeginCommandBuffer(fd.command_buffer, &command_buffer_begin);

        const VkClearValue clear_values[] = {
            {
                .color = {
                    .float32 = { 0.1f, 0.1f, 0.1f, 1.0f }
                }
            },
            {
                .depthStencil = {
                    .depth = 1.0f,
                    .stencil = 0
                }
            }
        };

        const VkRenderPassBeginInfo render_pass_begin = {
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass = context.render_pass,
            .framebuffer = fd.framebuffer,
            .renderArea = {
                .offset = { 0, 0 },
                .extent = context.swapchain_extent
            },
            .clearValueCount =
                static_cast<uint32_t>(sizeof(clear_values) / sizeof(clear_values[0])),
            .pClearValues = clear_values,
        };

        vkCmdBeginRenderPass(
            fd.command_buffer,
            &render_pass_begin,
            VK_SUBPASS_CONTENTS_INLINE
        );

        vkCmdEndRenderPass(fd.command_buffer);

        vkEndCommandBuffer(fd.command_buffer);
    }

} // namespace application