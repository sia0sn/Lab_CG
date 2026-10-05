#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

namespace graphics {

    struct Buffer {
        VkBuffer         buffer = VK_NULL_HANDLE;
        VmaAllocation    allocation = VK_NULL_HANDLE;
        VmaAllocationInfo info{};
        void* mapped = nullptr;
    };

    Buffer createBuffer(VkDeviceSize       size,
        VkBufferUsageFlags usage,
        VmaMemoryUsage     memory_usage,
        bool               persistent_map,
        const void* data = nullptr);

    void destroyBuffer(Buffer& b);

    VkShaderModule loadShaderModule(const std::string& path);
    void           destroyShaderModule(VkShaderModule m);

    VkPipelineLayout createPipelineLayout(VkDescriptorSetLayout set_layout);
    VkPipeline       createGraphicsPipeline(
        VkRenderPass           render_pass,
        VkPipelineLayout       layout,
        VkShaderModule         vert_shader,
        VkShaderModule         frag_shader,
        VkVertexInputBindingDescription                binding,
        const std::vector<VkVertexInputAttributeDescription>& attributes);

    VkDescriptorSetLayout createDescriptorSetLayout();
    VkDescriptorPool      createDescriptorPool(uint32_t max_sets);
    VkDescriptorSet       allocateDescriptorSet(VkDescriptorPool      pool,
        VkDescriptorSetLayout layout);

    // Ищет файл (например "shaders/object.vert.spv") в нескольких
    // возможных рабочих директориях. Возвращает первый найденный путь.
    std::string findShaderPath(const std::string& filename);

} // namespace graphics