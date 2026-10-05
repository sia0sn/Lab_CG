#include "graphics.hpp"
#include "graphics_internal.hpp"

#include <cstring>
#include <fstream>
#include <iostream>

namespace graphics {

    std::string findShaderPath(const std::string& filename)
    {
        const std::vector<std::string> candidates = {
            filename,
            "../" + filename,
            "../../" + filename,
            "../../../" + filename,
            "shaders/" + filename,
            "../shaders/" + filename,
            "../../shaders/" + filename,
        };
        for (const auto& path : candidates) {
            std::ifstream f(path, std::ios::binary);
            if (f.good()) return path;
        }
        return filename;
    }

    Buffer createBuffer(VkDeviceSize size,
        VkBufferUsageFlags usage,
        VmaMemoryUsage memory_usage,
        bool persistent_map,
        const void* data)
    {
        VkBufferCreateInfo buffer_info{};
        buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buffer_info.size = size;
        buffer_info.usage = usage;
        buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo alloc_info{};
        alloc_info.usage = memory_usage;
        alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
            VMA_ALLOCATION_CREATE_MAPPED_BIT;

        Buffer result;
        if (vmaCreateBuffer(internal::context.allocator, &buffer_info, &alloc_info,
            &result.buffer, &result.allocation, &result.info) != VK_SUCCESS) {
            std::cerr << "[graphics] Failed to create buffer\n";
            return {};
        }

        if (persistent_map) {
            result.mapped = result.info.pMappedData;
        }

        if (data && result.info.pMappedData) {
            std::memcpy(result.info.pMappedData, data, static_cast<size_t>(size));
        }

        return result;
    }

    void destroyBuffer(Buffer& b)
    {
        if (b.buffer) {
            vmaDestroyBuffer(internal::context.allocator, b.buffer, b.allocation);
            b = {};
        }
    }

    VkShaderModule loadShaderModule(const std::string& path)
    {
        const std::string real_path = findShaderPath(path);
        std::ifstream file(real_path, std::ios::ate | std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "[graphics] Failed to open shader: " << real_path << '\n';
            return VK_NULL_HANDLE;
        }
        const size_t size = static_cast<size_t>(file.tellg());
        std::vector<char> code(size);
        file.seekg(0);
        file.read(code.data(), static_cast<std::streamsize>(size));

        VkShaderModuleCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        info.codeSize = code.size();
        info.pCode = reinterpret_cast<const uint32_t*>(code.data());

        VkShaderModule module = VK_NULL_HANDLE;
        if (vkCreateShaderModule(internal::context.device, &info, nullptr, &module) != VK_SUCCESS) {
            std::cerr << "[graphics] Failed to create shader module: " << real_path << '\n';
        }
        return module;
    }

    void destroyShaderModule(VkShaderModule m)
    {
        if (m) vkDestroyShaderModule(internal::context.device, m, nullptr);
    }

    VkPipelineLayout createPipelineLayout(VkDescriptorSetLayout set_layout)
    {
        VkPipelineLayoutCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        info.setLayoutCount = 1;
        info.pSetLayouts = &set_layout;

        VkPipelineLayout layout = VK_NULL_HANDLE;
        vkCreatePipelineLayout(internal::context.device, &info, nullptr, &layout);
        return layout;
    }

    VkPipeline createGraphicsPipeline(
        VkRenderPass render_pass,
        VkPipelineLayout layout,
        VkShaderModule vert_shader,
        VkShaderModule frag_shader,
        VkVertexInputBindingDescription binding,
        const std::vector<VkVertexInputAttributeDescription>& attributes)
    {
        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vert_shader;
        stages[0].pName = "main";
        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = frag_shader;
        stages[1].pName = "main";

        VkPipelineVertexInputStateCreateInfo vertex_input{};
        vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertex_input.vertexBindingDescriptionCount = 1;
        vertex_input.pVertexBindingDescriptions = &binding;
        vertex_input.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributes.size());
        vertex_input.pVertexAttributeDescriptions = attributes.data();

        VkPipelineInputAssemblyStateCreateInfo input_asm{};
        input_asm.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        input_asm.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo viewport{};
        viewport.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewport.viewportCount = 1;
        viewport.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo raster{};
        raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        raster.polygonMode = VK_POLYGON_MODE_FILL;
        raster.cullMode = VK_CULL_MODE_NONE;
        raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        raster.lineWidth = 1.0f;

        VkPipelineMultisampleStateCreateInfo multisample{};
        multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo depth{};
        depth.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depth.depthTestEnable = VK_TRUE;
        depth.depthWriteEnable = VK_TRUE;
        depth.depthCompareOp = VK_COMPARE_OP_LESS;

        VkPipelineColorBlendAttachmentState blend_attachment{};
        blend_attachment.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        blend_attachment.blendEnable = VK_FALSE;

        VkPipelineColorBlendStateCreateInfo blend{};
        blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        blend.attachmentCount = 1;
        blend.pAttachments = &blend_attachment;

        const VkDynamicState dynamic_states[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo dynamic{};
        dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamic.dynamicStateCount = 2;
        dynamic.pDynamicStates = dynamic_states;

        VkGraphicsPipelineCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        info.stageCount = 2;
        info.pStages = stages;
        info.pVertexInputState = &vertex_input;
        info.pInputAssemblyState = &input_asm;
        info.pViewportState = &viewport;
        info.pRasterizationState = &raster;
        info.pMultisampleState = &multisample;
        info.pDepthStencilState = &depth;
        info.pColorBlendState = &blend;
        info.pDynamicState = &dynamic;
        info.layout = layout;
        info.renderPass = render_pass;
        info.subpass = 0;

        VkPipeline pipeline = VK_NULL_HANDLE;
        if (vkCreateGraphicsPipelines(internal::context.device, VK_NULL_HANDLE,
            1, &info, nullptr, &pipeline) != VK_SUCCESS) {
            std::cerr << "[graphics] Failed to create graphics pipeline\n";
        }
        return pipeline;
    }

    VkDescriptorSetLayout createDescriptorSetLayout()
    {
        VkDescriptorSetLayoutBinding ubo{};
        ubo.binding = 0;
        ubo.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        ubo.descriptorCount = 1;
        ubo.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        info.bindingCount = 1;
        info.pBindings = &ubo;

        VkDescriptorSetLayout layout = VK_NULL_HANDLE;
        vkCreateDescriptorSetLayout(internal::context.device, &info, nullptr, &layout);
        return layout;
    }

    VkDescriptorPool createDescriptorPool(uint32_t max_sets)
    {
        VkDescriptorPoolSize sizes[] = {
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, max_sets },
        };

        VkDescriptorPoolCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        info.maxSets = max_sets;
        info.poolSizeCount = 1;
        info.pPoolSizes = sizes;

        VkDescriptorPool pool = VK_NULL_HANDLE;
        vkCreateDescriptorPool(internal::context.device, &info, nullptr, &pool);
        return pool;
    }

    VkDescriptorSet allocateDescriptorSet(VkDescriptorPool pool, VkDescriptorSetLayout layout)
    {
        VkDescriptorSetAllocateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        info.descriptorPool = pool;
        info.descriptorSetCount = 1;
        info.pSetLayouts = &layout;

        VkDescriptorSet set = VK_NULL_HANDLE;
        vkAllocateDescriptorSets(internal::context.device, &info, &set);
        return set;
    }

} // namespace graphics