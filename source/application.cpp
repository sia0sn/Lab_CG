#include "application.hpp"
#include "graphics.hpp"

#include <imgui.h>
#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

namespace application {

    namespace {

        using graphics::Buffer;

        struct Vertex { glm::vec3 position; };

        struct UniformBufferObject {
            alignas(16) glm::mat4 model;
            alignas(16) glm::mat4 view;
            alignas(16) glm::mat4 proj;
            alignas(16) glm::vec4 color;
        };

        // ------- Глобальное состояние рендера -------
        VkDescriptorSetLayout g_set_layout = VK_NULL_HANDLE;
        VkPipelineLayout      g_pipeline_layout = VK_NULL_HANDLE;
        VkPipeline            g_pipeline = VK_NULL_HANDLE;
        VkDescriptorPool      g_descriptor_pool = VK_NULL_HANDLE;

        // Задание 6: два объекта — два UBO и два descriptor set
        constexpr int OBJECT_COUNT = 2;
        Buffer          g_uniform_buffers[OBJECT_COUNT]{};
        VkDescriptorSet g_descriptor_sets[OBJECT_COUNT]{};

        Buffer g_vertex_buffer{};
        Buffer g_index_buffer{};
        uint32_t g_index_count = 0;

        // ------- Состояние трансформации (для UI и анимации) -------
        struct ObjectState {
            glm::vec3 position{ 0.0f };
            glm::vec3 rotation{ 0.0f };
            glm::vec3 scale{ 1.0f };
            glm::vec4 color{ 1.0f, 0.75f, 0.3f, 1.0f };

            // Задание 3 — анимация
            bool  animate = false;
            float anim_speed = 1.0f;
            float anim_radius = 1.5f;
            float anim_phase = 0.0f;

            // Задание 6
            bool  visible = true;
        };

        ObjectState g_objects[OBJECT_COUNT];

        // Глобальные настройки
        bool g_use_perspective = true;

        // ------- Генерация додекаэдра -------
        void generateDodecahedron(std::vector<Vertex>& vertices,
            std::vector<uint32_t>& indices)
        {
            const float phi = (1.0f + std::sqrt(5.0f)) * 0.5f;
            const float iphi = 1.0f / phi;

            vertices = {
                {{-1, -1, -1}}, {{-1, -1,  1}}, {{-1,  1, -1}}, {{-1,  1,  1}},
                {{ 1, -1, -1}}, {{ 1, -1,  1}}, {{ 1,  1, -1}}, {{ 1,  1,  1}},
                {{ 0, -iphi, -phi}}, {{ 0, -iphi,  phi}},
                {{ 0,  iphi, -phi}}, {{ 0,  iphi,  phi}},
                {{-iphi, -phi, 0}}, {{-iphi,  phi, 0}},
                {{ iphi, -phi, 0}}, {{ iphi,  phi, 0}},
                {{-phi, 0, -iphi}}, {{-phi, 0,  iphi}},
                {{ phi, 0, -iphi}}, {{ phi, 0,  iphi}},
            };

            static const uint32_t faces[12][5] = {
                { 0, 16,  2, 10,  8},
                { 0, 12, 14,  4,  8},
                { 0, 16, 17,  1, 12},
                { 2, 13, 15,  6, 10},
                { 3, 11,  9,  1, 17},
                { 3, 13, 15,  7, 11},
                { 8, 10,  6, 18,  4},
                { 4, 18, 19,  5, 14},
                { 5,  9, 11,  7, 19},
                { 5, 14, 12,  1,  9},
                { 2, 16, 17,  3, 13},
                { 6, 18, 19,  7, 15},
            };

            indices.clear();
            indices.reserve(12 * 3 * 3);
            for (const auto& f : faces) {
                indices.push_back(f[0]); indices.push_back(f[1]); indices.push_back(f[2]);
                indices.push_back(f[0]); indices.push_back(f[2]); indices.push_back(f[3]);
                indices.push_back(f[0]); indices.push_back(f[3]); indices.push_back(f[4]);
            }

            // Нормируем радиус ~ 1
            for (auto& v : vertices) {
                v.position = glm::normalize(v.position) * 1.0f;
            }
        }

        glm::mat4 computeModel(const ObjectState& s)
        {
            glm::mat4 model = glm::mat4(1.0f);
            model = glm::translate(model, s.position);
            model = glm::rotate(model, glm::radians(s.rotation.x), { 1, 0, 0 });
            model = glm::rotate(model, glm::radians(s.rotation.y), { 0, 1, 0 });
            model = glm::rotate(model, glm::radians(s.rotation.z), { 0, 0, 1 });
            model = glm::scale(model, s.scale);
            return model;
        }

        void updateUniformBuffers()
        {
            auto& ctx = graphics::internal::context;

            const float w = static_cast<float>(ctx.swapchain_extent.width);
            const float h = static_cast<float>(ctx.swapchain_extent.height);
            const float aspect = (h > 0.0f) ? (w / h) : 1.0f;

            const glm::mat4 view = glm::lookAt(glm::vec3(0.0f, 0.0f, 5.0f),
                glm::vec3(0.0f, 0.0f, 0.0f),
                glm::vec3(0.0f, 1.0f, 0.0f));

            glm::mat4 proj;
            if (g_use_perspective) {
                proj = glm::perspective(glm::radians(60.0f), aspect, 0.1f, 100.0f);
                proj[1][1] *= -1.0f; // Vulkan Y-flip
            }
            else {
                const float ortho_size = 3.0f;
                proj = glm::ortho(-ortho_size * aspect, ortho_size * aspect,
                    -ortho_size, ortho_size, 0.1f, 100.0f);
                proj[1][1] *= -1.0f;
            }

            for (int i = 0; i < OBJECT_COUNT; ++i) {
                UniformBufferObject ubo{};
                ubo.model = computeModel(g_objects[i]);
                ubo.view = view;
                ubo.proj = proj;
                ubo.color = g_objects[i].color;
                std::memcpy(g_uniform_buffers[i].mapped, &ubo, sizeof(ubo));
            }
        }

    } // namespace

    // ============================================================
    bool initialize()
    {
        std::cerr << "[app] initialize() start\n";

        std::vector<Vertex>   vertices;
        std::vector<uint32_t> indices;
        generateDodecahedron(vertices, indices);
        g_index_count = static_cast<uint32_t>(indices.size());
        std::cerr << "[app] dodecahedron: " << vertices.size() << " vertices, "
            << g_index_count << " indices\n";

        // Vertex buffer
        g_vertex_buffer = graphics::createBuffer(
            vertices.size() * sizeof(Vertex),
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            VMA_MEMORY_USAGE_AUTO, false,
            vertices.data());
        if (!g_vertex_buffer.buffer) { std::cerr << "[app] vertex buffer failed\n"; return false; }

        // Index buffer
        g_index_buffer = graphics::createBuffer(
            indices.size() * sizeof(uint32_t),
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            VMA_MEMORY_USAGE_AUTO, false,
            indices.data());
        if (!g_index_buffer.buffer) { std::cerr << "[app] index buffer failed\n"; return false; }

        // Uniform buffers
        for (int i = 0; i < OBJECT_COUNT; ++i) {
            g_uniform_buffers[i] = graphics::createBuffer(
                sizeof(UniformBufferObject),
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VMA_MEMORY_USAGE_AUTO, true);
            if (!g_uniform_buffers[i].buffer) {
                std::cerr << "[app] uniform buffer " << i << " failed\n";
                return false;
            }
        }

        // Descriptor set layout + pool + sets
        g_set_layout = graphics::createDescriptorSetLayout();
        g_descriptor_pool = graphics::createDescriptorPool(16);
        if (!g_descriptor_pool) { std::cerr << "[app] descriptor pool failed\n"; return false; }

        for (int i = 0; i < OBJECT_COUNT; ++i) {
            g_descriptor_sets[i] = graphics::allocateDescriptorSet(g_descriptor_pool, g_set_layout);
            if (!g_descriptor_sets[i]) {
                std::cerr << "[app] descriptor set " << i << " failed\n";
                return false;
            }

            VkDescriptorBufferInfo ubo_info{};
            ubo_info.buffer = g_uniform_buffers[i].buffer;
            ubo_info.offset = 0;
            ubo_info.range = sizeof(UniformBufferObject);

            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = g_descriptor_sets[i];
            write.dstBinding = 0;
            write.descriptorCount = 1;
            write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            write.pBufferInfo = &ubo_info;

            vkUpdateDescriptorSets(graphics::internal::context.device, 1, &write, 0, nullptr);
        }

        // Pipeline
        std::cerr << "[app] loading shaders...\n";
        VkShaderModule vert = graphics::loadShaderModule("shaders/object.vert.spv");
        VkShaderModule frag = graphics::loadShaderModule("shaders/object.frag.spv");
        if (!vert || !frag) {
            std::cerr << "[app] shader loading failed (vert=" << vert
                << ", frag=" << frag << ")\n";
            return false;
        }

        g_pipeline_layout = graphics::createPipelineLayout(g_set_layout);

        VkVertexInputBindingDescription binding{};
        binding.binding = 0;
        binding.stride = sizeof(Vertex);
        binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        std::vector<VkVertexInputAttributeDescription> attrs = {
            { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position) },
        };

        g_pipeline = graphics::createGraphicsPipeline(
            graphics::internal::context.render_pass,
            g_pipeline_layout,
            vert, frag,
            binding, attrs);

        graphics::destroyShaderModule(vert);
        graphics::destroyShaderModule(frag);

        if (!g_pipeline) { std::cerr << "[app] pipeline failed\n"; return false; }

        // Настройки по умолчанию для второго объекта
        g_objects[0].position = glm::vec3(-1.2f, 0.0f, 0.0f);
        g_objects[1].position = glm::vec3(1.2f, 0.0f, 0.0f);
        g_objects[1].scale = glm::vec3(0.6f);
        g_objects[1].color = glm::vec4(0.4f, 0.85f, 1.0f, 1.0f);

        std::cerr << "[app] initialize() OK\n";
        return true;
    }

    // ============================================================
    void shutdown()
    {
        auto& context = graphics::internal::context;
        vkQueueWaitIdle(context.graphics_queue);

        if (g_pipeline)        vkDestroyPipeline(context.device, g_pipeline, nullptr);
        if (g_pipeline_layout) vkDestroyPipelineLayout(context.device, g_pipeline_layout, nullptr);

        for (int i = 0; i < OBJECT_COUNT; ++i) {
            graphics::destroyBuffer(g_uniform_buffers[i]);
        }
        graphics::destroyBuffer(g_index_buffer);
        graphics::destroyBuffer(g_vertex_buffer);

        if (g_descriptor_pool) vkDestroyDescriptorPool(context.device, g_descriptor_pool, nullptr);
        if (g_set_layout)      vkDestroyDescriptorSetLayout(context.device, g_set_layout, nullptr);
    }

    // ============================================================
    void update(double time)
    {
        // Обновляем анимацию
        static double last_time = 0.0;
        const float dt = (last_time == 0.0) ? 0.0f : static_cast<float>(time - last_time);
        last_time = time;

        for (int i = 0; i < OBJECT_COUNT; ++i) {
            auto& o = g_objects[i];
            if (o.animate) {
                o.anim_phase += dt * o.anim_speed;
                o.position.x = o.anim_radius * std::cos(o.anim_phase);
                o.position.z = o.anim_radius * std::sin(o.anim_phase);
                o.rotation.y = glm::degrees(o.anim_phase) * 2.0f;
            }
        }

        // --- ImGui ---
        if (ImGui::Begin("Lab 1 - Controls")) {
            // Задание 1: переключение проекции
            ImGui::SeparatorText("Projection (task 1)");
            if (ImGui::RadioButton("Perspective", g_use_perspective)) {
                g_use_perspective = true;
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("Orthographic", !g_use_perspective)) {
                g_use_perspective = false;
            }

            // Задание 4: цвет — для каждого объекта
            for (int i = 0; i < OBJECT_COUNT; ++i) {
                auto& o = g_objects[i];
                ImGui::SeparatorText(("Object " + std::to_string(i)).c_str());

                // Задание 6: видимость
                ImGui::Checkbox("Visible##vis", &o.visible);

                // Задание 2: трансформации
                ImGui::SliderFloat3("Position", &o.position.x, -5.0f, 5.0f);
                ImGui::SliderFloat3("Rotation", &o.rotation.x, -180.0f, 180.0f);
                ImGui::SliderFloat3("Scale", &o.scale.x, 0.1f, 3.0f);

                // Задание 4: цвет
                ImGui::ColorEdit4("Color", &o.color.x);

                // Задание 3: анимация
                ImGui::Checkbox("Animate##a", &o.animate);
                ImGui::SliderFloat("Speed", &o.anim_speed, 0.0f, 5.0f);
                ImGui::SliderFloat("Radius", &o.anim_radius, 0.0f, 5.0f);
            }
        }
        ImGui::End();

        // Демо-окно ImGui (можно убрать)
        // ImGui::ShowDemoWindow();
    }

    // ============================================================
    void render(const graphics::internal::FrameData& fd)
    {
        auto& context = graphics::internal::context;

        updateUniformBuffers();

        vkResetCommandBuffer(fd.command_buffer, 0);

        const VkCommandBufferBeginInfo begin = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };
        vkBeginCommandBuffer(fd.command_buffer, &begin);

        const VkClearValue clear_values[] = {
            {.color = {.float32 = { 0.1f, 0.1f, 0.12f, 1.0f } } },
            {.depthStencil = {.depth = 1.0f, .stencil = 0 } },
        };

        const VkRenderPassBeginInfo rp = {
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass = context.render_pass,
            .framebuffer = fd.framebuffer,
            .renderArea = {.offset = {0,0}, .extent = context.swapchain_extent },
            .clearValueCount = 2,
            .pClearValues = clear_values,
        };
        vkCmdBeginRenderPass(fd.command_buffer, &rp, VK_SUBPASS_CONTENTS_INLINE);

        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = static_cast<float>(context.swapchain_extent.width);
        viewport.height = static_cast<float>(context.swapchain_extent.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);

        VkRect2D scissor{ {0, 0}, context.swapchain_extent };
        vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);

        vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, g_pipeline);

        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &g_vertex_buffer.buffer, &offset);
        vkCmdBindIndexBuffer(fd.command_buffer, g_index_buffer.buffer, 0, VK_INDEX_TYPE_UINT32);

        // Задание 6: рисуем каждый объект со своим descriptor set
        for (int i = 0; i < OBJECT_COUNT; ++i) {
            if (!g_objects[i].visible) continue;

            vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                g_pipeline_layout, 0, 1, &g_descriptor_sets[i], 0, nullptr);

            vkCmdDrawIndexed(fd.command_buffer, g_index_count, 1, 0, 0, 0);
        }

        vkCmdEndRenderPass(fd.command_buffer);
        vkEndCommandBuffer(fd.command_buffer);
    }

} // namespace application