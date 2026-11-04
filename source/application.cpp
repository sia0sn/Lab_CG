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

        struct Vertex {
            glm::vec3 position;
            glm::vec3 normal;
        };

        // Выравнивание памяти для GPU (std140)
        struct alignas(16) SceneUBO {
            glm::mat4 view;
            glm::mat4 proj;
            glm::vec4 cameraPos;
            glm::vec4 ambientColor;
            glm::vec4 dirLightDir;
            glm::vec4 dirLightColor;
        };

        struct alignas(16) ModelUBO {
            glm::mat4 model;
            glm::vec4 albedo;
            glm::vec4 specColor;
            glm::vec4 specParams; // x: shininess
        };

        struct alignas(16) PointLight {
            glm::vec4 position; // w - не используется (или intensity)
            glm::vec4 color;
            glm::vec4 params;   // x: constant, y: linear, z: quadratic
        };

        struct alignas(16) SpotLight {
            glm::vec4 position;
            glm::vec4 direction;
            glm::vec4 color;
            glm::vec4 params;   // x: constant, y: linear, z: quadratic, w: cos(inner)
            glm::vec4 params2;  // x: cos(outer)
        };

        // Global Vulkan Handles
        VkDescriptorSetLayout g_scene_layout = VK_NULL_HANDLE;
        VkDescriptorSetLayout g_model_layout = VK_NULL_HANDLE;
        VkPipelineLayout      g_object_pipeline_layout = VK_NULL_HANDLE;
        VkPipelineLayout      g_simple_pipeline_layout = VK_NULL_HANDLE;
        VkPipeline            g_object_pipeline = VK_NULL_HANDLE;
        VkPipeline            g_simple_pipeline = VK_NULL_HANDLE;
        VkDescriptorPool      g_descriptor_pool = VK_NULL_HANDLE;

        constexpr int MAX_LIGHTS = 10;

        Buffer g_vertex_buffer{};
        Buffer g_index_buffer{};
        uint32_t g_index_count = 0;

        Buffer g_scene_ubo{};
        Buffer g_point_ssbo{};
        Buffer g_spot_ssbo{};
        VkDescriptorSet g_scene_set = VK_NULL_HANDLE;

        // Data
        SceneUBO g_scene_data{};
        std::vector<PointLight> g_point_lights;
        std::vector<SpotLight> g_spot_lights;

        struct ModelData {
            ModelUBO ubo;
            Buffer ubo_buffer;
            VkDescriptorSet set;
            glm::vec3 position{ 0.0f };
            glm::vec3 rotation{ 0.0f };
            glm::vec3 scale{ 1.0f };
            glm::vec4 albedo{ 0.8f, 0.8f, 0.8f, 1.0f };
            glm::vec4 specColor{ 1.0f, 1.0f, 1.0f, 1.0f };
            float shininess = 32.0f;
        };

        ModelData g_object;

        // Camera State
        float g_cam_yaw = 0.0f;
        float g_cam_pitch = 0.0f;
        float g_cam_radius = 5.0f;
        float g_fov = 60.0f;

        // Visualize Light
        bool g_show_lights = true;
        ModelData g_light_boxes[MAX_LIGHTS * 2];
        Buffer g_cube_vertex_buffer{};
        Buffer g_cube_index_buffer{};
        uint32_t g_cube_index_count = 0;

        void generateDodecahedron(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices)
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
                { 0, 16,  2, 10,  8}, { 0, 12, 14,  4,  8}, { 0, 16, 17,  1, 12},
                { 2, 13, 15,  6, 10}, { 3, 11,  9,  1, 17}, { 3, 13, 15,  7, 11},
                { 8, 10,  6, 18,  4}, { 4, 18, 19,  5, 14}, { 5,  9, 11,  7, 19},
                { 5, 14, 12,  1,  9}, { 2, 16, 17,  3, 13}, { 6, 18, 19,  7, 15},
            };

            indices.clear();
            for (const auto& f : faces) {
                indices.push_back(f[0]); indices.push_back(f[1]); indices.push_back(f[2]);
                indices.push_back(f[0]); indices.push_back(f[2]); indices.push_back(f[3]);
                indices.push_back(f[0]); indices.push_back(f[3]); indices.push_back(f[4]);
            }

            for (auto& v : vertices) {
                v.position = glm::normalize(v.position) * 1.0f;
                v.normal = v.position; // Для додекаэдра нормаль = позиция (смотрит от центра)
            }
        }

        void generateCube(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices) {
            float s = 0.1f;
            vertices = {
                {{-s,-s,-s}, {0,0,-1}}, {{-s, s,-s}, {0,0,-1}}, {{ s, s,-s}, {0,0,-1}}, {{ s,-s,-s}, {0,0,-1}},
                {{-s,-s, s}, {0,0, 1}}, {{ s,-s, s}, {0,0, 1}}, {{ s, s, s}, {0,0, 1}}, {{-s, s, s}, {0,0, 1}},
                {{-s,-s,-s}, {-1,0,0}}, {{-s,-s, s}, {-1,0,0}}, {{-s, s, s}, {-1,0,0}}, {{-s, s,-s}, {-1,0,0}},
                {{ s,-s,-s}, { 1,0,0}}, {{ s, s,-s}, { 1,0,0}}, {{ s, s, s}, { 1,0,0}}, {{ s,-s, s}, { 1,0,0}},
                {{-s, s,-s}, {0, 1,0}}, {{-s, s, s}, {0, 1,0}}, {{ s, s, s}, {0, 1,0}}, {{ s, s,-s}, {0, 1,0}},
                {{-s,-s,-s}, {0,-1,0}}, {{ s,-s,-s}, {0,-1,0}}, {{ s,-s, s}, {0,-1,0}}, {{-s,-s, s}, {0,-1,0}},
            };
            indices = {
                0,1,2, 2,3,0,    4,5,6, 6,7,4,    8,9,10, 10,11,8,
                12,13,14, 14,15,12, 16,17,18, 18,19,16, 20,21,22, 22,23,20
            };
        }

        void updateSceneUniforms() {
            g_scene_data.cameraPos = glm::vec4(g_cam_radius * cos(glm::radians(g_cam_pitch)) * sin(glm::radians(g_cam_yaw)),
                g_cam_radius * sin(glm::radians(g_cam_pitch)),
                g_cam_radius * cos(glm::radians(g_cam_pitch)) * cos(glm::radians(g_cam_yaw)), 1.0f);

            g_scene_data.view = glm::lookAt(glm::vec3(g_scene_data.cameraPos), glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));

            float aspect = (float)graphics::internal::context.swapchain_extent.width / (float)graphics::internal::context.swapchain_extent.height;
            g_scene_data.proj = glm::perspective(glm::radians(g_fov), aspect, 0.1f, 100.0f);
            g_scene_data.proj[1][1] *= -1.0f; // Vulkan Y-flip

            std::memcpy(g_scene_ubo.mapped, &g_scene_data, sizeof(SceneUBO));

            // Обновляем SSBO для света
            if (g_point_lights.size() < MAX_LIGHTS) {
                // Записываем количество в 0-й элемент (хитрость)
                g_point_lights[0].position.w = (float)g_point_lights.size() - 1;
                std::memcpy(g_point_ssbo.mapped, g_point_lights.data(), sizeof(PointLight) * MAX_LIGHTS);
            }
            if (g_spot_lights.size() < MAX_LIGHTS) {
                g_spot_lights[0].position.w = (float)g_spot_lights.size() - 1;
                std::memcpy(g_spot_ssbo.mapped, g_spot_lights.data(), sizeof(SpotLight) * MAX_LIGHTS);
            }
        }

        void updateModelUniform(ModelData& model, const glm::vec4& overrideColor = glm::vec4(-1.0f)) {
            model.ubo.model = glm::mat4(1.0f);
            model.ubo.model = glm::translate(model.ubo.model, model.position);
            model.ubo.model = glm::rotate(model.ubo.model, glm::radians(model.rotation.x), { 1,0,0 });
            model.ubo.model = glm::rotate(model.ubo.model, glm::radians(model.rotation.y), { 0,1,0 });
            model.ubo.model = glm::rotate(model.ubo.model, glm::radians(model.rotation.z), { 0,0,1 });
            model.ubo.model = glm::scale(model.ubo.model, model.scale);

            if (overrideColor.x >= 0.0f) {
                model.ubo.albedo = overrideColor;
                model.ubo.specColor = glm::vec4(0.0f);
                model.ubo.specParams = glm::vec4(1.0f);
            }
            else {
                model.ubo.albedo = model.albedo;
                model.ubo.specColor = model.specColor;
                model.ubo.specParams = glm::vec4(model.shininess, 0.0f, 0.0f, 0.0f);
            }
            std::memcpy(model.ubo_buffer.mapped, &model.ubo, sizeof(ModelUBO));
        }

        void updateUniformBuffers() {
            updateSceneUniforms();
            updateModelUniform(g_object);

            // Обновляем кубики для света
            int boxIdx = 0;
            for (size_t i = 1; i < g_point_lights.size() && boxIdx < MAX_LIGHTS * 2; ++i) {
                g_light_boxes[boxIdx].position = glm::vec3(g_point_lights[i].position);
                g_light_boxes[boxIdx].scale = glm::vec3(1.0f);
                updateModelUniform(g_light_boxes[boxIdx], g_point_lights[i].color);
                boxIdx++;
            }
            for (size_t i = 1; i < g_spot_lights.size() && boxIdx < MAX_LIGHTS * 2; ++i) {
                g_light_boxes[boxIdx].position = glm::vec3(g_spot_lights[i].position);
                g_light_boxes[boxIdx].scale = glm::vec3(1.0f);
                updateModelUniform(g_light_boxes[boxIdx], g_spot_lights[i].color);
                boxIdx++;
            }
        }

    } // namespace

    // ============================================================
    bool initialize()
    {
        // Setup Default lights
        PointLight defPoint{};
        defPoint.params = glm::vec4(1.0f, 0.14f, 0.07f, 1.0f); // constant, linear, quadratic
        g_point_lights.push_back(defPoint); // 0-й элемент - заглушка для счетчика

        SpotLight defSpot{};
        defSpot.params = glm::vec4(1.0f, 0.14f, 0.07f, cos(glm::radians(12.5f)));
        defSpot.params2 = glm::vec4(cos(glm::radians(25.0f)), 0.0f, 0.0f, 0.0f);
        g_spot_lights.push_back(defSpot);

        // Default scene
        g_scene_data.ambientColor = glm::vec4(0.1f, 0.1f, 0.1f, 1.0f);
        g_scene_data.dirLightDir = glm::vec4(-0.5f, -1.0f, -0.3f, 0.0f);
        g_scene_data.dirLightColor = glm::vec4(0.5f, 0.5f, 0.5f, 1.0f);

        // Geometry
        std::vector<Vertex> vertices, cubeVerts;
        std::vector<uint32_t> indices, cubeIndices;
        generateDodecahedron(vertices, indices);
        generateCube(cubeVerts, cubeIndices);
        g_index_count = static_cast<uint32_t>(indices.size());
        g_cube_index_count = static_cast<uint32_t>(cubeIndices.size());

        g_vertex_buffer = graphics::createBuffer(vertices.size() * sizeof(Vertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO, false, vertices.data());
        g_index_buffer = graphics::createBuffer(indices.size() * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO, false, indices.data());
        g_cube_vertex_buffer = graphics::createBuffer(cubeVerts.size() * sizeof(Vertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO, false, cubeVerts.data());
        g_cube_index_buffer = graphics::createBuffer(cubeIndices.size() * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO, false, cubeIndices.data());

        // Buffers
        g_scene_ubo = graphics::createBuffer(sizeof(SceneUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO, true);
        g_point_ssbo = graphics::createBuffer(sizeof(PointLight) * MAX_LIGHTS, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO, true);
        g_spot_ssbo = graphics::createBuffer(sizeof(SpotLight) * MAX_LIGHTS, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO, true);

        g_object.ubo_buffer = graphics::createBuffer(sizeof(ModelUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO, true);
        for (int i = 0; i < MAX_LIGHTS * 2; ++i) {
            g_light_boxes[i].ubo_buffer = graphics::createBuffer(sizeof(ModelUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO, true);
        }

        // Descriptors
        g_scene_layout = graphics::createSceneSetLayout();
        g_model_layout = graphics::createModelSetLayout();
        g_descriptor_pool = graphics::createDescriptorPool(MAX_LIGHTS * 3); // запас

        g_scene_set = graphics::allocateDescriptorSet(g_descriptor_pool, g_scene_layout);
        g_object.set = graphics::allocateDescriptorSet(g_descriptor_pool, g_model_layout);
        for (int i = 0; i < MAX_LIGHTS * 2; ++i) {
            g_light_boxes[i].set = graphics::allocateDescriptorSet(g_descriptor_pool, g_model_layout);
        }

        // Write Descriptors
        VkDescriptorBufferInfo scene_info{ g_scene_ubo.buffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo point_info{ g_point_ssbo.buffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo spot_info{ g_spot_ssbo.buffer, 0, VK_WHOLE_SIZE };

        VkWriteDescriptorSet scene_writes[3]{};
        scene_writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; scene_writes[0].dstSet = g_scene_set; scene_writes[0].dstBinding = 0; scene_writes[0].descriptorCount = 1; scene_writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; scene_writes[0].pBufferInfo = &scene_info;
        scene_writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; scene_writes[1].dstSet = g_scene_set; scene_writes[1].dstBinding = 1; scene_writes[1].descriptorCount = 1; scene_writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; scene_writes[1].pBufferInfo = &point_info;
        scene_writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; scene_writes[2].dstSet = g_scene_set; scene_writes[2].dstBinding = 2; scene_writes[2].descriptorCount = 1; scene_writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; scene_writes[2].pBufferInfo = &spot_info;
        vkUpdateDescriptorSets(graphics::internal::context.device, 3, scene_writes, 0, nullptr);

        auto writeModelSet = [&](VkDescriptorSet set, Buffer& buf) {
            VkDescriptorBufferInfo info{ buf.buffer, 0, VK_WHOLE_SIZE };
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; write.dstSet = set; write.dstBinding = 0; write.descriptorCount = 1; write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; write.pBufferInfo = &info;
            vkUpdateDescriptorSets(graphics::internal::context.device, 1, &write, 0, nullptr);
            };
        writeModelSet(g_object.set, g_object.ubo_buffer);
        for (int i = 0; i < MAX_LIGHTS * 2; ++i) writeModelSet(g_light_boxes[i].set, g_light_boxes[i].ubo_buffer);

        // Pipelines
        g_object_pipeline_layout = graphics::createPipelineLayout(g_model_layout); // Пайплайн лайаут не знает про два сета :( Нужно изменить.
        // ЖДЕМ: мы поменяли graphics.cpp, там VkPipelineLayoutCreateInfo должен брать массив.
        // (В коде ниже я поправил вызов, но для этого нужно обновить graphics.hpp/cpp, чтобы они принимали массив)
        VkDescriptorSetLayout layouts[] = { g_scene_layout, g_model_layout };
        VkPipelineLayoutCreateInfo pl_info{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        pl_info.setLayoutCount = 2;
        pl_info.pSetLayouts = layouts;
        vkCreatePipelineLayout(graphics::internal::context.device, &pl_info, nullptr, &g_object_pipeline_layout);
        vkCreatePipelineLayout(graphics::internal::context.device, &pl_info, nullptr, &g_simple_pipeline_layout);

        VkShaderModule vert = graphics::loadShaderModule("shaders/object.vert.spv");
        VkShaderModule frag = graphics::loadShaderModule("shaders/object.frag.spv");
        VkVertexInputBindingDescription binding{ 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX };
        std::vector<VkVertexInputAttributeDescription> attrs = { { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position) }, { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal) } };
        g_object_pipeline = graphics::createGraphicsPipeline(graphics::internal::context.render_pass, g_object_pipeline_layout, vert, frag, binding, attrs);
        graphics::destroyShaderModule(vert); graphics::destroyShaderModule(frag);

        vert = graphics::loadShaderModule("shaders/simple.vert.spv");
        frag = graphics::loadShaderModule("shaders/simple.frag.spv");
        g_simple_pipeline = graphics::createGraphicsPipeline(graphics::internal::context.render_pass, g_simple_pipeline_layout, vert, frag, binding, attrs);
        graphics::destroyShaderModule(vert); graphics::destroyShaderModule(frag);

        return g_object_pipeline != VK_NULL_HANDLE;
    }

    // ============================================================
    void shutdown()
    {
        auto& context = graphics::internal::context;
        vkQueueWaitIdle(context.graphics_queue);

        if (g_object_pipeline) vkDestroyPipeline(context.device, g_object_pipeline, nullptr);
        if (g_simple_pipeline) vkDestroyPipeline(context.device, g_simple_pipeline, nullptr);
        if (g_object_pipeline_layout) vkDestroyPipelineLayout(context.device, g_object_pipeline_layout, nullptr);
        if (g_simple_pipeline_layout) vkDestroyPipelineLayout(context.device, g_simple_pipeline_layout, nullptr);

        graphics::destroyBuffer(g_scene_ubo);
        graphics::destroyBuffer(g_point_ssbo);
        graphics::destroyBuffer(g_spot_ssbo);

        graphics::destroyBuffer(g_object.ubo_buffer);
        for (int i = 0; i < MAX_LIGHTS * 2; ++i) {
            graphics::destroyBuffer(g_light_boxes[i].ubo_buffer);
        }

        graphics::destroyBuffer(g_cube_index_buffer);
        graphics::destroyBuffer(g_cube_vertex_buffer);
        graphics::destroyBuffer(g_index_buffer);
        graphics::destroyBuffer(g_vertex_buffer);

        if (g_descriptor_pool) vkDestroyDescriptorPool(context.device, g_descriptor_pool, nullptr);
        if (g_model_layout) vkDestroyDescriptorSetLayout(context.device, g_model_layout, nullptr);
        if (g_scene_layout) vkDestroyDescriptorSetLayout(context.device, g_scene_layout, nullptr);
    }

    // ============================================================
    void update(double time)
    {
        if (ImGui::Begin("Lab 2 - Camera & Lighting")) {
            ImGui::SeparatorText("Camera");
            ImGui::SliderFloat("Yaw", &g_cam_yaw, -180.0f, 180.0f);
            ImGui::SliderFloat("Pitch", &g_cam_pitch, -89.0f, 89.0f);
            ImGui::SliderFloat("Radius", &g_cam_radius, 1.0f, 20.0f);
            ImGui::SliderFloat("FOV", &g_fov, 30.0f, 90.0f);

            ImGui::SeparatorText("Ambient & Directional Light");
            ImGui::ColorEdit3("Ambient", &g_scene_data.ambientColor.x);
            ImGui::ColorEdit3("Dir Color", &g_scene_data.dirLightColor.x);
            ImGui::SliderFloat3("Dir Direction", &g_scene_data.dirLightDir.x, -1.0f, 1.0f);

            ImGui::SeparatorText("Material (Dodecahedron)");
            ImGui::ColorEdit3("Albedo", &g_object.albedo.x);
            ImGui::ColorEdit3("Specular", &g_object.specColor.x);
            ImGui::SliderFloat("Shininess", &g_object.shininess, 1.0f, 256.0f);

            ImGui::SeparatorText("Point Lights (Task 2)");
            if (ImGui::Button("Add Point Light") && g_point_lights.size() < MAX_LIGHTS) {
                PointLight l{};
                l.position = glm::vec4((rand() % 100) / 50.0f - 1.0f, 1.0f, (rand() % 100) / 50.0f - 1.0f, 1.0f);
                l.color = glm::vec4(1.0f, 0.5f, 0.2f, 1.0f);
                l.params = glm::vec4(1.0f, 0.14f, 0.07f, 1.0f);
                g_point_lights.push_back(l);
            }
            for (size_t i = 1; i < g_point_lights.size(); ++i) {
                ImGui::PushID((int)i);
                ImGui::Text("Point %d", (int)i);
                ImGui::SliderFloat3("Pos", &g_point_lights[i].position.x, -5.0f, 5.0f);
                ImGui::ColorEdit3("Col", &g_point_lights[i].color.x);
                if (ImGui::Button("Remove")) { g_point_lights.erase(g_point_lights.begin() + i); ImGui::PopID(); break; }
                ImGui::PopID();
            }

            ImGui::SeparatorText("Spot Lights (Task 3)");
            if (ImGui::Button("Add Spot Light") && g_spot_lights.size() < MAX_LIGHTS) {
                SpotLight l{};
                l.position = glm::vec4(2.0f, 2.0f, 2.0f, 1.0f);
                l.direction = glm::vec4(-1.0f, -1.0f, -1.0f, 0.0f);
                l.color = glm::vec4(0.2f, 0.5f, 1.0f, 1.0f);
                l.params = glm::vec4(1.0f, 0.14f, 0.07f, cos(glm::radians(12.5f)));
                l.params2 = glm::vec4(cos(glm::radians(25.0f)), 0.0f, 0.0f, 0.0f);
                g_spot_lights.push_back(l);
            }
            for (size_t i = 1; i < g_spot_lights.size(); ++i) {
                ImGui::PushID((int)(i + 100));
                ImGui::Text("Spot %d", (int)i);
                ImGui::SliderFloat3("Pos", &g_spot_lights[i].position.x, -5.0f, 5.0f);
                ImGui::SliderFloat3("Dir", &g_spot_lights[i].direction.x, -1.0f, 1.0f);
                ImGui::ColorEdit3("Col", &g_spot_lights[i].color.x);
                if (ImGui::Button("Remove")) { g_spot_lights.erase(g_spot_lights.begin() + i); ImGui::PopID(); break; }
                ImGui::PopID();
            }

            ImGui::Separator();
            ImGui::Checkbox("Show Light Sources (Task 4)", &g_show_lights);
        }
        ImGui::End();
    }

    // ============================================================
    void render(const graphics::internal::FrameData& fd)
    {
        auto& context = graphics::internal::context;
        updateUniformBuffers();

        vkResetCommandBuffer(fd.command_buffer, 0);
        VkCommandBufferBeginInfo begin{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(fd.command_buffer, &begin);

        VkClearValue clear_values[] = {
            {.color = {{0.1f, 0.1f, 0.12f, 1.0f}} },
            {.depthStencil = {1.0f, 0} }
        };
        VkRenderPassBeginInfo rp{ VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
        rp.renderPass = context.render_pass;
        rp.framebuffer = fd.framebuffer;
        rp.renderArea = { {0,0}, context.swapchain_extent };
        rp.clearValueCount = 2;
        rp.pClearValues = clear_values;
        vkCmdBeginRenderPass(fd.command_buffer, &rp, VK_SUBPASS_CONTENTS_INLINE);

        VkViewport viewport{ 0, 0, (float)context.swapchain_extent.width, (float)context.swapchain_extent.height, 0, 1 };
        VkRect2D scissor{ {0, 0}, context.swapchain_extent };
        vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);
        vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);

        // 1. Рисуем додекаэдр (с Блинн-Фонгом)
        vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, g_object_pipeline);
        vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, g_object_pipeline_layout, 0, 1, &g_scene_set, 0, nullptr);
        vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, g_object_pipeline_layout, 1, 1, &g_object.set, 0, nullptr);

        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &g_vertex_buffer.buffer, &offset);
        vkCmdBindIndexBuffer(fd.command_buffer, g_index_buffer.buffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(fd.command_buffer, g_index_count, 1, 0, 0, 0);

        // 2. Рисуем светящиеся кубики (Визуализация источников)
        if (g_show_lights) {
            vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, g_simple_pipeline);
            vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, g_simple_pipeline_layout, 0, 1, &g_scene_set, 0, nullptr);
            vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &g_cube_vertex_buffer.buffer, &offset);
            vkCmdBindIndexBuffer(fd.command_buffer, g_cube_index_buffer.buffer, 0, VK_INDEX_TYPE_UINT32);

            int boxIdx = 0;
            // Кубики для точечных источников
            for (size_t i = 1; i < g_point_lights.size(); ++i) {
                vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, g_simple_pipeline_layout, 1, 1, &g_light_boxes[boxIdx].set, 0, nullptr);
                vkCmdDrawIndexed(fd.command_buffer, g_cube_index_count, 1, 0, 0, 0);
                boxIdx++;
            }
            // Кубики для прожекторов
            for (size_t i = 1; i < g_spot_lights.size(); ++i) {
                vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, g_simple_pipeline_layout, 1, 1, &g_light_boxes[boxIdx].set, 0, nullptr);
                vkCmdDrawIndexed(fd.command_buffer, g_cube_index_count, 1, 0, 0, 0);
                boxIdx++;
            }
        }

        vkCmdEndRenderPass(fd.command_buffer);
        vkEndCommandBuffer(fd.command_buffer);
    }

} // namespace application