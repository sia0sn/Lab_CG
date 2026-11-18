#include "application.hpp"
#include "graphics.hpp"

#include <imgui.h>
#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <GLFW/glfw3.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>
#include <fstream>

namespace application {

    namespace {

        using graphics::Buffer;

        struct Vertex {
            glm::vec3 position;
            glm::vec3 normal;
            glm::vec2 uv;
        };

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
            glm::vec4 specParams;
        };

        struct alignas(16) PointLight {
            glm::vec4 position;
            glm::vec4 color;
            glm::vec4 params;
        };

        struct alignas(16) SpotLight {
            glm::vec4 position;
            glm::vec4 direction;
            glm::vec4 color;
            glm::vec4 params;
            glm::vec4 params2;
        };

        struct Texture {
            VkImage image = VK_NULL_HANDLE;
            VmaAllocation allocation = VK_NULL_HANDLE;
            VkImageView view = VK_NULL_HANDLE;
            VkSampler sampler = VK_NULL_HANDLE;
        };

        // Глобальные переменные
        VkDescriptorSetLayout g_scene_layout = VK_NULL_HANDLE;
        VkDescriptorSetLayout g_model_layout = VK_NULL_HANDLE;
        VkDescriptorSetLayout g_texture_layout = VK_NULL_HANDLE;
        VkPipelineLayout      g_object_pipeline_layout = VK_NULL_HANDLE;
        VkPipelineLayout      g_simple_pipeline_layout = VK_NULL_HANDLE;
        VkPipelineLayout      g_line_pipeline_layout = VK_NULL_HANDLE;
        VkPipeline            g_object_pipeline = VK_NULL_HANDLE;
        VkPipeline            g_simple_pipeline = VK_NULL_HANDLE;
        VkPipeline            g_line_pipeline = VK_NULL_HANDLE;
        VkDescriptorPool      g_descriptor_pool = VK_NULL_HANDLE;

        constexpr int MAX_LIGHTS = 10;
        constexpr int NUM_ALBEDO_TEXTURES = 3;

        Buffer g_vertex_buffer{};
        Buffer g_index_buffer{};
        uint32_t g_index_count = 0;

        Buffer g_cube_vertex_buffer{};
        Buffer g_cube_index_buffer{};
        uint32_t g_cube_index_count = 0;

        Buffer g_line_vertex_buffer{};
        uint32_t g_line_vertex_count = 0;

        Buffer g_scene_ubo{};
        Buffer g_point_ssbo{};
        Buffer g_spot_ssbo{};
        VkDescriptorSet g_scene_set = VK_NULL_HANDLE;

        Texture g_albedo_textures[NUM_ALBEDO_TEXTURES];
        Texture g_specular_texture;
        Texture g_emissive_texture;

        SceneUBO g_scene_data{};
        std::vector<PointLight> g_point_lights;
        std::vector<SpotLight> g_spot_lights;

        struct ModelData {
            ModelUBO ubo;
            Buffer ubo_buffer;
            VkDescriptorSet model_set;
            VkDescriptorSet texture_set; // Свой набор текстур для каждого объекта

            glm::vec3 position{ 0.0f };
            glm::vec3 rotation{ 0.0f };
            glm::vec3 scale{ 1.0f };

            // Свойства материала
            glm::vec4 albedo{ 0.8f, 0.8f, 0.8f, 1.0f };
            glm::vec4 specColor{ 1.0f, 1.0f, 1.0f, 1.0f };
            float shininess = 128.0f;

            // Анимация и текстуры
            bool visible = true;
            bool animate = false;
            float anim_speed = 1.0f;
            float anim_radius = 2.0f;
            float anim_phase = 0.0f;

            int current_albedo_index = 0;
            bool enable_emissive = true;
            bool enable_specular = true;
        };

        ModelData g_objects[2]; // Два объекта!
        ModelData g_light_boxes[MAX_LIGHTS * 2];
        ModelData g_spot_cones[MAX_LIGHTS];

        // Камера
        float g_cam_yaw = 0.0f;
        float g_cam_pitch = 0.0f;
        float g_cam_radius = 5.0f;
        float g_fov = 60.0f;

        bool g_show_lights = true;

        // ==========================================
        // ГЕНЕРАЦИЯ ГЕОМЕТРИИ
        // ==========================================

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
                v.normal = v.position;
                v.uv = glm::vec2(v.normal.x * 0.5 + 0.5, 1.0 - (v.normal.y * 0.5 + 0.5));
            }
        }

        void generateCube(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices) {
            float s = 0.1f;
            vertices = {
                {{-s,-s,-s}, {0,0,-1}, {0,0}}, {{-s, s,-s}, {0,0,-1}, {0,0}}, {{ s, s,-s}, {0,0,-1}, {0,0}}, {{ s,-s,-s}, {0,0,-1}, {0,0}},
                {{-s,-s, s}, {0,0, 1}, {0,0}}, {{ s,-s, s}, {0,0, 1}, {0,0}}, {{ s, s, s}, {0,0, 1}, {0,0}}, {{-s, s, s}, {0,0, 1}, {0,0}},
                {{-s,-s,-s}, {-1,0,0}, {0,0}}, {{-s,-s, s}, {-1,0,0}, {0,0}}, {{-s, s, s}, {-1,0,0}, {0,0}}, {{-s, s,-s}, {-1,0,0}, {0,0}},
                {{ s,-s,-s}, { 1,0,0}, {0,0}}, {{ s, s,-s}, { 1,0,0}, {0,0}}, {{ s, s, s}, { 1,0,0}, {0,0}}, {{ s,-s, s}, { 1,0,0}, {0,0}},
                {{-s, s,-s}, {0, 1,0}, {0,0}}, {{-s, s, s}, {0, 1,0}, {0,0}}, {{ s, s, s}, {0, 1,0}, {0,0}}, {{ s, s,-s}, {0, 1,0}, {0,0}},
                {{-s,-s,-s}, {0,-1,0}, {0,0}}, {{ s,-s,-s}, {0,-1,0}, {0,0}}, {{ s,-s, s}, {0,-1,0}, {0,0}}, {{-s,-s, s}, {0,-1,0}, {0,0}},
            };
            indices = {
                0,1,2, 2,3,0,    4,5,6, 6,7,4,    8,9,10, 10,11,8,
                12,13,14, 14,15,12, 16,17,18, 18,19,16, 20,21,22, 22,23,20
            };
        }

        void generateConeLines(std::vector<Vertex>& vertices) {
            vertices.clear();
            float length = 2.0f;
            float outerAngle = glm::radians(25.0f);
            float radius = length * std::tan(outerAngle);

            glm::vec3 apex(0, 0, 0);
            const int segments = 16;
            std::vector<glm::vec3> circlePoints;

            for (int i = 0; i < segments; ++i) {
                float angle = (float)i / segments * 2.0f * 3.14159265f;
                circlePoints.push_back(glm::vec3(radius * std::cos(angle), radius * std::sin(angle), -length));
            }

            for (int i = 0; i < segments; ++i) {
                vertices.push_back({ apex, {}, {} });
                vertices.push_back({ circlePoints[i], {}, {} });
            }

            for (int i = 0; i < segments; ++i) {
                vertices.push_back({ circlePoints[i], {}, {} });
                vertices.push_back({ circlePoints[(i + 1) % segments], {}, {} });
            }
        }

        // ==========================================
        // ЗАГРУЗКА ТЕКСТУР
        // ==========================================

        Texture createTexture(const char* filename) {
            Texture tex;
            std::string real_path = filename;
            std::vector<std::string> candidates = { filename, "../" + std::string(filename), "../../" + std::string(filename) };
            for (const auto& path : candidates) {
                std::ifstream f(path, std::ios::binary);
                if (f.good()) { real_path = path; break; }
            }

            int texWidth, texHeight, texChannels;
            stbi_uc* pixels = stbi_load(real_path.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
            VkDeviceSize imageSize = texWidth * texHeight * 4;

            if (!pixels) {
                std::cerr << "Failed to load texture file: " << real_path << std::endl;
                return tex;
            }

            Buffer stagingBuffer = graphics::createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_AUTO, false, pixels);
            stbi_image_free(pixels);

            VkImageCreateInfo imageInfo{};
            imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            imageInfo.imageType = VK_IMAGE_TYPE_2D;
            imageInfo.extent.width = texWidth;
            imageInfo.extent.height = texHeight;
            imageInfo.extent.depth = 1;
            imageInfo.mipLevels = 1;
            imageInfo.arrayLayers = 1;
            imageInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
            imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
            imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
            imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
            imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

            VmaAllocationCreateInfo allocInfo{};
            allocInfo.usage = VMA_MEMORY_USAGE_AUTO;

            vmaCreateImage(graphics::internal::context.allocator, &imageInfo, &allocInfo, &tex.image, &tex.allocation, nullptr);

            VkCommandPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
            poolInfo.queueFamilyIndex = graphics::internal::context.graphics_queue_index;
            VkCommandPool tempPool;
            vkCreateCommandPool(graphics::internal::context.device, &poolInfo, nullptr, &tempPool);

            VkCommandBufferAllocateInfo allocCmd{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
            allocCmd.commandPool = tempPool;
            allocCmd.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocCmd.commandBufferCount = 1;
            VkCommandBuffer cmd;
            vkAllocateCommandBuffers(graphics::internal::context.device, &allocCmd, &cmd);

            VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkBeginCommandBuffer(cmd, &beginInfo);

            VkImageMemoryBarrier barrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
            barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = tex.image;
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            barrier.subresourceRange.levelCount = 1;
            barrier.subresourceRange.layerCount = 1;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

            VkBufferImageCopy region{};
            region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.imageSubresource.layerCount = 1;
            region.imageExtent = { (uint32_t)texWidth, (uint32_t)texHeight, 1 };
            vkCmdCopyBufferToImage(cmd, stagingBuffer.buffer, tex.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

            vkEndCommandBuffer(cmd);
            VkSubmitInfo submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &cmd;
            vkQueueSubmit(graphics::internal::context.graphics_queue, 1, &submitInfo, VK_NULL_HANDLE);
            vkQueueWaitIdle(graphics::internal::context.graphics_queue);
            vkDestroyCommandPool(graphics::internal::context.device, tempPool, nullptr);
            graphics::destroyBuffer(stagingBuffer);

            VkImageViewCreateInfo viewInfo{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
            viewInfo.image = tex.image;
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
            viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            viewInfo.subresourceRange.levelCount = 1;
            viewInfo.subresourceRange.layerCount = 1;
            vkCreateImageView(graphics::internal::context.device, &viewInfo, nullptr, &tex.view);

            VkSamplerCreateInfo samplerInfo{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
            samplerInfo.magFilter = VK_FILTER_LINEAR;
            samplerInfo.minFilter = VK_FILTER_LINEAR;
            samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
            vkCreateSampler(graphics::internal::context.device, &samplerInfo, nullptr, &tex.sampler);

            return tex;
        }

        // ==========================================
        // ОБНОВЛЕНИЕ ДАННЫХ
        // ==========================================

        void updateSceneUniforms() {
            g_scene_data.cameraPos = glm::vec4(g_cam_radius * cos(glm::radians(g_cam_pitch)) * sin(glm::radians(g_cam_yaw)),
                g_cam_radius * sin(glm::radians(g_cam_pitch)),
                g_cam_radius * cos(glm::radians(g_cam_pitch)) * cos(glm::radians(g_cam_yaw)), 1.0f);

            g_scene_data.view = glm::lookAt(glm::vec3(g_scene_data.cameraPos), glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));

            float aspect = (float)graphics::internal::context.swapchain_extent.width / (float)graphics::internal::context.swapchain_extent.height;
            g_scene_data.proj = glm::perspective(glm::radians(g_fov), aspect, 0.1f, 100.0f);
            g_scene_data.proj[1][1] *= -1.0f;

            std::memcpy(g_scene_ubo.mapped, &g_scene_data, sizeof(SceneUBO));
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

    } // namespace

    // ==========================================
    // ИНИЦИАЛИЗАЦИЯ И ОТРИСОВКА
    // ==========================================

    bool initialize()
    {
        // 1. Загрузка текстур
        g_albedo_textures[0] = createTexture("textures/albedo1.jpg");
        g_albedo_textures[1] = createTexture("textures/albedo2.jpg");
        g_albedo_textures[2] = createTexture("textures/albedo3.jpg");
        g_specular_texture = createTexture("textures/specular.jpg");
        g_emissive_texture = createTexture("textures/emissive.jpg");

        // 2. Настройка объектов
        g_objects[0].position = glm::vec3(-1.2f, 0.0f, 0.0f);
        g_objects[0].current_albedo_index = 0; // Смайлик

        g_objects[1].position = glm::vec3(1.2f, 0.0f, 0.0f);
        g_objects[1].scale = glm::vec3(0.6f);
        g_objects[1].current_albedo_index = 2; // Единорог
        g_objects[1].anim_phase = 3.14159f; // В противофазе

        PointLight defPoint{};
        defPoint.params = glm::vec4(1.0f, 0.14f, 0.07f, 1.0f);
        g_point_lights.push_back(defPoint);

        SpotLight defSpot{};
        defSpot.params = glm::vec4(1.0f, 0.14f, 0.07f, cos(glm::radians(12.5f)));
        defSpot.params2 = glm::vec4(cos(glm::radians(25.0f)), 0.0f, 0.0f, 0.0f);
        g_spot_lights.push_back(defSpot);

        g_scene_data.ambientColor = glm::vec4(0.1f, 0.1f, 0.1f, 1.0f);
        g_scene_data.dirLightDir = glm::vec4(-0.5f, -1.0f, -0.3f, 0.0f);
        g_scene_data.dirLightColor = glm::vec4(0.5f, 0.5f, 0.5f, 1.0f);

        std::vector<Vertex> vertices, cubeVerts, lineVerts;
        std::vector<uint32_t> indices, cubeIndices;
        generateDodecahedron(vertices, indices);
        generateCube(cubeVerts, cubeIndices);
        generateConeLines(lineVerts);

        g_index_count = static_cast<uint32_t>(indices.size());
        g_cube_index_count = static_cast<uint32_t>(cubeIndices.size());
        g_line_vertex_count = static_cast<uint32_t>(lineVerts.size());

        g_vertex_buffer = graphics::createBuffer(vertices.size() * sizeof(Vertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO, false, vertices.data());
        g_index_buffer = graphics::createBuffer(indices.size() * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO, false, indices.data());
        g_cube_vertex_buffer = graphics::createBuffer(cubeVerts.size() * sizeof(Vertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO, false, cubeVerts.data());
        g_cube_index_buffer = graphics::createBuffer(cubeIndices.size() * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO, false, cubeIndices.data());
        g_line_vertex_buffer = graphics::createBuffer(lineVerts.size() * sizeof(Vertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO, false, lineVerts.data());

        g_scene_ubo = graphics::createBuffer(sizeof(SceneUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO, true);
        g_point_ssbo = graphics::createBuffer(sizeof(PointLight) * MAX_LIGHTS, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO, true);
        g_spot_ssbo = graphics::createBuffer(sizeof(SpotLight) * MAX_LIGHTS, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO, true);

        for (int i = 0; i < 2; ++i) {
            g_objects[i].ubo_buffer = graphics::createBuffer(sizeof(ModelUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO, true);
        }
        for (int i = 0; i < MAX_LIGHTS * 2; ++i) {
            g_light_boxes[i].ubo_buffer = graphics::createBuffer(sizeof(ModelUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO, true);
        }
        for (int i = 0; i < MAX_LIGHTS; ++i) {
            g_spot_cones[i].ubo_buffer = graphics::createBuffer(sizeof(ModelUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO, true);
        }

        g_scene_layout = graphics::createSceneSetLayout();
        g_model_layout = graphics::createModelSetLayout();
        g_texture_layout = graphics::createTextureSetLayout();
        g_descriptor_pool = graphics::createDescriptorPool(MAX_LIGHTS * 6);

        g_scene_set = graphics::allocateDescriptorSet(g_descriptor_pool, g_scene_layout);

        for (int i = 0; i < 2; ++i) {
            g_objects[i].model_set = graphics::allocateDescriptorSet(g_descriptor_pool, g_model_layout);
            g_objects[i].texture_set = graphics::allocateDescriptorSet(g_descriptor_pool, g_texture_layout);
        }
        for (int i = 0; i < MAX_LIGHTS * 2; ++i) g_light_boxes[i].model_set = graphics::allocateDescriptorSet(g_descriptor_pool, g_model_layout);
        for (int i = 0; i < MAX_LIGHTS; ++i) g_spot_cones[i].model_set = graphics::allocateDescriptorSet(g_descriptor_pool, g_model_layout);

        // Привязка дескрипторов
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
        for (int i = 0; i < 2; ++i) writeModelSet(g_objects[i].model_set, g_objects[i].ubo_buffer);
        for (int i = 0; i < MAX_LIGHTS * 2; ++i) writeModelSet(g_light_boxes[i].model_set, g_light_boxes[i].ubo_buffer);
        for (int i = 0; i < MAX_LIGHTS; ++i) writeModelSet(g_spot_cones[i].model_set, g_spot_cones[i].ubo_buffer);

        auto writeTextureSet = [&](VkDescriptorSet set, int albedoIdx) {
            VkDescriptorImageInfo albedoInfo{ g_albedo_textures[albedoIdx].sampler, g_albedo_textures[albedoIdx].view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
            VkDescriptorImageInfo specInfo{ g_specular_texture.sampler, g_specular_texture.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
            VkDescriptorImageInfo emisInfo{ g_emissive_texture.sampler, g_emissive_texture.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };

            VkWriteDescriptorSet texWrites[3]{};
            texWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; texWrites[0].dstSet = set; texWrites[0].dstBinding = 0; texWrites[0].descriptorCount = 1; texWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; texWrites[0].pImageInfo = &albedoInfo;
            texWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; texWrites[1].dstSet = set; texWrites[1].dstBinding = 1; texWrites[1].descriptorCount = 1; texWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; texWrites[1].pImageInfo = &specInfo;
            texWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; texWrites[2].dstSet = set; texWrites[2].dstBinding = 2; texWrites[2].descriptorCount = 1; texWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; texWrites[2].pImageInfo = &emisInfo;
            vkUpdateDescriptorSets(graphics::internal::context.device, 3, texWrites, 0, nullptr);
            };
        for (int i = 0; i < 2; ++i) writeTextureSet(g_objects[i].texture_set, g_objects[i].current_albedo_index);

        // Pipeline Layout
        VkDescriptorSetLayout layouts[] = { g_scene_layout, g_model_layout, g_texture_layout };
        VkPipelineLayoutCreateInfo pl_info{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        pl_info.setLayoutCount = 3;
        pl_info.pSetLayouts = layouts;

        VkPushConstantRange push_constant{};
        push_constant.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        push_constant.offset = 0;
        push_constant.size = sizeof(float) * 4; // 4 флоата теперь

        pl_info.pushConstantRangeCount = 1;
        pl_info.pPushConstantRanges = &push_constant;

        vkCreatePipelineLayout(graphics::internal::context.device, &pl_info, nullptr, &g_object_pipeline_layout);
        vkCreatePipelineLayout(graphics::internal::context.device, &pl_info, nullptr, &g_simple_pipeline_layout);
        vkCreatePipelineLayout(graphics::internal::context.device, &pl_info, nullptr, &g_line_pipeline_layout);

        VkShaderModule vert = graphics::loadShaderModule("shaders/object.vert.spv");
        VkShaderModule frag = graphics::loadShaderModule("shaders/object.frag.spv");
        VkVertexInputBindingDescription binding{ 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX };
        std::vector<VkVertexInputAttributeDescription> attrs = {
            { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position) },
            { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal) },
            { 2, 0, VK_FORMAT_R32G32_SFLOAT,    offsetof(Vertex, uv) }
        };

        g_object_pipeline = graphics::createGraphicsPipeline(graphics::internal::context.render_pass, g_object_pipeline_layout, vert, frag, binding, attrs, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
        graphics::destroyShaderModule(vert); graphics::destroyShaderModule(frag);

        vert = graphics::loadShaderModule("shaders/simple.vert.spv");
        frag = graphics::loadShaderModule("shaders/simple.frag.spv");
        g_simple_pipeline = graphics::createGraphicsPipeline(graphics::internal::context.render_pass, g_simple_pipeline_layout, vert, frag, binding, attrs, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
        graphics::destroyShaderModule(vert); graphics::destroyShaderModule(frag);

        vert = graphics::loadShaderModule("shaders/lines.vert.spv");
        frag = graphics::loadShaderModule("shaders/lines.frag.spv");
        g_line_pipeline = graphics::createGraphicsPipeline(graphics::internal::context.render_pass, g_line_pipeline_layout, vert, frag, binding, attrs, VK_PRIMITIVE_TOPOLOGY_LINE_LIST);
        graphics::destroyShaderModule(vert); graphics::destroyShaderModule(frag);

        return g_object_pipeline != VK_NULL_HANDLE;
    }

    void shutdown()
    {
        auto& context = graphics::internal::context;
        vkQueueWaitIdle(context.graphics_queue);

        if (g_object_pipeline) vkDestroyPipeline(context.device, g_object_pipeline, nullptr);
        if (g_simple_pipeline) vkDestroyPipeline(context.device, g_simple_pipeline, nullptr);
        if (g_line_pipeline) vkDestroyPipeline(context.device, g_line_pipeline, nullptr);
        if (g_object_pipeline_layout) vkDestroyPipelineLayout(context.device, g_object_pipeline_layout, nullptr);
        if (g_simple_pipeline_layout) vkDestroyPipelineLayout(context.device, g_simple_pipeline_layout, nullptr);
        if (g_line_pipeline_layout) vkDestroyPipelineLayout(context.device, g_line_pipeline_layout, nullptr);

        for (int i = 0; i < NUM_ALBEDO_TEXTURES; ++i) {
            vkDestroySampler(context.device, g_albedo_textures[i].sampler, nullptr);
            vkDestroyImageView(context.device, g_albedo_textures[i].view, nullptr);
            vmaDestroyImage(context.allocator, g_albedo_textures[i].image, g_albedo_textures[i].allocation);
        }
        vkDestroySampler(context.device, g_specular_texture.sampler, nullptr);
        vkDestroyImageView(context.device, g_specular_texture.view, nullptr);
        vmaDestroyImage(context.allocator, g_specular_texture.image, g_specular_texture.allocation);
        vkDestroySampler(context.device, g_emissive_texture.sampler, nullptr);
        vkDestroyImageView(context.device, g_emissive_texture.view, nullptr);
        vmaDestroyImage(context.allocator, g_emissive_texture.image, g_emissive_texture.allocation);

        graphics::destroyBuffer(g_scene_ubo); graphics::destroyBuffer(g_point_ssbo); graphics::destroyBuffer(g_spot_ssbo);
        for (int i = 0; i < 2; ++i) graphics::destroyBuffer(g_objects[i].ubo_buffer);
        for (int i = 0; i < MAX_LIGHTS * 2; ++i) graphics::destroyBuffer(g_light_boxes[i].ubo_buffer);
        for (int i = 0; i < MAX_LIGHTS; ++i) graphics::destroyBuffer(g_spot_cones[i].ubo_buffer);

        graphics::destroyBuffer(g_line_vertex_buffer);
        graphics::destroyBuffer(g_cube_index_buffer); graphics::destroyBuffer(g_cube_vertex_buffer);
        graphics::destroyBuffer(g_index_buffer); graphics::destroyBuffer(g_vertex_buffer);

        if (g_descriptor_pool) vkDestroyDescriptorPool(context.device, g_descriptor_pool, nullptr);
        if (g_texture_layout) vkDestroyDescriptorSetLayout(context.device, g_texture_layout, nullptr);
        if (g_model_layout) vkDestroyDescriptorSetLayout(context.device, g_model_layout, nullptr);
        if (g_scene_layout) vkDestroyDescriptorSetLayout(context.device, g_scene_layout, nullptr);
    }

    void update(double time)
    {
        static double last_time = 0.0;
        const float dt = (last_time == 0.0) ? 0.0f : static_cast<float>(time - last_time);
        last_time = time;

        for (int i = 0; i < 2; ++i) {
            if (g_objects[i].animate) {
                g_objects[i].anim_phase += dt * g_objects[i].anim_speed;
                g_objects[i].position.x = g_objects[i].anim_radius * std::cos(g_objects[i].anim_phase);
                g_objects[i].position.z = g_objects[i].anim_radius * std::sin(g_objects[i].anim_phase);
                g_objects[i].rotation.y = glm::degrees(g_objects[i].anim_phase) * 2.0f;
            }
        }

        if (ImGui::Begin("Lab 3 - Textures & Everything")) {
            ImGui::SeparatorText("Camera");
            ImGui::SliderFloat("Yaw", &g_cam_yaw, -180.0f, 180.0f);
            ImGui::SliderFloat("Pitch", &g_cam_pitch, -89.0f, 89.0f);
            ImGui::SliderFloat("Radius", &g_cam_radius, 1.0f, 20.0f);
            ImGui::SliderFloat("FOV", &g_fov, 30.0f, 90.0f);

            const char* albedo_names[] = { "Smoking Smiley", "Vas (Alien)", "Fat Unicorn" };

            for (int i = 0; i < 2; ++i) {
                auto& o = g_objects[i];
                ImGui::SeparatorText(("Object " + std::to_string(i)).c_str());

                ImGui::Checkbox(("Visible##vis" + std::to_string(i)).c_str(), &o.visible);
                ImGui::SliderFloat3(("Position##pos" + std::to_string(i)).c_str(), &o.position.x, -5.0f, 5.0f);
                ImGui::SliderFloat3(("Rotation##rot" + std::to_string(i)).c_str(), &o.rotation.x, -180.0f, 180.0f);
                ImGui::SliderFloat3(("Scale##scl" + std::to_string(i)).c_str(), &o.scale.x, 0.1f, 3.0f);

                ImGui::Checkbox(("Animate##a" + std::to_string(i)).c_str(), &o.animate);
                ImGui::SliderFloat(("Speed##spd" + std::to_string(i)).c_str(), &o.anim_speed, 0.0f, 5.0f);
                ImGui::SliderFloat(("Radius##rad" + std::to_string(i)).c_str(), &o.anim_radius, 0.0f, 5.0f);

                // Выбор текстуры для каждого объекта
                if (ImGui::Combo(("Albedo##" + std::to_string(i)).c_str(), &o.current_albedo_index, albedo_names, IM_ARRAYSIZE(albedo_names))) {
                    VkDescriptorImageInfo imageInfo{};
                    imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                    imageInfo.imageView = g_albedo_textures[o.current_albedo_index].view;
                    imageInfo.sampler = g_albedo_textures[o.current_albedo_index].sampler;

                    VkWriteDescriptorSet texWrite{};
                    texWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    texWrite.dstSet = o.texture_set;
                    texWrite.dstBinding = 0;
                    texWrite.descriptorCount = 1;
                    texWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                    texWrite.pImageInfo = &imageInfo;
                    vkUpdateDescriptorSets(graphics::internal::context.device, 1, &texWrite, 0, nullptr);
                }

                ImGui::Checkbox(("Emissive##" + std::to_string(i)).c_str(), &o.enable_emissive);
                ImGui::Checkbox(("Specular##" + std::to_string(i)).c_str(), &o.enable_specular);
            }

            ImGui::Separator();
            ImGui::SeparatorText("Lighting");
            if (ImGui::Button("Add Point Light") && g_point_lights.size() < MAX_LIGHTS) {
                PointLight l{};
                l.position = glm::vec4((rand() % 100) / 50.0f - 1.0f, 1.0f, (rand() % 100) / 50.0f - 1.0f, 1.0f);
                l.color = glm::vec4(1.0f, 0.5f, 0.2f, 1.0f);
                l.params = glm::vec4(1.0f, 0.14f, 0.07f, 1.0f);
                g_point_lights.push_back(l);
            }

            // Слайдеры для точечных источников
            for (size_t i = 1; i < g_point_lights.size(); ++i) {
                ImGui::PushID((int)i);
                ImGui::Text("Point %d", (int)i);
                ImGui::SliderFloat3("Pos", &g_point_lights[i].position.x, -5.0f, 5.0f);
                ImGui::ColorEdit3("Col", &g_point_lights[i].color.x);
                ImGui::PopID();
            }

            if (ImGui::Button("Add Spot Light") && g_spot_lights.size() < MAX_LIGHTS) {
                SpotLight l{};
                l.position = glm::vec4(2.0f, 2.0f, 2.0f, 1.0f);
                l.direction = glm::vec4(-1.0f, -1.0f, -1.0f, 0.0f);
                l.color = glm::vec4(0.2f, 0.5f, 1.0f, 1.0f);
                l.params = glm::vec4(1.0f, 0.14f, 0.07f, cos(glm::radians(12.5f)));
                l.params2 = glm::vec4(cos(glm::radians(25.0f)), 0.0f, 0.0f, 0.0f);
                g_spot_lights.push_back(l);
            }

            // Слайдеры для прожекторов
            for (size_t i = 1; i < g_spot_lights.size(); ++i) {
                ImGui::PushID((int)(i + 100));
                ImGui::Text("Spot %d", (int)i);
                ImGui::SliderFloat3("Pos", &g_spot_lights[i].position.x, -5.0f, 5.0f);
                ImGui::SliderFloat3("Dir", &g_spot_lights[i].direction.x, -1.0f, 1.0f);
                ImGui::ColorEdit3("Col", &g_spot_lights[i].color.x);
                ImGui::PopID();
            }
            ImGui::Checkbox("Show Light Sources", &g_show_lights);
        }
        ImGui::End();
    }

    void render(const graphics::internal::FrameData& fd)
    {
        auto& context = graphics::internal::context;
        updateSceneUniforms();

        for (int i = 0; i < 2; ++i) updateModelUniform(g_objects[i]);

        int boxIdx = 0;
        for (size_t i = 1; i < g_point_lights.size() && boxIdx < MAX_LIGHTS * 2; ++i) {
            g_light_boxes[boxIdx].position = glm::vec3(g_point_lights[i].position);
            updateModelUniform(g_light_boxes[boxIdx], g_point_lights[i].color);
            boxIdx++;
        }
        int coneIdx = 0;
        for (size_t i = 1; i < g_spot_lights.size() && boxIdx < MAX_LIGHTS * 2; ++i) {
            g_light_boxes[boxIdx].position = glm::vec3(g_spot_lights[i].position);
            updateModelUniform(g_light_boxes[boxIdx], g_spot_lights[i].color);
            boxIdx++;

            g_spot_cones[coneIdx].position = glm::vec3(g_spot_lights[i].position);
            glm::vec3 dir = glm::normalize(glm::vec3(g_spot_lights[i].direction));
            glm::mat4 rotMatrix = glm::mat4(1.0f);
            if (glm::length(dir) > 0.0f) {
                glm::vec3 zAxis = -dir;
                glm::vec3 up = (std::abs(zAxis.y) > 0.99f) ? glm::vec3(0.0f, 0.0f, 1.0f) : glm::vec3(0.0f, 1.0f, 0.0f);
                glm::vec3 xAxis = glm::normalize(glm::cross(up, zAxis));
                glm::vec3 yAxis = glm::cross(zAxis, xAxis);
                rotMatrix[0] = glm::vec4(xAxis, 0.0f); rotMatrix[1] = glm::vec4(yAxis, 0.0f);
                rotMatrix[2] = glm::vec4(zAxis, 0.0f); rotMatrix[3] = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
            }
            g_spot_cones[coneIdx].ubo.model = glm::translate(glm::mat4(1.0f), g_spot_cones[coneIdx].position) * rotMatrix;
            g_spot_cones[coneIdx].ubo.albedo = glm::vec4(1.0f);
            g_spot_cones[coneIdx].ubo.specColor = glm::vec4(0.0f);
            g_spot_cones[coneIdx].ubo.specParams = glm::vec4(1.0f);
            std::memcpy(g_spot_cones[coneIdx].ubo_buffer.mapped, &g_spot_cones[coneIdx].ubo, sizeof(ModelUBO));
            coneIdx++;
        }

        vkResetCommandBuffer(fd.command_buffer, 0);
        VkCommandBufferBeginInfo begin{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(fd.command_buffer, &begin);

        VkClearValue clear_values[] = { {.color = {{0.1f, 0.1f, 0.12f, 1.0f}} }, {.depthStencil = {1.0f, 0} } };
        VkRenderPassBeginInfo rp{ VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
        rp.renderPass = context.render_pass; rp.framebuffer = fd.framebuffer;
        rp.renderArea = { {0,0}, context.swapchain_extent };
        rp.clearValueCount = 2; rp.pClearValues = clear_values;
        vkCmdBeginRenderPass(fd.command_buffer, &rp, VK_SUBPASS_CONTENTS_INLINE);

        VkViewport viewport{ 0, 0, (float)context.swapchain_extent.width, (float)context.swapchain_extent.height, 0, 1 };
        VkRect2D scissor{ {0, 0}, context.swapchain_extent };
        vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);
        vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);

        VkDeviceSize offset = 0;

        // 1. Додекаэдры
        vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, g_object_pipeline);
        vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &g_vertex_buffer.buffer, &offset);
        vkCmdBindIndexBuffer(fd.command_buffer, g_index_buffer.buffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, g_object_pipeline_layout, 0, 1, &g_scene_set, 0, nullptr);

        float time = (float)glfwGetTime();
        for (int i = 0; i < 2; ++i) {
            if (!g_objects[i].visible) continue;

            vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, g_object_pipeline_layout, 1, 1, &g_objects[i].model_set, 0, nullptr);
            vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, g_object_pipeline_layout, 2, 1, &g_objects[i].texture_set, 0, nullptr);

            float pushData[4] = { time, 1.0f, g_objects[i].enable_emissive ? 1.0f : 0.0f, g_objects[i].enable_specular ? 1.0f : 0.0f };
            vkCmdPushConstants(fd.command_buffer, g_object_pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pushData), pushData);

            vkCmdDrawIndexed(fd.command_buffer, g_index_count, 1, 0, 0, 0);
        }

        // 2. Кубики и Конусы
        if (g_show_lights) {
            vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, g_simple_pipeline);
            vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, g_simple_pipeline_layout, 0, 1, &g_scene_set, 0, nullptr);
            vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &g_cube_vertex_buffer.buffer, &offset);
            vkCmdBindIndexBuffer(fd.command_buffer, g_cube_index_buffer.buffer, 0, VK_INDEX_TYPE_UINT32);

            int boxIdx2 = 0;
            for (size_t i = 1; i < g_point_lights.size(); ++i) {
                vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, g_simple_pipeline_layout, 1, 1, &g_light_boxes[boxIdx2].model_set, 0, nullptr);
                vkCmdDrawIndexed(fd.command_buffer, g_cube_index_count, 1, 0, 0, 0);
                boxIdx2++;
            }
            for (size_t i = 1; i < g_spot_lights.size(); ++i) {
                vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, g_simple_pipeline_layout, 1, 1, &g_light_boxes[boxIdx2].model_set, 0, nullptr);
                vkCmdDrawIndexed(fd.command_buffer, g_cube_index_count, 1, 0, 0, 0);
                boxIdx2++;
            }

            vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, g_line_pipeline);
            vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, g_line_pipeline_layout, 0, 1, &g_scene_set, 0, nullptr);
            vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &g_line_vertex_buffer.buffer, &offset);

            int coneIdx2 = 0;
            for (size_t i = 1; i < g_spot_lights.size(); ++i) {
                vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, g_line_pipeline_layout, 1, 1, &g_spot_cones[coneIdx2].model_set, 0, nullptr);
                vkCmdDraw(fd.command_buffer, g_line_vertex_count, 1, 0, 0);
                coneIdx2++;
            }
        }

        vkCmdEndRenderPass(fd.command_buffer);
        vkEndCommandBuffer(fd.command_buffer);
    }

} // namespace application