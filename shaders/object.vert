#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV; // <-- ДОБАВИЛИ ВХОДЯЩИЕ UV

layout(set = 0, binding = 0, std140) uniform SceneUBO {
    mat4 view;
    mat4 proj;
    vec4 cameraPos;
    vec4 ambientColor;
    vec4 dirLightDir;
    vec4 dirLightColor;
} scene;

layout(set = 1, binding = 0, std140) uniform ModelUBO {
    mat4 model;
    vec4 albedo;
    vec4 specColor;
    vec4 specParams;
} model;

layout(location = 0) out vec3 vWorldPos;
layout(location = 1) out vec3 vNormal;
layout(location = 2) out vec2 vUV; // <-- ДОБАВИЛИ ВЫХОДЯЩИЕ UV

void main()
{
    vec4 worldPos = model.model * vec4(inPosition, 1.0);
    vWorldPos = worldPos.xyz;
    vNormal = mat3(transpose(inverse(model.model))) * inNormal;
    vUV = vec2(inPosition.x * 0.5 + 0.5, 1.0 - (inPosition.y * 0.5 + 0.5)); // <-- ПЕРЕДАЕМ UV ДАЛЬШЕ

    gl_Position = scene.proj * scene.view * worldPos;
}