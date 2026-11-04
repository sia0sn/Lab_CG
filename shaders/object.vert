#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;

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
    vec4 specParams; // x: shininess
} model;

layout(location = 0) out vec3 vWorldPos;
layout(location = 1) out vec3 vNormal;

void main()
{
    vec4 worldPos = model.model * vec4(inPosition, 1.0);
    vWorldPos = worldPos.xyz;
    vNormal = mat3(transpose(inverse(model.model))) * inNormal;

    gl_Position = scene.proj * scene.view * worldPos;
}