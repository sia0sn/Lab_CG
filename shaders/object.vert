#version 450

layout(location = 0) in vec3 inPosition;

layout(set = 0, binding = 0, std140) uniform UBO {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec4 color;
} ubo;

layout(location = 0) out vec3 vLocalPos;

void main()
{
    vLocalPos   = inPosition;
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 1.0);
}