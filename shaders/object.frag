#version 450

layout(location = 0) in vec3 vLocalPos;

layout(set = 0, binding = 0, std140) uniform UBO {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec4 color;
} ubo;

layout(location = 0) out vec4 outColor;

void main()
{
    // Процедурный цвет вершины по её локальной позиции (задание 5)
    vec3 vertex_color = normalize(vLocalPos) * 0.5 + 0.5;

    // Умножаем на цвет из UI (задание 4)
    outColor = vec4(vertex_color * ubo.color.rgb, 1.0);
}