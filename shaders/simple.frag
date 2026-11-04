#version 450
layout(location = 0) in vec3 vColor;
layout(location = 0) out vec4 outColor;

layout(set = 1, binding = 0, std140) uniform ModelUBO {
    mat4 model;
    vec4 albedo; // Сюда мы передаем цвет лампы
    vec4 specColor;
    vec4 specParams;
} model;

void main() {
    // Красим кубик в цвет источника света (без учета освещения)
    outColor = vec4(model.albedo.rgb, 1.0); 
}