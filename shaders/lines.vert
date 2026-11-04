#version 450

layout(location = 0) in vec3 inPosition;

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

void main() {
    // Умножаем позицию на матрицу модели (чтобы конус вращался и перемещался вместе с источником света)
    // Затем на view (камера) и proj (перспектива)
    gl_Position = scene.proj * scene.view * model.model * vec4(inPosition, 1.0);
}