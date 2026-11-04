#version 450

layout(location = 0) in vec3 vWorldPos;
layout(location = 1) in vec3 vNormal;

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

// Структуры для SSBO (должны совпадать с C++)
struct PointLight {
    vec4 position;
    vec4 color;
    vec4 params; // x: constant, y: linear, z: quadratic
};

struct SpotLight {
    vec4 position;
    vec4 direction;
    vec4 color;
    vec4 params; // x: constant, y: linear, z: quadratic, w: cos(inner)
    vec4 params2; // x: cos(outer)
};

layout(set = 0, binding = 1, std430) readonly buffer PointLights { PointLight lights[]; } pointLights;
layout(set = 0, binding = 2, std430) readonly buffer SpotLights  { SpotLight lights[]; } spotLights;

layout(location = 0) out vec4 outColor;

vec3 blinnPhong(vec3 N, vec3 V, vec3 L, vec3 lightColor, float attenuation) {
    float diff = max(dot(N, L), 0.0);
    vec3 diffuse = diff * lightColor * model.albedo.rgb;

    vec3 H = normalize(V + L);
    float spec = pow(max(dot(N, H), 0.0), model.specParams.x);
    vec3 specular = spec * lightColor * model.specColor.rgb;

    return (diffuse + specular) * attenuation;
}

void main()
{
    vec3 N = normalize(vNormal);
    vec3 V = normalize(scene.cameraPos.xyz - vWorldPos);
    
    // Ambient
    vec3 finalColor = scene.ambientColor.rgb * model.albedo.rgb;

    // Directional Light
    vec3 L_dir = normalize(-scene.dirLightDir.xyz);
    float diffDir = max(dot(N, L_dir), 0.0);
    vec3 H_dir = normalize(V + L_dir);
    float specDir = pow(max(dot(N, H_dir), 0.0), model.specParams.x);
    finalColor += (diffDir * model.albedo.rgb + specDir * model.specColor.rgb) * scene.dirLightColor.rgb;

    // Point Lights
    int numPoint = int(pointLights.lights[0].position.w); // Передаем количество в 0-м элементе
    for (int i = 1; i <= numPoint; ++i) {
        vec3 L_pt = pointLights.lights[i].position.xyz - vWorldPos;
        float dist = length(L_pt);
        L_pt = normalize(L_pt);
        float att = 1.0 / (pointLights.lights[i].params.x + pointLights.lights[i].params.y * dist + pointLights.lights[i].params.z * dist * dist);
        finalColor += blinnPhong(N, V, L_pt, pointLights.lights[i].color.rgb, att);
    }

    // Spot Lights
    int numSpot = int(spotLights.lights[0].position.w);
    for (int i = 1; i <= numSpot; ++i) {
        vec3 L_spot = spotLights.lights[i].position.xyz - vWorldPos;
        float dist = length(L_spot);
        L_spot = normalize(L_spot);
        
        float theta = dot(L_spot, normalize(-spotLights.lights[i].direction.xyz));
        float epsilon = spotLights.lights[i].params.w - spotLights.lights[i].params2.x;
        float intensity = clamp((theta - spotLights.lights[i].params2.x) / epsilon, 0.0, 1.0);
        
        if (intensity > 0.0) {
            float att = 1.0 / (spotLights.lights[i].params.x + spotLights.lights[i].params.y * dist + spotLights.lights[i].params.z * dist * dist);
            finalColor += blinnPhong(N, V, L_spot, spotLights.lights[i].color.rgb, att * intensity);
        }
    }

    outColor = vec4(finalColor, 1.0);
}