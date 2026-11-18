#version 450

layout(location = 0) in vec3 vWorldPos;
layout(location = 1) in vec3 vNormal;
layout(location = 2) in vec2 vUV;

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

// Текстуры
layout(set = 2, binding = 0) uniform sampler2D albedoTex;
layout(set = 2, binding = 1) uniform sampler2D specularTex;
layout(set = 2, binding = 2) uniform sampler2D emissiveTex;

layout(push_constant) uniform PushConstants {
    float time;
    float enableEffect;
    float enableEmissive;
    float enableSpecular;
} pc;

struct PointLight {
    vec4 position;
    vec4 color;
    vec4 params;
};

struct SpotLight {
    vec4 position;
    vec4 direction;
    vec4 color;
    vec4 params;
    vec4 params2;
};

layout(set = 0, binding = 1, std430) readonly buffer PointLights { PointLight lights[]; } pointLights;
layout(set = 0, binding = 2, std430) readonly buffer SpotLights  { SpotLight lights[]; } spotLights;

layout(location = 0) out vec4 outColor;

vec3 blinnPhong(vec3 N, vec3 V, vec3 L, vec3 lightColor, float attenuation, vec3 baseColor, vec3 specMask) {
    float diff = max(dot(N, L), 0.0);
    vec3 diffuse = diff * lightColor * baseColor;

    vec3 H = normalize(V + L);
    float spec = pow(max(dot(N, H), 0.0), model.specParams.x);
    
    // Если флаг включен (1.0), умножаем на маску. Если выключен (0.0) - блик становится нулевым.
    vec3 specular = spec * lightColor * model.specColor.rgb * specMask * pc.enableSpecular;

    return (diffuse + specular) * attenuation;
}

void main()
{
    vec3 N = normalize(vNormal);
    vec3 V = normalize(scene.cameraPos.xyz - vWorldPos);
    
    // 1. Получаем цвета из всех текстур
    vec2 distortedUV = vUV;
    if (pc.enableEffect > 0.5) {
        float wobbleSpeed = 5.0;
        float wobbleAmount = 0.05;
        distortedUV.x += sin(vUV.y * 10.0 + pc.time * wobbleSpeed) * wobbleAmount;
        distortedUV.y += cos(vUV.x * 10.0 + pc.time * wobbleSpeed) * wobbleAmount;
    }
    
    vec3 texColor = texture(albedoTex, distortedUV).rgb;
    vec3 specMask = texture(specularTex, vUV).rgb;
    vec3 emissiveColor = texture(emissiveTex, vUV).rgb;

    // 2. Ambient
    vec3 finalColor = scene.ambientColor.rgb * texColor;

    // 3. Directional Light
    vec3 L_dir = normalize(-scene.dirLightDir.xyz);
    float diffDir = max(dot(N, L_dir), 0.0);
    vec3 H_dir = normalize(V + L_dir);
    float specDir = pow(max(dot(N, H_dir), 0.0), model.specParams.x);
    finalColor += (diffDir * texColor + specDir * model.specColor.rgb * specMask * pc.enableSpecular) * scene.dirLightColor.rgb;

    // 4. Point Lights
    int numPoint = int(pointLights.lights[0].position.w);
    for (int i = 1; i <= numPoint; ++i) {
        vec3 L_pt = pointLights.lights[i].position.xyz - vWorldPos;
        float dist = length(L_pt);
        L_pt = normalize(L_pt);
        float att = 1.0 / (pointLights.lights[i].params.x + pointLights.lights[i].params.y * dist + pointLights.lights[i].params.z * dist * dist);
        finalColor += blinnPhong(N, V, L_pt, pointLights.lights[i].color.rgb, att, texColor, specMask);
    }

    // 5. Spot Lights
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
            finalColor += blinnPhong(N, V, L_spot, spotLights.lights[i].color.rgb, att * intensity, texColor, specMask);
        }
    }

    // 6. Emissive (только если включено)
    if (pc.enableEmissive > 0.5) {
        finalColor += emissiveColor * 0.5; // Можно убрать 0.2 и сделать 0.5
    }

    outColor = vec4(finalColor, 1.0);
}