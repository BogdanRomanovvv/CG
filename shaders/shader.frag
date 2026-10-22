#version 450

struct PointLight {
    vec4 position;
    vec4 colorIntensity;
};
struct SpotLight {
    vec4 position;
    vec4 direction;
    vec4 colorIntensity;
    vec4 coneAngles;
};

layout(set = 0, binding = 0) uniform SceneUBO {
    mat4 view;
    mat4 proj;
    vec4 cameraPosition;
    vec4 ambient;
    vec4 directionalDirection;
    vec4 directionalColor;
    uvec4 lightCounts;
} scene;
// Runtime-массивы не задают максимальное число источников на стороне шейдера.
layout(set = 0, binding = 1, std430) readonly buffer PointLightBuffer { PointLight pointLights[]; } points;
layout(set = 0, binding = 2, std430) readonly buffer SpotLightBuffer { SpotLight spotLights[]; } spots;
layout(set = 1, binding = 0) uniform ModelUBO {
    mat4 model;
    vec4 albedo;
    vec4 specularAndShininess;
    vec4 emission;
} material;

layout(location = 0) in vec3 worldPosition;
layout(location = 1) in vec3 worldNormal;
layout(location = 2) in vec3 vertexColor;
layout(location = 0) out vec4 outColor;

vec3 evaluateBlinnPhong(vec3 N, vec3 V, vec3 L, vec3 radiance) {
    // Диффузная часть использует N·L, а блик — half-вектор между взглядом и светом.
    float nDotL = max(dot(N, L), 0.0);
    vec3 H = normalize(L + V);
    float specular = nDotL > 0.0
        ? pow(max(dot(N, H), 0.0), max(material.specularAndShininess.w, 1.0))
        : 0.0;
    return (material.albedo.rgb * vertexColor * nDotL + material.specularAndShininess.rgb * specular) * radiance;
}

void main() {
    if (material.emission.a > 0.5) {
        // Для маркеров и линий сразу выводим постоянный emission-цвет.
        outColor = vec4(material.emission.rgb, 1.0);
        return;
    }
    vec3 N = normalize(worldNormal);
    vec3 V = normalize(scene.cameraPosition.xyz - worldPosition);
    vec3 baseColor = material.albedo.rgb * vertexColor;
    vec3 result = baseColor * scene.ambient.rgb * scene.ambient.a;

    vec3 dirL = normalize(-scene.directionalDirection.xyz);
    result += evaluateBlinnPhong(N, V, dirL, scene.directionalColor.rgb * scene.directionalColor.a);

    for (uint i = 0; i < scene.lightCounts.x; ++i) {
        PointLight light = points.pointLights[i];
        vec3 delta = light.position.xyz - worldPosition;
        float distanceSquared = max(dot(delta, delta), 0.0001);
        // Точечный свет ослабевает обратно пропорционально квадрату расстояния.
        result += evaluateBlinnPhong(N, V, normalize(delta),
            light.colorIntensity.rgb * light.colorIntensity.a / distanceSquared);
    }

    for (uint i = 0; i < scene.lightCounts.y; ++i) {
        SpotLight light = spots.spotLights[i];
        vec3 delta = light.position.xyz - worldPosition;
        float distanceSquared = max(dot(delta, delta), 0.0001);
        vec3 fromLight = -normalize(delta);
        float theta = dot(fromLight, normalize(light.direction.xyz));
        // Плавно переходим от полного света внутри внутреннего угла к нулю снаружи.
        float cone = smoothstep(light.coneAngles.y, light.coneAngles.x, theta);
        result += evaluateBlinnPhong(N, V, normalize(delta),
            light.colorIntensity.rgb * light.colorIntensity.a * cone / distanceSquared);
    }

    outColor = vec4(result, 1.0);
}
