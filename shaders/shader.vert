#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;

// Shared camera transform (bound once per frame).
layout(set = 0, binding = 0) uniform CameraUBO {
    mat4 view;
    mat4 proj;
} camera;

// Per-object transform and color multiplier.
layout(set = 1, binding = 0) uniform ObjectUBO {
    mat4 model;
    vec4 color;
} object;

layout(location = 0) out vec3 fragColor;

// Fixed directional light for flat shading.
const vec3 LIGHT_DIR = normalize(vec3(0.4, 0.8, 0.6));

void main() {
    vec4 worldPos = object.model * vec4(inPosition, 1.0);
    gl_Position = camera.proj * camera.view * worldPos;

    // Transform the normal to world space (uniform scaling is assumed).
    vec3 normal = normalize(mat3(object.model) * inNormal);
    float diff = max(dot(normal, LIGHT_DIR), 0.0);
    float ambient = 0.30;
    float light = ambient + (1.0 - ambient) * diff;

    // Chosen color multiplies the procedural per-vertex color.
    fragColor = object.color.rgb * inColor * light;
}
