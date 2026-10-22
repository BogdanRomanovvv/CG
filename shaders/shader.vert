#version 450

// Атрибуты вершины совпадают с полями Vertex, описанными в C++.
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;

// Набор сцены (set 0): матрицы камеры и общие параметры освещения.
// Последний вектор содержит количество точечных источников и прожекторов.
layout(set = 0, binding = 0) uniform SceneUBO {
    mat4 view;
    mat4 proj;
    vec4 cameraPosition;
    vec4 ambient;
    vec4 directionalDirection;
    vec4 directionalColor;
    uvec4 lightCounts;
} scene;

// Набор модели (set 1): трансформация и свойства материала текущего объекта.
layout(set = 1, binding = 0) uniform ModelUBO {
    mat4 model;
    vec4 albedo;
    vec4 specularAndShininess;
    vec4 emission;
} material;

// Эти значения интерполируются по поверхности и используются фрагментным шейдером.
layout(location = 0) out vec3 worldPosition;
layout(location = 1) out vec3 worldNormal;
layout(location = 2) out vec3 vertexColor;

void main() {
    // Переводим позицию из локальных координат модели в мировые.
    vec4 world = material.model * vec4(inPosition, 1.0);
    worldPosition = world.xyz;
    // Обратная транспонированная матрица сохраняет нормаль при неравномерном масштабе.
    worldNormal = normalize(transpose(inverse(mat3(material.model))) * inNormal);
    // Цвет вершин задаёт процедурный оттенок, который учитывается вместе с альбедо.
    vertexColor = inColor;
    // Переводим вершину из локального пространства модели в пространство клиппинга.
    gl_Position = scene.proj * scene.view * world;
}
