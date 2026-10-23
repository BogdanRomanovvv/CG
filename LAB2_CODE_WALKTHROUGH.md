# Лабораторная №2: построчный разбор кода

Ниже разобраны конкретные строки текущей версии проекта. Сначала идут шейдеры, затем места в `source/application.cpp`, которые подготавливают для них данные и запускают отрисовку.

## `shaders/shader.vert`

```glsl
#version 450
```

Версия GLSL, под которую компилируется шейдер.

```glsl
layout(location = 0) in vec3 inPosition;
```

В текущую вершину приходит её локальная позиция — координаты относительно центра самой модели.

```glsl
layout(location = 1) in vec3 inNormal;
```

Приходит нормаль этой вершины. В нашем меше одной плоской грани вершины имеют одинаковую нормаль.

```glsl
layout(location = 2) in vec3 inColor;
```

Приходит процедурный RGB-цвет вершины из геометрии первой лабораторной.

```glsl
layout(set = 0, binding = 0) uniform SceneUBO {
    mat4 view;
    mat4 proj;
    vec4 cameraPosition;
    vec4 ambient;
    vec4 directionalDirection;
    vec4 directionalColor;
    uvec4 lightCounts;
} scene;
```

- `view` — матрица вида камеры.
- `proj` — матрица перспективной или ортографической проекции.
- `cameraPosition.xyz` — положение камеры, нужное фрагментному шейдеру для блика.
- `ambient.rgb` и `ambient.a` — цвет и интенсивность фонового света.
- `directionalDirection.xyz` — направление лучей направленного источника.
- `directionalColor.rgb` и `.a` — цвет и интенсивность этого источника.
- `lightCounts.x` — сколько точечных источников читать из первого SSBO.
- `lightCounts.y` — сколько прожекторов читать из второго SSBO.
- Имя `scene` — имя экземпляра этой структуры, через него ниже читаются поля.

```glsl
layout(set = 1, binding = 0) uniform ModelUBO {
    mat4 model;
    vec4 albedo;
    vec4 specularAndShininess;
    vec4 emission;
} material;
```

- `model` переводит координаты модели в мировые.
- `albedo.rgb` — основной цвет поверхности.
- `specularAndShininess.rgb` — цвет блика.
- `specularAndShininess.w` — степень блеска.
- `emission` задаёт цвет и флаг для маркеров света; обычные модели оставляют флаг выключенным.
- `material` — имя, через которое шейдер обращается к полям текущей модели.

```glsl
layout(location = 0) out vec3 worldPosition;
layout(location = 1) out vec3 worldNormal;
layout(location = 2) out vec3 vertexColor;
```

Эти три значения вершинный шейдер передаст следующему этапу. GPU интерполирует их по поверхности каждого треугольника, а фрагментный шейдер получит значение для каждого фрагмента.

```glsl
vec4 world = material.model * vec4(inPosition, 1.0);
```

Добавляется координата `w = 1`, затем позиция умножается на матрицу модели. В результате `world.xyz` — та же точка в координатах всей сцены, уже с учётом положения, поворота и масштаба объекта.

```glsl
worldPosition = world.xyz;
```

Сохраняем мировую позицию для расчёта расстояния до точечных источников и прожекторов во фрагментном шейдере.

```glsl
worldNormal = normalize(transpose(inverse(mat3(material.model))) * inNormal);
```

Из матрицы модели берётся часть 3×3, отвечающая за поворот и масштаб. Для нормали используется обратная транспонированная матрица: при неравномерном масштабе обычная матрица модели может исказить угол нормали к поверхности. `normalize` возвращает нормали длину 1, которая нужна формулам освещения.

```glsl
vertexColor = inColor;
```

Передаём процедурный цвет без изменений; он участвует в diffuse-составляющей во фрагментном шейдере.

```glsl
gl_Position = scene.proj * scene.view * world;
```

Последовательно применяются проекция, камера и уже рассчитанная мировая позиция. Порядок умножения здесь важен: сначала точка переводится из координат модели в мир, затем камера преобразует её в своё пространство, после чего проекция готовит её к растеризации.

## `shaders/shader.frag`

```glsl
struct PointLight {
    vec4 position;
    vec4 colorIntensity;
};
```

Описание одного точечного источника. `position.xyz` — его положение, `colorIntensity.rgb` — цвет, `.a` — интенсивность.

```glsl
struct SpotLight {
    vec4 position;
    vec4 direction;
    vec4 colorIntensity;
    vec4 coneAngles;
};
```

У прожектора помимо позиции и цвета есть направление. В `coneAngles.x` и `.y` CPU записывает косинусы внутреннего и внешнего углов.

```glsl
layout(set = 0, binding = 1, std430) readonly buffer PointLightBuffer {
    PointLight pointLights[];
} points;
```

Это массив точечных источников в storage buffer. `readonly` означает, что шейдер только читает этот массив. `std430` задаёт правила размещения структур в буфере. `pointLights[]` не имеет фиксированного размера в GLSL; предел цикла задаёт `scene.lightCounts.x`.

```glsl
layout(set = 0, binding = 2, std430) readonly buffer SpotLightBuffer {
    SpotLight spotLights[];
} spots;
```

Второй storage buffer с прожекторами. Данные читаются через имя экземпляра `spots`.

Структуры `SceneUBO` и `ModelUBO` во фрагментном шейдере содержат те же поля, что и в вершинном. Здесь они нужны для чтения камеры, источников света и материала во время расчёта цвета. Далее объявлены входы `worldPosition`, `worldNormal`, `vertexColor` и выход `outColor`: первые три приходят с предыдущего этапа, последний становится цветом фрагмента.

```glsl
vec3 evaluateBlinnPhong(vec3 N, vec3 V, vec3 L, vec3 radiance) {
```

Функция считает вклад одного направления света. `N` — нормаль поверхности, `V` — направление от поверхности к камере, `L` — направление от поверхности к источнику, `radiance` — цвет источника с его интенсивностью и затуханием.

```glsl
float nDotL = max(dot(N, L), 0.0);
```

`dot` вычисляет скалярное произведение нормали и направления на свет. Оно показывает, насколько поверхность повернута к источнику. `max(..., 0)` убирает отрицательное значение, если поверхность отвернута от света.

```glsl
vec3 H = normalize(L + V);
```

Складываем направления на свет и на камеру, нормализуем сумму и получаем half-вектор `H`. Именно к этому направлению должна быть обращена поверхность, чтобы появился сильный зеркальный блик.

```glsl
float specular = nDotL > 0.0
    ? pow(max(dot(N, H), 0.0), max(material.specularAndShininess.w, 1.0))
    : 0.0;
```

Если поверхность обращена к свету, вычисляется блик: скалярное произведение `N` и `H` обрезается снизу нулём, затем возводится в `shininess`. Большая степень даёт узкий блик, маленькая — широкий. Если поверхность отвернулась от источника, блик принудительно равен нулю.

```glsl
return (material.albedo.rgb * vertexColor * nDotL
      + material.specularAndShininess.rgb * specular) * radiance;
```

Складываются diffuse и specular цвета. Diffuse зависит от альбедо, цвета вершины и `N·L`; блик использует отдельный цвет материала. Затем вся сумма умножается на яркость текущего источника.

```glsl
if (material.emission.a > 0.5) {
    outColor = vec4(material.emission.rgb, 1.0);
    return;
}
```

Это специальная ветка для визуальных фигур. Если emission-флаг включён, шейдер выводит заданный цвет и сразу завершает работу. Поэтому маркеры и белые линии прожектора не проходят через расчёт Блинна–Фонга.

```glsl
vec3 N = normalize(worldNormal);
vec3 V = normalize(scene.cameraPosition.xyz - worldPosition);
```

Нормаль ещё раз нормализуется после интерполяции. `V` получается вычитанием позиции фрагмента из позиции камеры: это вектор от фрагмента к наблюдателю.

```glsl
vec3 baseColor = material.albedo.rgb * vertexColor;
vec3 result = baseColor * scene.ambient.rgb * scene.ambient.a;
```

Сначала объединяются основной цвет материала и цвет вершины. Затем к ним применяется ambient-цвет и его интенсивность. Это начальный цвет фрагмента до добавления направленного, точечного и прожекторного света.

```glsl
vec3 dirL = normalize(-scene.directionalDirection.xyz);
result += evaluateBlinnPhong(N, V, dirL,
    scene.directionalColor.rgb * scene.directionalColor.a);
```

Знак направления меняется, чтобы получить вектор от поверхности к источнику. Функция Блинна–Фонга добавляет направленный свет. У этого типа источника нет позиции, поэтому здесь нет вычисления расстояния и затухания.

```glsl
for (uint i = 0; i < scene.lightCounts.x; ++i) {
    PointLight light = points.pointLights[i];
```

Цикл проходит только по существующим точечным источникам. Текущие параметры копируются в переменную `light`.

```glsl
vec3 delta = light.position.xyz - worldPosition;
float distanceSquared = max(dot(delta, delta), 0.0001);
```

`delta` направлен от поверхности к источнику. Его длина в квадрате равна `dot(delta, delta)`. Корень извлекать не нужно, потому что формула затухания использует именно квадрат расстояния. Нижняя граница защищает от деления на ноль.

```glsl
result += evaluateBlinnPhong(N, V, normalize(delta),
    light.colorIntensity.rgb * light.colorIntensity.a / distanceSquared);
```

Нормализованный `delta` задаёт направление света. В `radiance` передаются RGB-цвет, интенсивность и множитель `1 / distanceSquared`. Поэтому при удвоении расстояния вклад источника уменьшается примерно в четыре раза.

```glsl
for (uint i = 0; i < scene.lightCounts.y; ++i) {
    SpotLight light = spots.spotLights[i];
```

Второй цикл аналогично проходит по всем существующим прожекторам.

```glsl
vec3 delta = light.position.xyz - worldPosition;
float distanceSquared = max(dot(delta, delta), 0.0001);
vec3 fromLight = -normalize(delta);
float theta = dot(fromLight, normalize(light.direction.xyz));
```

Первые две строки вычисляют направление и квадрат расстояния, как для точечного источника. `normalize(delta)` направлен от поверхности к свету, поэтому знак меняется: `fromLight` показывает направление от источника к поверхности. `theta` сравнивает его с направлением прожектора. Для нормализованных векторов это косинус угла между ними.

```glsl
float cone = smoothstep(light.coneAngles.y, light.coneAngles.x, theta);
```

Аргументами являются косинусы внешнего и внутреннего углов. Внутри внутреннего конуса `cone` равен примерно единице; за внешним углом — нулю. Между ними `smoothstep` создаёт плавный переход, чтобы граница света не была резкой.

```glsl
result += evaluateBlinnPhong(N, V, normalize(delta),
    light.colorIntensity.rgb * light.colorIntensity.a * cone / distanceSquared);
```

К точечному затуханию добавляется `cone`. Если поверхность вне конуса, весь вклад прожектора становится нулевым; внутри он рассчитывается по Блинну–Фонгу и ослабевает с расстоянием.

```glsl
outColor = vec4(result, 1.0);
```

Записываем окончательный цвет фрагмента и непрозрачную альфу.

## `source/application.cpp`: данные, которые использует шейдер

```cpp
struct Vertex {
    float px, py, pz;
    float nx, ny, nz;
    float r, g, b;
};
```

Именно эти поля соответствуют трём входам вершинного шейдера: позиция, нормаль и RGB-цвет.

```cpp
Vec3 normal = normalize(cross(positions[face[1]] - v0,
                              positions[face[2]] - v0));
if (dot(normal, centroid) < 0.0f)
    normal = -normal;
```

В `build_mesh()` нормаль грани получается векторным произведением двух рёбер. Проверка через центр разворачивает её наружу, если порядок вершин грани дал внутреннее направление.

```cpp
for (size_t k = 1; k + 1 < face.size(); ++k) {
    const Vec3 &a = positions[face[0]];
    const Vec3 &b = positions[face[k]];
    const Vec3 &c = positions[face[k + 1]];
```

Этот цикл разбивает выпуклую грань на треугольники веером от первой вершины. При добавлении каждой вершины треугольника в выходной массив к ней записывается рассчитанная нормаль грани.

```cpp
struct SceneUBO {
    Mat4 view, proj;
    float camera_position[4];
    float ambient[4];
    float directional_direction[4];
    float directional_color[4];
    uint32_t light_counts[4];
};
```

Это C++-представление полей `SceneUBO` в GLSL. В `render()` каждый кадр сюда записываются матрицы камеры, цвета и интенсивности света, положение камеры и количества точечных источников и прожекторов.

```cpp
struct ModelUBO {
    Mat4 model;
    float albedo[4];
    float specular_and_shininess[4];
    float emission[4];
};
```

Эти данные относятся к одной модели и совпадают по смыслу с GLSL `ModelUBO`: матрица модели, материал и параметры собственного свечения маркера.

```cpp
const VkDeviceSize point_size = light_buffer_size(point_lights.size(), sizeof(PointLightGPU));
const VkDeviceSize spot_size = light_buffer_size(spot_lights.size(), sizeof(SpotLightGPU));
```

`ensure_light_buffers()` вычисляет, сколько байт нужно для текущих массивов. Для пустого массива `light_buffer_size()` всё равно оставляет минимальный буфер.

```cpp
if (point_size > point_light_capacity) {
    destroy_buffer(point_light_ssbo);
    point_light_capacity = std::max(point_size, point_light_capacity * 2);
    point_light_ssbo = create_buffer(point_light_capacity,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true);
```

Если точечных источников стало больше, чем помещается в SSBO, старый буфер заменяется буфером нужного размера или вдвое большей ёмкости. Такой запас позволяет не выделять память заново при каждом добавлении.

```cpp
vkQueueWaitIdle(ctx.graphics_queue);
```

Перед заменой буфера код ждёт, пока GPU закончит команды, которые могли читать старый ресурс.

```cpp
VkDescriptorBufferInfo info{point_light_ssbo.buffer, 0, point_light_capacity};
VkWriteDescriptorSet write{
    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
    .dstSet = camera_set,
    .dstBinding = 1,
    .descriptorCount = 1,
    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    .pBufferInfo = &info
};
vkUpdateDescriptorSets(ctx.device, 1, &write, 0, nullptr);
```

`info` указывает на новый буфер и его диапазон. `dstSet` выбирает набор сцены, а `dstBinding = 1` — место, соответствующее массиву `points` во фрагментном шейдере. `vkUpdateDescriptorSets` заменяет привязку старого SSBO новой. Для прожекторов аналогичный код обновляет binding 2.

## `application.cpp`: заполнение данных перед отрисовкой

```cpp
const float yaw = radians(camera_yaw);
const float pitch = radians(camera_pitch);
const Vec3 forward{
    std::sin(yaw) * std::cos(pitch),
    std::sin(pitch),
    -std::cos(yaw) * std::cos(pitch)
};
camera_center = camera_eye + forward;
```

`Yaw` и `Pitch` из ImGui переводятся в радианы. По ним рассчитывается направление камеры, а прибавление этого направления к позиции камеры даёт точку, на которую она смотрит.

```cpp
camera.view = look_at(camera_eye, camera_center, camera_up);
```

`look_at()` строит матрицу вида из позиции, цели камеры и вектора вверх.

```cpp
camera.proj = perspective(...);
```

В перспективном режиме создаётся матрица с заданным FOV и соотношением сторон окна. В ортографическом режиме вместо неё записывается `orthographic(...)`.

```cpp
camera.light_counts[0] = static_cast<uint32_t>(point_lights.size());
camera.light_counts[1] = static_cast<uint32_t>(spot_lights.size());
write_ubo(camera_ubo, &camera, sizeof(SceneUBO));
```

Количество элементов в CPU-векторах передаётся в поля `lightCounts`, после чего вся структура сцены копируется в UBO. По этим количествам шейдер ограничивает оба цикла источников.

```cpp
dst.position[0] = src.position.x;
dst.position[1] = src.position.y;
dst.position[2] = src.position.z;
for (int c = 0; c < 3; ++c)
    dst.color_intensity[c] = src.color[c];
dst.color_intensity[3] = src.intensity;
```

Настройки одного источника из ImGui копируются в GPU-структуру: сначала позиция, затем RGB и интенсивность. После заполнения всех элементов массив записывается в соответствующий SSBO.

Для прожектора дополнительно нормализуется направление, а углы переводятся в радианы и косинусы:

```cpp
dst.cone_angles[0] = std::cos(radians(src.inner_angle));
dst.cone_angles[1] = std::cos(radians(src.outer_angle));
```

Поэтому во фрагментном шейдере можно сразу сравнивать `theta` с этими порогами.

## `application.cpp`: визуализации и команды draw

```cpp
write_emissive_visual(light.marker, marker, light.color);
```

Для точечного источника в UBO маркера записываются матрица маленького куба и цвет источника. У прожектора аналогично заполняется UBO цветного куба-маркера.

Для четырёх линий внешнего конуса вычисляется направление к каждой точке на окружности конуса. `oriented_box()` создаёт узкий вытянутый куб между позицией прожектора и этой точкой; его UBO получает белый emissive-цвет.

```cpp
vkCmdBindDescriptorSets(..., 0, 1, &camera_set, ...);
```

Перед рисованием моделей привязывается набор сцены: камера и массивы источников становятся доступны шейдеру.

```cpp
vkCmdBindDescriptorSets(..., 1, 1, &obj.descriptor_set, ...);
vkCmdBindVertexBuffers(..., &obj.mesh.buffer, ...);
vkCmdDraw(..., obj.mesh.vertex_count, ...);
```

Для каждой видимой модели привязывается её набор с матрицей и материалом, затем её vertex buffer и команда отрисовки.

Если включён `show_light_visualizations`, тем же кубическим мешем рисуются маркеры и линии конуса. Для каждой такой фигуры привязывается свой model descriptor set с emission-цветом. Так визуализация остаётся яркой независимо от освещения сцены.
