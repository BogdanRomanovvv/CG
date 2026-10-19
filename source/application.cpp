#include "application.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <imgui.h>
#include <vulkan/vulkan.h>

namespace
{

	// Простая структура трёхмерного вектора для координат и направлений.
	struct Vec3
	{
		float x, y, z;
	};

	// Матрица 4x4 в формате column-major, совместимом с GLSL/Vulkan.
	struct Mat4
	{
		float m[16];
	};

	// Переводит угол из градусов в радианы для математических функций C++.
	float radians(float degrees)
	{
		return degrees * 0.017453292519943295f;
	}

	// Складывает два трёхмерных вектора покомпонентно.
	Vec3 operator+(const Vec3 &a, const Vec3 &b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
	// Вычитает один трёхмерный вектор из другого.
	Vec3 operator-(const Vec3 &a, const Vec3 &b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
	// Меняет направление вектора на противоположное.
	Vec3 operator-(const Vec3 &a) { return {-a.x, -a.y, -a.z}; }
	// Умножает вектор на скаляр.
	Vec3 operator*(const Vec3 &a, float s) { return {a.x * s, a.y * s, a.z * s}; }

	// Вычисляет скалярное произведение двух векторов.
	float dot(const Vec3 &a, const Vec3 &b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

	// Вычисляет вектор, перпендикулярный двум заданным векторам.
	Vec3 cross(const Vec3 &a, const Vec3 &b)
	{
		return {
			a.y * b.z - a.z * b.y,
			a.z * b.x - a.x * b.z,
			a.x * b.y - a.y * b.x,
		};
	}

	// Нормализует вектор, то есть приводит его длину к единице.
	Vec3 normalize(const Vec3 &a)
	{
		const float len = std::sqrt(dot(a, a));
		if (len < 1e-8f)
		{
			return {0.0f, 0.0f, 0.0f};
		}
		return a * (1.0f / len);
	}

	// Возвращает единичную матрицу 4x4, которая не изменяет координаты.
	Mat4 identity()
	{
		Mat4 r{};
		r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0f;
		return r;
	}

	// Перемножает две матрицы 4x4 в формате хранения column-major.
	Mat4 multiply(const Mat4 &a, const Mat4 &b)
	{
		Mat4 r{};
		for (int c = 0; c < 4; ++c)
		{
			for (int row = 0; row < 4; ++row)
			{
				float sum = 0.0f;
				for (int k = 0; k < 4; ++k)
				{
					sum += a.m[k * 4 + row] * b.m[c * 4 + k];
				}
				r.m[c * 4 + row] = sum;
			}
		}
		return r;
	}

	// Строит перспективную матрицу проекции с глубиной Vulkan в диапазоне [0, 1].
	Mat4 perspective(float fovy_rad, float aspect, float z_near, float z_far)
	{
		const float f = 1.0f / std::tan(fovy_rad * 0.5f);
		Mat4 r{};
		r.m[0] = f / aspect;
		r.m[5] = -f; // отражение по Y для Vulkan
		r.m[10] = z_far / (z_near - z_far);
		r.m[11] = -1.0f;
		r.m[14] = (z_near * z_far) / (z_near - z_far);
		return r;
	}

	// Строит ортографическую матрицу проекции без перспективного сокращения.
	Mat4 orthographic(float left, float right, float bottom, float top,
					  float z_near, float z_far)
	{
		Mat4 r{};
		r.m[0] = 2.0f / (right - left);
		r.m[5] = -2.0f / (top - bottom); // отражение по Y для Vulkan
		r.m[10] = -1.0f / (z_far - z_near);
		r.m[12] = -(right + left) / (right - left);
		r.m[13] = (top + bottom) / (top - bottom);
		r.m[14] = -z_near / (z_far - z_near);
		r.m[15] = 1.0f;
		return r;
	}

	// Строит матрицу вида камеры по её положению, цели и направлению вверх.
	Mat4 look_at(const Vec3 &eye, const Vec3 &center, const Vec3 &up)
	{
		const Vec3 f = normalize(center - eye);
		const Vec3 s = normalize(cross(f, up));
		const Vec3 u = cross(s, f);
		Mat4 r{};
		r.m[0] = s.x;
		r.m[4] = s.y;
		r.m[8] = s.z;
		r.m[1] = u.x;
		r.m[5] = u.y;
		r.m[9] = u.z;
		r.m[2] = -f.x;
		r.m[6] = -f.y;
		r.m[10] = -f.z;
		r.m[12] = -dot(s, eye);
		r.m[13] = -dot(u, eye);
		r.m[14] = dot(f, eye);
		r.m[15] = 1.0f;
		return r;
	}

	// Создаёт матрицу переноса объекта на заданный вектор.
	Mat4 translate(const Vec3 &t)
	{
		Mat4 r = identity();
		r.m[12] = t.x;
		r.m[13] = t.y;
		r.m[14] = t.z;
		return r;
	}

	// Создаёт матрицу вращения вокруг оси X.
	Mat4 rotate_x(float rad)
	{
		Mat4 r = identity();
		const float c = std::cos(rad), s = std::sin(rad);
		r.m[5] = c;
		r.m[6] = s;
		r.m[9] = -s;
		r.m[10] = c;
		return r;
	}

	// Создаёт матрицу вращения вокруг оси Y.
	Mat4 rotate_y(float rad)
	{
		Mat4 r = identity();
		const float c = std::cos(rad), s = std::sin(rad);
		r.m[0] = c;
		r.m[2] = -s;
		r.m[8] = s;
		r.m[10] = c;
		return r;
	}

	// Создаёт матрицу вращения вокруг оси Z.
	Mat4 rotate_z(float rad)
	{
		Mat4 r = identity();
		const float c = std::cos(rad), s = std::sin(rad);
		r.m[0] = c;
		r.m[1] = s;
		r.m[4] = -s;
		r.m[5] = c;
		return r;
	}

	// Создаёт матрицу масштабирования по трём координатным осям.
	Mat4 scale(const Vec3 &s)
	{
		Mat4 r = identity();
		r.m[0] = s.x;
		r.m[5] = s.y;
		r.m[10] = s.z;
		return r;
	}

	// Сетки

	// Формат одной вершины: позиция, нормаль плоского освещения и цвет.
	struct Vertex
	{
		float px, py, pz; // позиция
		float nx, ny, nz; // нормаль грани (плоское затенение)
		float r, g, b;	  // процедурный цвет
	};

	constexpr size_t VERTEX_SIZE = sizeof(Vertex);

	// GPU-представление геометрии: vertex buffer, память VMA и число вершин.
	struct Mesh
	{
		VkBuffer buffer = VK_NULL_HANDLE;
		VmaAllocation allocation = VK_NULL_HANDLE;
		uint32_t vertex_count = 0;
	};

	// Вычисляет процедурный цвет вершины по её локальной позиции.
	Vec3 procedural_color(const Vec3 &p)
	{
		return {
			0.5f + 0.5f * p.x,
			0.5f + 0.5f * p.y,
			0.5f + 0.5f * p.z,
		};
	}

	// Строит плоско-затенённую сетку из позиций вершин и списка граней.
	// Каждая грань задаётся индексами вершин в циклическом порядке.
	std::vector<Vertex> build_mesh(const std::vector<Vec3> &positions,
								   const std::vector<std::vector<int>> &faces)
	{
		std::vector<Vertex> out;
		for (const auto &face : faces)
		{
			Vec3 centroid{0.0f, 0.0f, 0.0f};
			for (int idx : face)
			{
				centroid = centroid + positions[idx];
			}
			centroid = centroid * (1.0f / static_cast<float>(face.size()));

			const Vec3 &v0 = positions[face[0]];
			Vec3 normal = normalize(cross(positions[face[1]] - v0, positions[face[2]] - v0));
			if (dot(normal, centroid) < 0.0f)
			{
				normal = -normal;
			}

			// триангуляция веером выпуклого многоугольника
			for (size_t k = 1; k + 1 < face.size(); ++k)
			{
				const Vec3 &a = positions[face[0]];
				const Vec3 &b = positions[face[k]];
				const Vec3 &c = positions[face[k + 1]];
				for (const Vec3 *p : {&a, &b, &c})
				{
					const Vec3 color = procedural_color(*p);
					out.push_back({p->x, p->y, p->z, normal.x, normal.y, normal.z,
								   color.x, color.y, color.z});
				}
			}
		}
		return out;
	}

	// Создаёт геометрию правильного додекаэдра: 20 вершин и 12 пятиугольных граней.
	std::vector<Vertex> make_dodecahedron()
	{
		static const Vec3 vertices[20] = {
			{-0.57735027f, -0.57735027f, -0.57735027f},
			{-0.57735027f, -0.57735027f, 0.57735027f},
			{-0.57735027f, 0.57735027f, -0.57735027f},
			{-0.57735027f, 0.57735027f, 0.57735027f},
			{0.57735027f, -0.57735027f, -0.57735027f},
			{0.57735027f, -0.57735027f, 0.57735027f},
			{0.57735027f, 0.57735027f, -0.57735027f},
			{0.57735027f, 0.57735027f, 0.57735027f},
			{0.00000000f, -0.35682209f, -0.93417236f},
			{0.00000000f, -0.35682209f, 0.93417236f},
			{0.00000000f, 0.35682209f, -0.93417236f},
			{0.00000000f, 0.35682209f, 0.93417236f},
			{-0.35682209f, -0.93417236f, 0.00000000f},
			{-0.35682209f, 0.93417236f, 0.00000000f},
			{0.35682209f, -0.93417236f, 0.00000000f},
			{0.35682209f, 0.93417236f, 0.00000000f},
			{-0.93417236f, 0.00000000f, -0.35682209f},
			{-0.93417236f, 0.00000000f, 0.35682209f},
			{0.93417236f, 0.00000000f, -0.35682209f},
			{0.93417236f, 0.00000000f, 0.35682209f},
		};

		static const int faces[12][5] = {
			{17, 16, 0, 12, 1},
			{10, 8, 0, 16, 2},
			{14, 12, 0, 8, 4},
			{3, 17, 1, 9, 11},
			{5, 9, 1, 12, 14},
			{3, 13, 2, 16, 17},
			{6, 10, 2, 13, 15},
			{15, 13, 3, 11, 7},
			{5, 14, 4, 18, 19},
			{6, 18, 4, 8, 10},
			{11, 9, 5, 19, 7},
			{19, 18, 6, 15, 7},
		};

		std::vector<Vec3> positions(vertices, vertices + 20);
		std::vector<std::vector<int>> face_list;
		for (int f = 0; f < 12; ++f)
		{
			face_list.push_back({faces[f][0], faces[f][1], faces[f][2], faces[f][3], faces[f][4]});
		}
		return build_mesh(positions, face_list);
	}

	// Создаёт вспомогательный куб и преобразует его грани в список треугольников.
	std::vector<Vertex> make_cube()
	{
		const float s = 1.0f / std::sqrt(3.0f); // описанный радиус ~1
		const std::vector<Vec3> p = {
			{-s, -s, -s},
			{s, -s, -s},
			{s, s, -s},
			{-s, s, -s},
			{-s, -s, s},
			{s, -s, s},
			{s, s, s},
			{-s, s, s},
		};
		const std::vector<std::vector<int>> faces = {
			{0, 1, 2, 3},
			{4, 5, 6, 7},
			{0, 1, 5, 4},
			{2, 3, 7, 6},
			{0, 3, 7, 4},
			{1, 2, 6, 5},
		};
		return build_mesh(p, faces);
	}

	// Создаёт вспомогательный октаэдр из восьми треугольных граней.
	std::vector<Vertex> make_octahedron()
	{
		const std::vector<Vec3> p = {
			{1, 0, 0},
			{-1, 0, 0},
			{0, 1, 0},
			{0, -1, 0},
			{0, 0, 1},
			{0, 0, -1},
		};
		const std::vector<std::vector<int>> faces = {
			{0, 2, 4},
			{2, 1, 4},
			{1, 3, 4},
			{3, 0, 4},
			{2, 0, 5},
			{1, 2, 5},
			{3, 1, 5},
			{0, 3, 5},
		};
		return build_mesh(p, faces);
	}

	// Вспомогательные функции Vulkan

	const graphics::internal::Context &ctx = graphics::internal::context;

	// Загружает скомпилированный SPIR-V-файл и создаёт из него модуль Vulkan.
	VkShaderModule load_shader(const char *path)
	{
		std::ifstream file(path, std::ios::ate | std::ios::binary);
		if (!file.is_open())
		{
			std::cerr << "Failed to open shader file: " << path << '\n';
			return VK_NULL_HANDLE;
		}
		const size_t size = static_cast<size_t>(file.tellg());
		std::vector<char> code(size);
		file.seekg(0);
		file.read(code.data(), static_cast<std::streamsize>(size));
		file.close();

		VkShaderModuleCreateInfo create_info = {
			.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
			.codeSize = size,
			.pCode = reinterpret_cast<const uint32_t *>(code.data()),
		};
		VkShaderModule module = VK_NULL_HANDLE;
		if (vkCreateShaderModule(ctx.device, &create_info, nullptr, &module) != VK_SUCCESS)
		{
			std::cerr << "Failed to create shader module: " << path << '\n';
		}
		return module;
	}

	// Обёртка над Vulkan-буфером и выделенной VMA-памятью.
	struct Buffer
	{
		VkBuffer buffer = VK_NULL_HANDLE;
		VmaAllocation allocation = VK_NULL_HANDLE;
		void *mapped = nullptr;
	};

	// Создаёт Vulkan-буфер через VMA и при необходимости отображает его в память CPU.
	Buffer create_buffer(VkDeviceSize size, VkBufferUsageFlags usage, bool host_visible)
	{
		VkBufferCreateInfo buffer_info = {
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = size,
			.usage = usage,
			.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		};

		VmaAllocationCreateInfo alloc_info = {};
		alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
		if (host_visible)
		{
			alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
							   VMA_ALLOCATION_CREATE_MAPPED_BIT;
		}

		Buffer out;
		if (vmaCreateBuffer(ctx.allocator, &buffer_info, &alloc_info, &out.buffer,
							&out.allocation, nullptr) != VK_SUCCESS)
		{
			std::cerr << "Failed to allocate Vulkan buffer\n";
			return out;
		}
		if (host_visible)
		{
			vmaMapMemory(ctx.allocator, out.allocation, &out.mapped);
		}
		return out;
	}

	// Освобождает буфер, его память VMA и отображение памяти.
	void destroy_buffer(Buffer &b)
	{
		if (b.buffer == VK_NULL_HANDLE)
		{
			return;
		}
		if (b.mapped)
		{
			vmaUnmapMemory(ctx.allocator, b.allocation);
		}
		vmaDestroyBuffer(ctx.allocator, b.buffer, b.allocation);
		b.buffer = VK_NULL_HANDLE;
		b.allocation = VK_NULL_HANDLE;
		b.mapped = nullptr;
	}

	// Выделяет одноразовый командный буфер и начинает запись команд в него.
	VkCommandBuffer begin_one_time_command(VkCommandPool pool)
	{
		VkCommandBufferAllocateInfo alloc = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
			.commandPool = pool,
			.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
			.commandBufferCount = 1,
		};
		VkCommandBuffer cb = VK_NULL_HANDLE;
		vkAllocateCommandBuffers(ctx.device, &alloc, &cb);

		VkCommandBufferBeginInfo begin = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
		};
		vkBeginCommandBuffer(cb, &begin);
		return cb;
	}

	// Завершает запись одноразового буфера, отправляет его в очередь и освобождает.
	void end_one_time_command(VkCommandPool pool, VkCommandBuffer cb)
	{
		vkEndCommandBuffer(cb);

		VkSubmitInfo submit = {
			.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
			.commandBufferCount = 1,
			.pCommandBuffers = &cb,
		};
		vkQueueSubmit(ctx.graphics_queue, 1, &submit, VK_NULL_HANDLE);
		vkQueueWaitIdle(ctx.graphics_queue);
		vkFreeCommandBuffers(ctx.device, pool, 1, &cb);
	}

	// Копирует данные из CPU-доступного staging-буфера в буфер памяти GPU.
	void upload_to_device_buffer(Buffer &dst, const void *data, VkDeviceSize size,
								 VkCommandPool pool)
	{
		Buffer staging = create_buffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true);
		std::memcpy(staging.mapped, data, static_cast<size_t>(size));

		VkCommandBuffer cb = begin_one_time_command(pool);
		VkBufferCopy region = {0, 0, size};
		vkCmdCopyBuffer(cb, staging.buffer, dst.buffer, 1, &region);
		end_one_time_command(pool, cb);

		destroy_buffer(staging);
	}

	// Создаёт vertex buffer для меша и загружает в него массив вершин.
	Mesh create_mesh(const std::vector<Vertex> &vertices, VkCommandPool pool)
	{
		Mesh mesh;
		mesh.vertex_count = static_cast<uint32_t>(vertices.size());
		const VkDeviceSize size = vertices.size() * VERTEX_SIZE;
		Buffer vb = create_buffer(size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, false);
		mesh.buffer = vb.buffer;
		mesh.allocation = vb.allocation;
		upload_to_device_buffer(vb, vertices.data(), size, pool);
		return mesh;
	}

	// Уничтожает vertex buffer меша и освобождает его память.
	void destroy_mesh(Mesh &mesh)
	{
		if (mesh.buffer == VK_NULL_HANDLE)
		{
			return;
		}
		vmaDestroyBuffer(ctx.allocator, mesh.buffer, mesh.allocation);
		mesh.buffer = VK_NULL_HANDLE;
		mesh.allocation = VK_NULL_HANDLE;
	}

	// Состояние сцены

	// Общие параметры камеры и освещения передаются в наборе сцены.
	struct SceneUBO
	{
		Mat4 view, proj;
		float camera_position[4];
		float ambient[4];
		float directional_direction[4];
		float directional_color[4];
		uint32_t light_counts[4];
	};
	struct ModelUBO
	{
		Mat4 model;
		float albedo[4];
		float specular_and_shininess[4];
	};
	struct PointLightGPU { float position[4]; float color_intensity[4]; };
	struct SpotLightGPU { float position[4]; float direction[4]; float color_intensity[4]; float cone_angles[4]; };
	struct PointLightSettings
	{
		Vec3 position{1.5f, 1.5f, 2.0f};
		float color[3]{1.0f, 0.78f, 0.55f};
		float intensity = 8.0f;
	};
	struct SpotLightSettings
	{
		Vec3 position{-1.5f, 2.2f, 1.5f};
		Vec3 direction{0.4f, -0.6f, -0.7f};
		float color[3]{0.55f, 0.7f, 1.0f};
		float intensity = 14.0f;
		float inner_angle = 18.0f;
		float outer_angle = 28.0f;
	};

	// Состояние объекта сцены и его Vulkan-ресурсы.
	struct Object
	{
		Mesh mesh;
		Buffer ubo;
		VkDescriptorSet descriptor_set = VK_NULL_HANDLE;

		Vec3 position{0.0f, 0.0f, 0.0f};
		Vec3 rotation{0.0f, 0.0f, 0.0f}; // градусы
		Vec3 scale_f{1.0f, 1.0f, 1.0f};
		float albedo[3] = {1.0f, 1.0f, 1.0f};
		float specular_color[3] = {0.8f, 0.8f, 0.8f};
		float shininess = 48.0f;

		bool visible = true;
		bool animate = false;

		Vec3 anim_position{0.0f, 0.0f, 0.0f};
		Vec3 anim_rotation{0.0f, 0.0f, 0.0f};

		// Собирает модельную матрицу: перенос, вращения, затем масштабирование.
		Mat4 model_matrix() const
		{
			const Vec3 pos = position + anim_position;
			const Vec3 rot = rotation + anim_rotation;
			Mat4 m = translate(pos);
			m = multiply(m, rotate_z(radians(rot.z)));
			m = multiply(m, rotate_y(radians(rot.y)));
			m = multiply(m, rotate_x(radians(rot.x)));
			m = multiply(m, scale(scale_f));
			return m;
		}
	};

	// Камера
	bool use_perspective = true;
	Vec3 camera_eye{0.0f, 1.6f, 4.6f};
	Vec3 camera_center{0.0f, 0.0f, 0.0f};
	Vec3 camera_up{0.0f, 1.0f, 0.0f};
	float camera_fov = 45.0f;
	float camera_yaw = 0.0f;
	float camera_pitch = -19.2f;
	float ambient_color[3]{0.18f, 0.18f, 0.22f};
	float ambient_intensity = 0.7f;
	Vec3 directional_direction{0.4f, -0.8f, -0.6f};
	float directional_color[3]{1.0f, 0.96f, 0.88f};
	float directional_intensity = 1.2f;
	std::vector<PointLightSettings> point_lights;
	std::vector<SpotLightSettings> spot_lights;

	// Анимация
	bool animation_playing = true;
	float animation_speed = 1.0f;
	float animation_radius = 1.4f;
	double animation_time = 0.0;
	double last_frame_time = 0.0;

	// Ресурсы Vulkan
	VkDescriptorSetLayout camera_layout = VK_NULL_HANDLE;
	VkDescriptorSetLayout object_layout = VK_NULL_HANDLE;
	VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
	VkDescriptorSet camera_set = VK_NULL_HANDLE;
	Buffer camera_ubo;
	Buffer point_light_ssbo;
	Buffer spot_light_ssbo;
	VkDeviceSize point_light_capacity = 0;
	VkDeviceSize spot_light_capacity = 0;

	VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
	VkPipeline pipeline = VK_NULL_HANDLE;
	VkCommandPool command_pool = VK_NULL_HANDLE;

	std::vector<Object> objects;

	// Записывает структуру с параметрами в отображённый uniform buffer.
	void write_ubo(Buffer &buf, const void *data, VkDeviceSize size)
	{
		std::memcpy(buf.mapped, data, static_cast<size_t>(size));
		vmaFlushAllocation(ctx.allocator, buf.allocation, 0, size);
	}

	VkDeviceSize light_buffer_size(size_t count, size_t item_size)
	{
		return static_cast<VkDeviceSize>(std::max<size_t>(count, 1) * item_size);
	}

	void ensure_light_buffers()
	{
		const VkDeviceSize point_size = light_buffer_size(point_lights.size(), sizeof(PointLightGPU));
		const VkDeviceSize spot_size = light_buffer_size(spot_lights.size(), sizeof(SpotLightGPU));
		if (point_size <= point_light_capacity && spot_size <= spot_light_capacity)
			return;
		vkQueueWaitIdle(ctx.graphics_queue);
		if (point_size > point_light_capacity)
		{
			destroy_buffer(point_light_ssbo);
			point_light_capacity = std::max(point_size, point_light_capacity * 2);
			point_light_ssbo = create_buffer(point_light_capacity, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true);
			VkDescriptorBufferInfo info{point_light_ssbo.buffer, 0, point_light_capacity};
			VkWriteDescriptorSet write{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = camera_set,
				.dstBinding = 1, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &info};
			vkUpdateDescriptorSets(ctx.device, 1, &write, 0, nullptr);
		}
		if (spot_size > spot_light_capacity)
		{
			destroy_buffer(spot_light_ssbo);
			spot_light_capacity = std::max(spot_size, spot_light_capacity * 2);
			spot_light_ssbo = create_buffer(spot_light_capacity, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true);
			VkDescriptorBufferInfo info{spot_light_ssbo.buffer, 0, spot_light_capacity};
			VkWriteDescriptorSet write{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = camera_set,
				.dstBinding = 2, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &info};
			vkUpdateDescriptorSets(ctx.device, 1, &write, 0, nullptr);
		}
	}

} // пространство имён

// Публичный интерфейс

namespace application
{

	// Создаёт все ресурсы приложения: меши, UBO, descriptor sets и graphics pipeline.
	bool initialize()
	{
		// 1. Создание пула команд для загрузки через staging.
		VkCommandPoolCreateInfo pool_info = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
			.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
			.queueFamilyIndex = ctx.graphics_queue_index,
		};
		if (vkCreateCommandPool(ctx.device, &pool_info, nullptr, &command_pool) != VK_SUCCESS)
		{
			std::cerr << "Failed to create application command pool\n";
			return false;
		}

		// 2. Создание сеток и добавление трёх объектов на сцену.
		Mesh dodecahedron = create_mesh(make_dodecahedron(), command_pool);
		Mesh cube = create_mesh(make_cube(), command_pool);
		Mesh octahedron = create_mesh(make_octahedron(), command_pool);

		objects.resize(3);
		objects[0].mesh = dodecahedron;
		objects[1].mesh = cube;
		objects[2].mesh = octahedron;

		// Начальное расположение, масштаб и видимость объектов.
		objects[0].position = {0.0f, 0.0f, 0.0f};
		objects[0].scale_f = {1.4f, 1.4f, 1.4f};
		objects[0].animate = true;

		objects[1].position = {2.0f, -0.4f, 0.0f};
		objects[1].scale_f = {0.55f, 0.55f, 0.55f};
		objects[1].albedo[0] = 1.0f;
		objects[1].albedo[1] = 1.0f;
		objects[1].albedo[2] = 1.0f;
		objects[1].visible = true;

		objects[2].position = {-2.0f, -0.4f, 0.0f};
		objects[2].scale_f = {0.7f, 0.7f, 0.7f};
		objects[2].visible = true;

		// 3. Макеты наборов дескрипторов: set 0 для камеры и set 1 для объектов.
		VkDescriptorSetLayoutBinding scene_bindings[] = {
			{.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1,
			 .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT},
			{.binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1,
			 .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT},
			{.binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1,
			 .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT},
		};
		VkDescriptorSetLayoutCreateInfo camera_layout_info = {
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
			.bindingCount = 3,
			.pBindings = scene_bindings,
		};
		if (vkCreateDescriptorSetLayout(ctx.device, &camera_layout_info, nullptr, &camera_layout) != VK_SUCCESS)
		{
			std::cerr << "Failed to create camera descriptor set layout\n";
			return false;
		}

		VkDescriptorSetLayoutBinding object_binding = {
			.binding = 0,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
		};
		VkDescriptorSetLayoutCreateInfo object_layout_info = {
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
			.bindingCount = 1,
			.pBindings = &object_binding,
		};
		if (vkCreateDescriptorSetLayout(ctx.device, &object_layout_info, nullptr, &object_layout) != VK_SUCCESS)
		{
			std::cerr << "Failed to create object descriptor set layout\n";
			return false;
		}

		// 4. Пул дескрипторов: один набор для камеры и отдельный набор для каждого объекта.
		const uint32_t object_count = static_cast<uint32_t>(objects.size());
		VkDescriptorPoolSize pool_sizes[] = {
			{.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = object_count + 1},
			{.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 2},
		};
		VkDescriptorPoolCreateInfo pool_create = {
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
			.maxSets = object_count + 1,
			.poolSizeCount = 2,
			.pPoolSizes = pool_sizes,
		};
		if (vkCreateDescriptorPool(ctx.device, &pool_create, nullptr, &descriptor_pool) != VK_SUCCESS)
		{
			std::cerr << "Failed to create descriptor pool\n";
			return false;
		}

		// 5. UBO камеры и общий descriptor set с матрицами вида и проекции.
		camera_ubo = create_buffer(sizeof(SceneUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true);
		point_light_capacity = light_buffer_size(0, sizeof(PointLightGPU));
		spot_light_capacity = light_buffer_size(0, sizeof(SpotLightGPU));
		point_light_ssbo = create_buffer(point_light_capacity,
			VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true);
		spot_light_ssbo = create_buffer(spot_light_capacity,
			VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true);

		VkDescriptorSetAllocateInfo camera_alloc = {
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
			.descriptorPool = descriptor_pool,
			.descriptorSetCount = 1,
			.pSetLayouts = &camera_layout,
		};
		if (vkAllocateDescriptorSets(ctx.device, &camera_alloc, &camera_set) != VK_SUCCESS)
		{
			std::cerr << "Failed to allocate camera descriptor set\n";
			return false;
		}

		VkDescriptorBufferInfo scene_buffer_infos[] = {
			{
			.buffer = camera_ubo.buffer,
			.offset = 0,
			.range = sizeof(SceneUBO),
			},
			{.buffer = point_light_ssbo.buffer, .offset = 0, .range = point_light_capacity},
			{.buffer = spot_light_ssbo.buffer, .offset = 0, .range = spot_light_capacity},
		};
		VkWriteDescriptorSet scene_writes[] = {
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = camera_set,
			.dstBinding = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.pBufferInfo = &scene_buffer_infos[0],
		},
		{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = camera_set, .dstBinding = 1,
		 .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &scene_buffer_infos[1]},
		{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = camera_set, .dstBinding = 2,
		 .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &scene_buffer_infos[2]},
		};
		vkUpdateDescriptorSets(ctx.device, 3, scene_writes, 0, nullptr);

		// 6. Для каждого объекта создаются собственные UBO и descriptor set.
		for (Object &obj : objects)
		{
			obj.ubo = create_buffer(sizeof(ModelUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true);

			VkDescriptorSetAllocateInfo alloc = {
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
				.descriptorPool = descriptor_pool,
				.descriptorSetCount = 1,
				.pSetLayouts = &object_layout,
			};
			if (vkAllocateDescriptorSets(ctx.device, &alloc, &obj.descriptor_set) != VK_SUCCESS)
			{
				std::cerr << "Failed to allocate object descriptor set\n";
				return false;
			}

			VkDescriptorBufferInfo buffer_info = {
				.buffer = obj.ubo.buffer,
				.offset = 0,
				.range = sizeof(ModelUBO),
			};
			VkWriteDescriptorSet write = {
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = obj.descriptor_set,
				.dstBinding = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.pBufferInfo = &buffer_info,
			};
			vkUpdateDescriptorSets(ctx.device, 1, &write, 0, nullptr);
		}

		// 7. Макет пайплайна связывает layout камеры и layout объекта с шейдерами.
		const VkDescriptorSetLayout layouts[] = {camera_layout, object_layout};
		VkPipelineLayoutCreateInfo pipeline_layout_info = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.setLayoutCount = 2,
			.pSetLayouts = layouts,
		};
		if (vkCreatePipelineLayout(ctx.device, &pipeline_layout_info, nullptr, &pipeline_layout) != VK_SUCCESS)
		{
			std::cerr << "Failed to create pipeline layout\n";
			return false;
		}

		// 8. Загрузка SPIR-V-шэйдеров и описание двух стадий graphics pipeline.
		VkShaderModule vert = load_shader("shaders/shader.vert.spv");
		VkShaderModule frag = load_shader("shaders/shader.frag.spv");
		if (vert == VK_NULL_HANDLE || frag == VK_NULL_HANDLE)
		{
			return false;
		}

		VkPipelineShaderStageCreateInfo vert_stage = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vert,
			.pName = "main",
		};
		VkPipelineShaderStageCreateInfo frag_stage = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = frag,
			.pName = "main",
		};
		const VkPipelineShaderStageCreateInfo stages[] = {vert_stage, frag_stage};

		// Описание формата Vertex: позиция, нормаль и процедурный цвет.
		const VkVertexInputBindingDescription binding = {
			.binding = 0,
			.stride = VERTEX_SIZE,
			.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
		};
		const VkVertexInputAttributeDescription attributes[] = {
			{0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0},
			{1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, nx)},
			{2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, r)},
		};
		VkPipelineVertexInputStateCreateInfo vertex_input = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
			.vertexBindingDescriptionCount = 1,
			.pVertexBindingDescriptions = &binding,
			.vertexAttributeDescriptionCount = 3,
			.pVertexAttributeDescriptions = attributes,
		};

		// Вершины собираются по три, поэтому пятиугольники уже должны
		// быть заранее разбиты на треугольники в build_mesh().
		VkPipelineInputAssemblyStateCreateInfo input_assembly = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
			.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
			.primitiveRestartEnable = VK_FALSE,
		};

		VkPipelineViewportStateCreateInfo viewport_state = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
			.viewportCount = 1,
			.scissorCount = 1,
		};

		VkPipelineRasterizationStateCreateInfo rasterizer = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
			.depthClampEnable = VK_FALSE,
			.rasterizerDiscardEnable = VK_FALSE,
			.polygonMode = VK_POLYGON_MODE_FILL,
			.cullMode = VK_CULL_MODE_NONE,
			.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
			.depthBiasEnable = VK_FALSE,
			.lineWidth = 1.0f,
		};

		// Для учебной сцены отключаем отсечение граней, чтобы были видны
		// обе стороны геометрии.
		VkPipelineMultisampleStateCreateInfo multisampling = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
			.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
		};

		VkPipelineDepthStencilStateCreateInfo depth_stencil = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
			.depthTestEnable = VK_TRUE,
			.depthWriteEnable = VK_TRUE,
			.depthCompareOp = VK_COMPARE_OP_LESS,
			.depthBoundsTestEnable = VK_FALSE,
			.stencilTestEnable = VK_FALSE,
		};

		// Тест глубины оставляет ближайший к камере фрагмент.
		VkPipelineColorBlendAttachmentState color_attachment = {
			.blendEnable = VK_FALSE,
			.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
							  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
		};
		VkPipelineColorBlendStateCreateInfo color_blend = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
			.attachmentCount = 1,
			.pAttachments = &color_attachment,
		};

		const VkDynamicState dynamic_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
		VkPipelineDynamicStateCreateInfo dynamic_state = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
			.dynamicStateCount = 2,
			.pDynamicStates = dynamic_states,
		};

		VkGraphicsPipelineCreateInfo pipeline_info = {
			.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
			.stageCount = 2,
			.pStages = stages,
			.pVertexInputState = &vertex_input,
			.pInputAssemblyState = &input_assembly,
			.pViewportState = &viewport_state,
			.pRasterizationState = &rasterizer,
			.pMultisampleState = &multisampling,
			.pDepthStencilState = &depth_stencil,
			.pColorBlendState = &color_blend,
			.pDynamicState = &dynamic_state,
			.layout = pipeline_layout,
			.renderPass = ctx.render_pass,
			.subpass = 0,
		};

		if (vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline) != VK_SUCCESS)
		{
			std::cerr << "Failed to create graphics pipeline\n";
			return false;
		}

		vkDestroyShaderModule(ctx.device, vert, nullptr);
		vkDestroyShaderModule(ctx.device, frag, nullptr);

		return true;
	}

	// Освобождает все созданные Vulkan-ресурсы приложения в обратном порядке.
	void shutdown()
	{
		auto &context = graphics::internal::context;
		vkQueueWaitIdle(context.graphics_queue);

		if (pipeline != VK_NULL_HANDLE)
		{
			vkDestroyPipeline(context.device, pipeline, nullptr);
		}
		if (pipeline_layout != VK_NULL_HANDLE)
		{
			vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr);
		}

		for (Object &obj : objects)
		{
			destroy_mesh(obj.mesh);
			destroy_buffer(obj.ubo);
		}
		destroy_buffer(camera_ubo);
		destroy_buffer(point_light_ssbo);
		destroy_buffer(spot_light_ssbo);

		if (descriptor_pool != VK_NULL_HANDLE)
		{
			vkDestroyDescriptorPool(context.device, descriptor_pool, nullptr);
		}
		if (camera_layout != VK_NULL_HANDLE)
		{
			vkDestroyDescriptorSetLayout(context.device, camera_layout, nullptr);
		}
		if (object_layout != VK_NULL_HANDLE)
		{
			vkDestroyDescriptorSetLayout(context.device, object_layout, nullptr);
		}
		if (command_pool != VK_NULL_HANDLE)
		{
			vkDestroyCommandPool(context.device, command_pool, nullptr);
		}
	}

	// Обрабатывает ввод и ImGui, обновляет время и параметры анимации.
	void update(double time)
	{
		// Дельта времени кадра для анимации.
		const double dt = (last_frame_time == 0.0) ? 0.0 : (time - last_frame_time);
		last_frame_time = time;
		if (animation_playing)
		{
			animation_time += dt * animation_speed;
		}

		Object &dodecahedron = objects[0];
		const ImGuiIO &io = ImGui::GetIO();
		const bool input_captured_by_ui = io.WantCaptureMouse || io.WantCaptureKeyboard;

		// Управление додекаэдром: зажатая правая кнопка мыши вращает объект,
		// клавиши WASD перемещают его по горизонтальной плоскости.
		if (!input_captured_by_ui)
		{
			if (ImGui::IsMouseDragging(ImGuiMouseButton_Right))
			{
				dodecahedron.rotation.y += io.MouseDelta.x * 0.35f;
				dodecahedron.rotation.x += io.MouseDelta.y * 0.35f;
			}

			const float move_speed = 2.5f * static_cast<float>(dt);
			if (ImGui::IsKeyDown(ImGuiKey_W))
			{
				dodecahedron.position.z -= move_speed;
			}
			if (ImGui::IsKeyDown(ImGuiKey_S))
			{
				dodecahedron.position.z += move_speed;
			}
			if (ImGui::IsKeyDown(ImGuiKey_A))
			{
				dodecahedron.position.x -= move_speed;
			}
			if (ImGui::IsKeyDown(ImGuiKey_D))
			{
				dodecahedron.position.x += move_speed;
			}
		}

		// ----- Панель управления ImGui -----
		ImGui::Begin("Control Panel");

		ImGui::Text("Variant 10: regular dodecahedron");
		ImGui::Text("RMB + mouse: rotate | WASD: move");

		ImGui::Separator();
		ImGui::Text("Camera");
		ImGui::DragFloat3("Position##camera", &camera_eye.x, 0.05f, -10.0f, 10.0f);
		ImGui::SliderFloat("Yaw", &camera_yaw, -180.0f, 180.0f);
		ImGui::SliderFloat("Pitch", &camera_pitch, -89.0f, 89.0f);
		ImGui::SliderFloat("Field of view", &camera_fov, 20.0f, 100.0f);
		ImGui::Separator();
		ImGui::Text("Projection");
		int projection = use_perspective ? 0 : 1;
		const char *projection_items[] = {"Perspective", "Orthographic"};
		if (ImGui::Combo("Projection", &projection, projection_items, 2))
		{
			use_perspective = (projection == 0);
		}
		ImGui::Separator();
		ImGui::Separator();
		// Основные элементы управления объектом (додекаэдр).
		ImGui::Text("Dodecahedron transform");
		ImGui::DragFloat3("Position", &dodecahedron.position.x, 0.05f, -5.0f, 5.0f);
		ImGui::DragFloat3("Rotation (deg)", &dodecahedron.rotation.x, 0.5f, -360.0f, 360.0f);
		ImGui::DragFloat3("Scale", &dodecahedron.scale_f.x, 0.05f, 0.1f, 4.0f);

		ImGui::Separator();
		ImGui::Text("Material (Blinn-Phong)");
		ImGui::ColorEdit3("Albedo", dodecahedron.albedo);
		ImGui::ColorEdit3("Specular color", dodecahedron.specular_color);
		ImGui::SliderFloat("Shininess", &dodecahedron.shininess, 1.0f, 256.0f, "%.0f");

		ImGui::Separator();
		ImGui::Text("Ambient light");
		ImGui::ColorEdit3("Ambient color", ambient_color);
		ImGui::SliderFloat("Ambient intensity", &ambient_intensity, 0.0f, 3.0f);
		ImGui::Text("Directional light");
		ImGui::DragFloat3("Direction toward scene", &directional_direction.x, 0.02f, -1.0f, 1.0f);
		ImGui::ColorEdit3("Directional color", directional_color);
		ImGui::SliderFloat("Directional intensity", &directional_intensity, 0.0f, 8.0f);

		ImGui::Separator();
		ImGui::Text("Point lights (inverse-square falloff)");
		if (ImGui::Button("Add point light")) point_lights.emplace_back();
		for (size_t i = 0; i < point_lights.size(); ++i)
		{
			ImGui::PushID(static_cast<int>(i));
			PointLightSettings &light = point_lights[i];
			if (ImGui::TreeNode("Point light"))
			{
				ImGui::DragFloat3("Position", &light.position.x, 0.05f, -8.0f, 8.0f);
				ImGui::ColorEdit3("Color", light.color);
				ImGui::SliderFloat("Intensity", &light.intensity, 0.0f, 80.0f);
				if (ImGui::Button("Remove point light")) { point_lights.erase(point_lights.begin() + i); ImGui::TreePop(); ImGui::PopID(); break; }
				ImGui::TreePop();
			}
			ImGui::PopID();
		}

		ImGui::Separator();
		ImGui::Text("Spot lights (inverse-square falloff)");
		if (ImGui::Button("Add spot light")) spot_lights.emplace_back();
		for (size_t i = 0; i < spot_lights.size(); ++i)
		{
			ImGui::PushID(static_cast<int>(i));
			SpotLightSettings &light = spot_lights[i];
			if (ImGui::TreeNode("Spot light"))
			{
				ImGui::DragFloat3("Position", &light.position.x, 0.05f, -8.0f, 8.0f);
				ImGui::DragFloat3("Direction", &light.direction.x, 0.02f, -1.0f, 1.0f);
				ImGui::ColorEdit3("Color", light.color);
				ImGui::SliderFloat("Intensity", &light.intensity, 0.0f, 120.0f);
				ImGui::SliderFloat("Inner angle", &light.inner_angle, 1.0f, light.outer_angle);
				ImGui::SliderFloat("Outer angle", &light.outer_angle, light.inner_angle, 89.0f);
				if (ImGui::Button("Remove spot light")) { spot_lights.erase(spot_lights.begin() + i); ImGui::TreePop(); ImGui::PopID(); break; }
				ImGui::TreePop();
			}
			ImGui::PopID();
		}

		ImGui::Separator();
		ImGui::Text("Animation (complex trajectory)");
		ImGui::Checkbox("Enabled", &dodecahedron.animate);
		if (dodecahedron.animate)
		{
			ImGui::Checkbox("Play", &animation_playing);
			ImGui::DragFloat("Speed", &animation_speed, 0.05f, 0.1f, 5.0f);
			ImGui::DragFloat("Trajectory radius", &animation_radius, 0.05f, 0.2f, 3.0f);
		}

		ImGui::Separator();
		ImGui::Text("Additional objects (own descriptor sets)");
		for (size_t i = 1; i < objects.size(); ++i)
		{
			Object &obj = objects[i];
			const char *label = (i == 1) ? "Cube" : "Octahedron";
			ImGui::Checkbox(label, &obj.visible);
			if (obj.visible)
			{
				ImGui::SameLine();
				ImGui::DragFloat3(("##pos" + std::to_string(i)).c_str(), &obj.position.x, 0.05f, -5.0f, 5.0f);
				ImGui::PushID(static_cast<int>(i));
				if (ImGui::TreeNode("Material"))
				{
					ImGui::ColorEdit3("Albedo", obj.albedo);
					ImGui::ColorEdit3("Specular color", obj.specular_color);
					ImGui::SliderFloat("Shininess", &obj.shininess, 1.0f, 256.0f, "%.0f");
					ImGui::TreePop();
				}
				ImGui::PopID();
			}
		}

		ImGui::End();

		// ----- Вычисление смещений анимации по кривой Лиссажу -----
		for (Object &obj : objects)
		{
			if (!obj.animate)
			{
				obj.anim_position = {0.0f, 0.0f, 0.0f};
				obj.anim_rotation = {0.0f, 0.0f, 0.0f};
				continue;
			}
			const float t = static_cast<float>(animation_time);
			// Трёхмерная траектория в стиле Лиссажу.
			obj.anim_position = {
				animation_radius * std::sin(t * 1.0f),
				animation_radius * 0.6f * std::sin(t * 2.0f + 0.7f),
				animation_radius * std::cos(t * 1.3f),
			};
			obj.anim_rotation = {
				t * 40.0f,
				t * 25.0f,
				t * 15.0f,
			};
		}
	}

	// Записывает команды Vulkan для построения и отрисовки текущего кадра.
	void render(const graphics::internal::FrameData &fd)
	{
		auto &context = graphics::internal::context;

		if (fd.command_buffer == VK_NULL_HANDLE)
		{
			return;
		}

		// Формирование матриц камеры и запись их в общий UBO.
		const float aspect = static_cast<float>(context.swapchain_extent.width) /
							 static_cast<float>(context.swapchain_extent.height);
		const float yaw = radians(camera_yaw);
		const float pitch = radians(camera_pitch);
		const Vec3 forward{std::sin(yaw) * std::cos(pitch), std::sin(pitch), -std::cos(yaw) * std::cos(pitch)};
		camera_center = camera_eye + forward;
		SceneUBO camera{};
		camera.view = look_at(camera_eye, camera_center, camera_up);
		if (use_perspective)
		{
			camera.proj = perspective(radians(camera_fov), aspect, 0.1f, 100.0f);
		}
		else
		{
			const float half_h = 2.4f;
			const float half_w = half_h * aspect;
			camera.proj = orthographic(-half_w, half_w, -half_h, half_h, 0.1f, 100.0f);
		}
		camera.camera_position[0] = camera_eye.x;
		camera.camera_position[1] = camera_eye.y;
		camera.camera_position[2] = camera_eye.z;
		for (int i = 0; i < 3; ++i)
		{
			camera.ambient[i] = ambient_color[i];
			camera.directional_color[i] = directional_color[i];
		}
		camera.ambient[3] = ambient_intensity;
		camera.directional_direction[0] = directional_direction.x;
		camera.directional_direction[1] = directional_direction.y;
		camera.directional_direction[2] = directional_direction.z;
		camera.directional_color[3] = directional_intensity;
		camera.light_counts[0] = static_cast<uint32_t>(point_lights.size());
		camera.light_counts[1] = static_cast<uint32_t>(spot_lights.size());
		write_ubo(camera_ubo, &camera, sizeof(SceneUBO));

		ensure_light_buffers();
		std::vector<PointLightGPU> point_gpu(point_lights.size());
		for (size_t i = 0; i < point_lights.size(); ++i)
		{
			const auto &src = point_lights[i]; auto &dst = point_gpu[i];
			dst.position[0] = src.position.x; dst.position[1] = src.position.y; dst.position[2] = src.position.z; dst.position[3] = 1.0f;
			for (int c = 0; c < 3; ++c) dst.color_intensity[c] = src.color[c];
			dst.color_intensity[3] = src.intensity;
		}
		std::vector<SpotLightGPU> spot_gpu(spot_lights.size());
		for (size_t i = 0; i < spot_lights.size(); ++i)
		{
			const auto &src = spot_lights[i]; auto &dst = spot_gpu[i];
			dst.position[0] = src.position.x; dst.position[1] = src.position.y; dst.position[2] = src.position.z; dst.position[3] = 1.0f;
			const Vec3 direction = normalize(src.direction);
			dst.direction[0] = direction.x; dst.direction[1] = direction.y; dst.direction[2] = direction.z; dst.direction[3] = 0.0f;
			for (int c = 0; c < 3; ++c) dst.color_intensity[c] = src.color[c];
			dst.color_intensity[3] = src.intensity;
			dst.cone_angles[0] = std::cos(radians(src.inner_angle));
			dst.cone_angles[1] = std::cos(radians(src.outer_angle));
		}
		if (!point_gpu.empty()) write_ubo(point_light_ssbo, point_gpu.data(), point_gpu.size() * sizeof(PointLightGPU));
		if (!spot_gpu.empty()) write_ubo(spot_light_ssbo, spot_gpu.data(), spot_gpu.size() * sizeof(SpotLightGPU));

		// Обновление модельной матрицы и цвета каждого объекта.
		for (Object &obj : objects)
		{
			ModelUBO ubo{};
			ubo.model = obj.model_matrix();
			for (int c = 0; c < 3; ++c)
			{
				ubo.albedo[c] = obj.albedo[c];
				ubo.specular_and_shininess[c] = obj.specular_color[c];
			}
			ubo.albedo[3] = 1.0f;
			ubo.specular_and_shininess[3] = obj.shininess;
			write_ubo(obj.ubo, &ubo, sizeof(ModelUBO));
		}

		// Запись команд: очистка кадра, render pass, pipeline и draw-вызовы.
		VkCommandBuffer cb = fd.command_buffer;
		vkResetCommandBuffer(cb, 0);

		VkCommandBufferBeginInfo begin = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
		};
		vkBeginCommandBuffer(cb, &begin);

		VkClearValue clear_values[2] = {
			{.color = {{0.08f, 0.08f, 0.10f, 1.0f}}},
			{.depthStencil = {1.0f, 0}},
		};
		VkRenderPassBeginInfo render_pass_begin = {
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
			.renderPass = context.render_pass,
			.framebuffer = fd.framebuffer,
			.renderArea = {{0, 0}, context.swapchain_extent},
			.clearValueCount = 2,
			.pClearValues = clear_values,
		};
		vkCmdBeginRenderPass(cb, &render_pass_begin, VK_SUBPASS_CONTENTS_INLINE);

		vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

		VkViewport viewport = {
			.x = 0.0f,
			.y = 0.0f,
			.width = static_cast<float>(context.swapchain_extent.width),
			.height = static_cast<float>(context.swapchain_extent.height),
			.minDepth = 0.0f,
			.maxDepth = 1.0f,
		};
		vkCmdSetViewport(cb, 0, 1, &viewport);

		VkRect2D scissor = {
			.offset = {0, 0},
			.extent = context.swapchain_extent,
		};
		vkCmdSetScissor(cb, 0, 1, &scissor);

		// Камера общая для всех объектов (набор дескрипторов 0).
		vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
								0, 1, &camera_set, 0, nullptr);

		// Каждый объект использует собственный набор дескрипторов (набор дескрипторов 1).
		for (const Object &obj : objects)
		{
			if (!obj.visible)
			{
				continue;
			}
			vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
									1, 1, &obj.descriptor_set, 0, nullptr);
			VkDeviceSize offset = 0;
			vkCmdBindVertexBuffers(cb, 0, 1, &obj.mesh.buffer, &offset);
			vkCmdDraw(cb, obj.mesh.vertex_count, 1, 0, 0);
		}

		vkCmdEndRenderPass(cb);
		vkEndCommandBuffer(cb);
	}

} // пространство имён application
