#include "kernel.cuh"

extern "C" void launchRender(dim3 blocks, dim3 threads, cudaSurfaceObject_t surface, int width, int height, int samples, Camera** camera, Hittable** world, float dt, curandState* d_randState) {
	render <<<blocks, threads>>> (surface, width, height, samples, camera, world, dt, d_randState);
}

__global__ void render(cudaSurfaceObject_t surface, int width, int height, int samples, Camera** camera, Hittable** world, float dt, curandState* randState) {
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	if ((x >= width) || (y >= height)) return;
	int pixel_index = y * width + x;
	curandState local_rand_state = randState[pixel_index];
	Vec3 col(0, 0, 0);
	for (int s = 0; s < samples; s++) {
		float u = float(x + curand_uniform(&local_rand_state)) / float(width);
		float v = float(y + curand_uniform(&local_rand_state)) / float(height);
		Ray r = (*camera)->get_ray(u, v);
		
		col += color(r, world);
	}

	col /= float(samples);

	uchar4 data = make_uchar4(col[0], col[1], col[2], 255);
	surf2Dwrite(data, surface, x * (int)sizeof(uchar4), y, cudaBoundaryModeClamp);
}

__device__ Vec3 color(const Ray& r, Hittable** world) {
	HitRecord rec;
	if ((*world)->hit(r, 0.0, FLT_MAX, rec)) {
		return 0.5f * Vec3(rec.normal.x() + 1.0f, rec.normal.y() + 1.0f, rec.normal.z() + 1.0f);
	}
	else {
		Vec3 unit_direction = unit_vector(r.direction());
		float t = 0.5f * (unit_direction.y() + 1.0f);
		return (1.0f - t) * Vec3(1.0, 1.0, 1.0) + t * Vec3(0.5, 0.7, 1.0);
	}
}