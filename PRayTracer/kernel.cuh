#pragma once
#include "Renderer.cuh"


extern "C" void launchRender(dim3 blocks, dim3 threads, cudaSurfaceObject_t surface, int width, int height, int samples, Camera** camera, Hittable** world, float dt, curandState* d_randState);

__global__ void render(cudaSurfaceObject_t surface, int width, int height, int samples, Camera** camera, Hittable** world, float dt, curandState* randState);

__device__ Vec3 color(const Ray& ray, Hittable** world);