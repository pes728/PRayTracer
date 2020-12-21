#pragma once
#include "Window.h"
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>

#include "HittableList.h"
#include "Sphere.h"
#include "Camera.h"

#include "checkCuda.h"

__global__ void initRand(int width, int height, curandState* randState);
__global__ void createWorld(Hittable** d_list, Hittable** d_world, Camera** d_camera);
__global__ void freeWorld(Hittable** d_list, Hittable** d_world, Camera** d_camera);

extern "C" void callInitRand(dim3 block, dim3 threads, int width, int height, curandState * randState);
extern "C" void callCreateWorld(Hittable** d_list, Hittable** d_world, Camera** d_camera);
extern "C" void callFreeWorld(Hittable** d_list, Hittable** d_world, Camera** d_camera);

class Renderer {
public:
	Window* window;
	int samples;
	curandState* d_randState;
	dim3 blocks, threads;
	float lastTime = 0;

	Hittable** d_list;
	Hittable** d_world;
	Camera** d_camera;
	
	void init(Window* window, int samples = 10, int threadX = 16, int threadY = 16);
	void setupScene();
	void render(void (*renderFunc)(dim3 blocks, dim3 threads, cudaSurfaceObject_t surface, int width, int height, int samples, Camera** camera, Hittable** world, float dt, curandState* randState));
	void cleanup();
};