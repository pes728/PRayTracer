#pragma once
#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <curand_kernel.h>

#include <stdlib.h>

#include <iostream>

class Window {
public:
	bool isFullscreen;
	int width, height;
	GLFWwindow* windowHandle;
	unsigned int texture;
	unsigned int framebuffer;
	cudaGraphicsResource_t textureCudaResource;

	Window(const char* windowName, bool isFullscreen = false, int width = 1200, int height = 600);
	void createWindow(const char* windowName);
	void linkKeyIn(void(*keyIn)(GLFWwindow* window, int key, int scancode, int action, int mods));
	void cleanup();
private:
	void initOpenGL();
};

void toggleFullscreen(GLFWwindow* window);

void keyIn(GLFWwindow* window, int key, int scancode, int action, int mods);