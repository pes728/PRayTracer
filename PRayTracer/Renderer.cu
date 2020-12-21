#include "Renderer.cuh"

extern "C" void callInitRand(dim3 blocks, dim3 threads, int width, int height, curandState* randState) {
    initRand <<<blocks, threads>>> (width, height, randState);
}

extern "C" void callCreateWorld(Hittable** d_list, Hittable** d_world, Camera** d_camera) {
    createWorld <<<1, 1>>>(d_list, d_world, d_camera);
}

extern "C" void callFreeWorld(Hittable** d_list, Hittable** d_world, Camera** d_camera) {
    freeWorld <<<1, 1>>>(d_list, d_world, d_camera);
}


__global__ void initRand(int width, int height, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= width) || (j >= height)) return;
    int pixel_index = j * width + i;
    //Each thread gets same seed, a different sequence number, no offset
    curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void createWorld(Hittable** d_list, Hittable** d_world, Camera** d_camera) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *(d_list) = new Sphere(Vec3(0, 0, -1), 0.5);
        *(d_list + 1) = new Sphere(Vec3(0, -100.5, -1), 100);
        *d_world = new HittableList(d_list, 2);
        *d_camera = new Camera();
    }
}

__global__ void freeWorld(Hittable** d_list, Hittable** d_world, Camera** d_camera) {
    delete* (d_list);
    delete* (d_list + 1);
    delete* d_world;
    delete* d_camera;
}

void Renderer::render(void (*renderFunc)(dim3 blocks, dim3 threads, cudaSurfaceObject_t surface, int width, int height, int samples, Camera** camera, Hittable** world, float dt, curandState* randState)) {
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    cudaGraphicsMapResources(1, &window->textureCudaResource);
    {
        cudaArray_t viewCudaArray;
        cudaGraphicsSubResourceGetMappedArray(&viewCudaArray, window->textureCudaResource, 0, 0);
        cudaResourceDesc viewCudaArrayResourceDesc;
        {
            viewCudaArrayResourceDesc.resType = cudaResourceTypeArray;
            viewCudaArrayResourceDesc.res.array.array = viewCudaArray;
        }
        cudaSurfaceObject_t viewCudaSurfaceObject;
        cudaCreateSurfaceObject(&viewCudaSurfaceObject, &viewCudaArrayResourceDesc);
        {
            renderFunc(blocks, threads, viewCudaSurfaceObject, window->width, window->height, samples, d_camera, d_world, (float)glfwGetTime() - lastTime, d_randState);
        }
        cudaDestroySurfaceObject(viewCudaSurfaceObject);
    }
    cudaGraphicsUnmapResources(1, &window->textureCudaResource);

    cudaStreamSynchronize(0);

    /* Render here */


    glBlitFramebuffer(0, 0, window->width, window->height, 0, 0, window->width, window->height, GL_COLOR_BUFFER_BIT, GL_NEAREST);

    glfwSwapBuffers(window->windowHandle);

    /* Poll for and process events */
    glfwPollEvents();
}


void Renderer::init(Window* window, int samples, int threadX, int threadY)
{
    this->window = window;
    blocks = dim3(window->width / threadX + 1, window->height / threadY + 1);
    threads = dim3(threadX, threadY);
    this->samples = samples;
}

void Renderer::setupScene() {
    checkCudaErrors(cudaMalloc((void**)&d_randState, window->width * window->height * sizeof(curandState)));

    callInitRand(blocks, threads, window->width, window->height, d_randState);
    checkCudaErrors(cudaMalloc((void**)&d_list, 2 * sizeof(Hittable*)));
    checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(Hittable*)));
    checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(Camera*)));
    callCreateWorld(d_list, d_world, d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

void Renderer::cleanup() {
    checkCudaErrors(cudaDeviceSynchronize());
    callFreeWorld(d_list, d_world, d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_randState));
    window->cleanup();

    cudaDeviceReset();
}

