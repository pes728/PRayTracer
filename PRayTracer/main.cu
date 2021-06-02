#include "Renderer.cuh"
#include "kernel.cuh"
#include <time.h>

int main(int argc, char** argv) {
    Window* window = new Window("Peter's Window", false, 1920, 1080);
    Renderer renderer;

    renderer.init(window, 1);
    renderer.setupScene();

    clock_t start, stop;
    while (!glfwWindowShouldClose(renderer.window->windowHandle))
    {
        start = clock();
        renderer.render(launchRender);
        stop = clock();
        double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
        std::cerr << "took " << timer_seconds << " seconds.\n";
    }

    renderer.cleanup();
}

