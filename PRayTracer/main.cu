#include "Renderer.cuh"
#include "kernel.cuh"

int main(int argc, char** argv) {
    Window* window = new Window("Peter's Window");
    Renderer renderer;

    renderer.init(window);
    renderer.setupScene();

    while (!glfwWindowShouldClose(renderer.window->windowHandle))
    {
        renderer.render(launchRender);
    }

    renderer.cleanup();
}