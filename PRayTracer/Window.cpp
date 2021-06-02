#include "Window.h"

Window::Window(const char* windowName, bool isFullscreen, int width, int height)
{
    this->width = width;
    this->height = height;
    this->isFullscreen = isFullscreen;
    createWindow(windowName);
}

void Window::createWindow(const char* windowName)
{
    if (!glfwInit())
        exit(-1);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWmonitor* monitor = glfwGetPrimaryMonitor();

    const GLFWvidmode* mode = glfwGetVideoMode(monitor);

    if (isFullscreen) {
        windowHandle = glfwCreateWindow(width, height, windowName, monitor, NULL);
    }
    else {
        windowHandle = glfwCreateWindow(width, height, windowName, NULL, NULL);
    }

    if (!windowHandle)
    {
        glfwTerminate();
        exit(-1);
    }

    glfwMakeContextCurrent(windowHandle);

    initOpenGL();
}



void Window::initOpenGL()
{
    if (glewInit() != GLEW_OK)
        std::cout << "ERROR!" << std::endl;

    std::cout << glGetString(GL_VERSION) << std::endl;



    //setup texture
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);

    //setup framebuffer
    glGenFramebuffers(1, &framebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);


    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete!" << std::endl;

    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);

    cudaGraphicsGLRegisterImage(&textureCudaResource, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);

}

void Window::linkKeyIn(void(*keyIn)(GLFWwindow* window, int key, int scancode, int action, int mods))
{
    glfwSetKeyCallback(windowHandle, keyIn);
}

void Window::cleanup()
{
    glDeleteTextures(1, &texture);
    glDeleteFramebuffers(1, &framebuffer);

    glfwDestroyWindow(windowHandle);
}

void keyIn(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (GLFW_PRESS == glfwGetKey(window, GLFW_KEY_F11)) {
        toggleFullscreen(window);
    }
}
