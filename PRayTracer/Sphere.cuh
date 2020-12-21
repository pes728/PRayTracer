#pragma once
#include <cuda_runtime.h>
#include "Hittable.cuh"

class Sphere : public Hittable {
public:
    __device__ Sphere() {}
    __device__ Sphere(Vec3 cen, float r) : center(cen), radius(r) {}
    __device__ bool hit(const Ray& r, float tmin, float tmax, HitRecord& rec) const;

    Vec3 center;
    float radius;
};