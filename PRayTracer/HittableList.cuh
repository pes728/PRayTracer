#pragma once
#include <cuda_runtime.h>
#include "Hittable.cuh"

class HittableList : public Hittable {
public:
    __device__ HittableList() {}
    __device__ HittableList(Hittable** l, int n) { list = l; list_size = n; }
    virtual __device__ bool hit(const Ray& r, float tmin, float tmax, HitRecord& rec) const;

    Hittable** list;
    int list_size;
};


