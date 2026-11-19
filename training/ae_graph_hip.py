"""Optional HIP smoke test for AMD Vega 7 (gfx90c).

This deliberately tests the device path independently of the training model.
It fails clearly when ROCm or the target GPU is unavailable instead of silently
claiming that CPU execution is GPU training.
"""

from __future__ import annotations

import argparse
import ctypes
import pathlib
import subprocess
import tempfile


KERNEL = r'''
#include <hip/hip_runtime.h>
extern "C" __global__ void add_one(const float* input, float* output, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) output[i] = input[i] + 1.0f;
}
extern "C" int run_add_one(float* input, float* output, int n) {
  float *d_in = nullptr, *d_out = nullptr;
  if (hipMalloc(&d_in, n*sizeof(float)) != hipSuccess) return 1;
  if (hipMalloc(&d_out, n*sizeof(float)) != hipSuccess) return 2;
  hipMemcpy(d_in, input, n*sizeof(float), hipMemcpyHostToDevice);
  hipLaunchKernelGGL(add_one, dim3((n+255)/256), dim3(256), 0, 0, d_in, d_out, n);
  hipError_t err = hipDeviceSynchronize();
  if (err == hipSuccess) err = hipMemcpy(output, d_out, n*sizeof(float), hipMemcpyDeviceToHost);
  hipFree(d_in); hipFree(d_out);
  return err == hipSuccess ? 0 : 3;
}
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="gfx90c:xnack-")
    args = parser.parse_args()
    hipcc = pathlib.Path("/opt/rocm/bin/hipcc")
    if not hipcc.exists():
        raise SystemExit("hipcc not found; install ROCm or set the HIP compiler path")
    with tempfile.TemporaryDirectory() as tmp:
        src = pathlib.Path(tmp) / "smoke.cpp"
        lib = pathlib.Path(tmp) / "libsmoke.so"
        src.write_text(KERNEL)
        subprocess.run([str(hipcc), "--offload-arch=" + args.target, "-shared", "-fPIC", str(src), "-o", str(lib)], check=True)
        try:
            fn = ctypes.CDLL(str(lib)).run_add_one
        except OSError as exc:
            raise SystemExit(
                "HIP library compiled but could not load on the active GPU. "
                "Check ROCm XNACK compatibility and the gfx target. "
                f"Original error: {exc}"
            ) from exc
        fn.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        fn.restype = ctypes.c_int
        n = 1024
        input_data = (ctypes.c_float * n)(*range(n))
        output_data = (ctypes.c_float * n)()
        try:
            rc = fn(input_data, output_data, n)
        except OSError as exc:
            raise SystemExit(
                "HIP kernel dispatch failed. The compiler target was accepted, "
                "but the installed ROCm runtime rejected execution. "
                f"Original error: {exc}"
            ) from exc
        if rc:
            raise SystemExit(f"HIP smoke test failed with code {rc}")
        if any(output_data[i] != i + 1 for i in range(n)):
            raise SystemExit("HIP smoke test returned invalid data")
        print(f"HIP smoke test passed for {args.target}: {n} elements")


if __name__ == "__main__":
    main()
