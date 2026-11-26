"""HIP smoke test for AMD Vega 7 (gfx90c / gfx900 target).

On the tested APU (Radeon Graphics, gfx90c, XNACK enabled) ROCm 7.16 aborts
when loading a gfx90c code object.  The working configuration is the classic
override:

    HSA_OVERRIDE_GFX_VERSION=9.0.0 hipcc --offload-arch=gfx900

The script applies that override by default; use ``--hsa-override ""`` to test
the native target instead.

This tests the device path independently of the training model.  It fails
clearly when ROCm or the target GPU is unavailable instead of silently
claiming that CPU execution is GPU execution.

ADR: docs/adr/0012-trainable-graph-contour.md, docs/adr/0014-build-portability-and-hip.md
"""

from __future__ import annotations

import argparse
import ctypes
import os
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
  hipError_t err = hipMalloc(&d_in, n * sizeof(float));
  if (err != hipSuccess) return 1;
  err = hipMalloc(&d_out, n * sizeof(float));
  if (err != hipSuccess) { (void)hipFree(d_in); return 2; }
  err = hipMemcpy(d_in, input, n * sizeof(float), hipMemcpyHostToDevice);
  if (err != hipSuccess) { (void)hipFree(d_in); (void)hipFree(d_out); return 3; }
  hipLaunchKernelGGL(add_one, dim3((n + 255) / 256), dim3(256), 0, 0, d_in, d_out, n);
  err = hipDeviceSynchronize();
  if (err == hipSuccess) {
    err = hipMemcpy(output, d_out, n * sizeof(float), hipMemcpyDeviceToHost);
  }
  (void)hipFree(d_in);
  (void)hipFree(d_out);
  return err == hipSuccess ? 0 : 4;
}
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="gfx900")
    parser.add_argument("--hsa-override", default="9.0.0",
                        help="HSA_OVERRIDE_GFX_VERSION value; empty string disables the override")
    parser.add_argument("--hipcc", default="/opt/rocm/bin/hipcc")
    parser.add_argument("--elements", type=int, default=1024)
    args = parser.parse_args()

    if args.hsa_override:
        os.environ["HSA_OVERRIDE_GFX_VERSION"] = args.hsa_override

    hipcc = pathlib.Path(args.hipcc)
    if not hipcc.exists():
        raise SystemExit(f"hipcc not found at {hipcc}; install ROCm or pass --hipcc")

    with tempfile.TemporaryDirectory() as tmp:
        src = pathlib.Path(tmp) / "smoke.cpp"
        lib = pathlib.Path(tmp) / "libsmoke.so"
        src.write_text(KERNEL)
        compile_command = [str(hipcc), "--offload-arch=" + args.target, "-shared", "-fPIC",
                           str(src), "-o", str(lib)]
        subprocess.run(compile_command, check=True)

        try:
            function = ctypes.CDLL(str(lib)).run_add_one
        except OSError as exc:
            raise SystemExit(
                "HIP library compiled but could not load on the active GPU. "
                "Check ROCm XNACK compatibility and the gfx target. "
                f"Original error: {exc}"
            ) from exc

        function.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        function.restype = ctypes.c_int

        n = args.elements
        input_data = (ctypes.c_float * n)(*range(n))
        output_data = (ctypes.c_float * n)()
        try:
            code = function(input_data, output_data, n)
        except OSError as exc:
            raise SystemExit(
                "HIP kernel dispatch failed. The compiler target was accepted, "
                "but the installed ROCm runtime rejected execution. "
                f"Original error: {exc}"
            ) from exc

        if code:
            raise SystemExit(f"HIP smoke test failed with code {code}")
        if any(output_data[i] != i + 1 for i in range(n)):
            raise SystemExit("HIP smoke test returned invalid data")
        print(f"HIP smoke test passed for {args.target}: {n} elements, "
              f"HSA_OVERRIDE_GFX_VERSION={args.hsa_override or '<unset>'}")


if __name__ == "__main__":
    main()
