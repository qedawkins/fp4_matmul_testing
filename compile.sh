
iree-compile only_matmul.mlir \
    --iree-hip-target=gfx950 \
    --iree-hal-target-backends=rocm \
    --mlir-disable-threading \
    --iree-codegen-enable-default-tuning-specs=true \
    -o tmp/matmul.vmfb

iree-compile calls.mlir \
    --iree-hip-target=gfx950 \
    --iree-hal-target-backends=rocm \
    --mlir-disable-threading \
    --iree-codegen-enable-default-tuning-specs=true \
    -o tmp/calls.vmfb

iree-compile conversions.mlir \
    --iree-hip-target=gfx950 \
    --iree-hal-target-backends=rocm \
    --mlir-disable-threading \
    --iree-codegen-enable-default-tuning-specs=true \
    --iree-llvmgpu-set-workgroup-distribution-along=x \
    -o tmp/conversions.vmfb

iree-e2e-matmul-test --device=hip \
  --module=tmp/matmul.vmfb \
  --module=tmp/conversions.vmfb \
  --module=tmp/calls.vmfb \
  --require_exact_results=true \
  --acceptable_fp_delta=1e-04 \
  --max_elements_to_check=0
