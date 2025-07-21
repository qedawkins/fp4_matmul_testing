
builtin.module @calls attributes {
  
} {

func.func private @matmul_test.generate_random_matrix(%device: !hal.device, %dim0: i64, %dim1: i64, %element_type: i32, %seed: i32) -> !hal.buffer_view
func.func private @matmul_test.check_matmul_results(%device: !hal.device, %m: i64, %k: i64, %n: i64, %transpose_rhs: i32, %lhs: !hal.buffer_view, %rhs: !hal.buffer_view, %acc: !hal.buffer_view, %actual_result: !hal.buffer_view)

func.func private @conversions.scaled_f4_to_f32(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %block_size: i64) -> !hal.buffer_view
func.func private @conversions.avoid_nan_scale(%arg0: !hal.buffer_view) -> !hal.buffer_view

func.func private @module.matmul(%lhs: !hal.buffer_view, %rhs: !hal.buffer_view, %lhs_scales: !hal.buffer_view, %rhs_scales: !hal.buffer_view) -> !hal.buffer_view

func.func @matmul() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 1024x1024x1024"}
} {
  %m = arith.constant 1024 : i64
  %k = arith.constant 1024 : i64
  %n = arith.constant 1024 : i64
  %k_outer = arith.constant 32 : i64
  %k_packed = arith.constant 512 : i64
  %block_size = arith.constant 32 : i64
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device

  %byte_type = hal.element_type<i8> : i32
  %lhs_seed = arith.constant 5 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %m, %k_packed, %byte_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view

  %lhs_scale_seed = arith.constant 6 : i32
  %lhs_scale = call @matmul_test.generate_random_matrix(%device, %m, %k_outer, %byte_type, %lhs_scale_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %lhs_scale_fixed = call @conversions.avoid_nan_scale(%lhs_scale) : (!hal.buffer_view) -> !hal.buffer_view

  %rhs_seed = arith.constant 7 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %n, %k_packed, %byte_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view

  %rhs_scale_seed = arith.constant 8 : i32
  %rhs_scale = call @matmul_test.generate_random_matrix(%device, %n, %k_outer, %byte_type, %rhs_scale_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_scale_fixed = call @conversions.avoid_nan_scale(%rhs_scale) : (!hal.buffer_view) -> !hal.buffer_view

  %result = call @module.matmul(%lhs, %rhs, %lhs_scale_fixed, %rhs_scale_fixed) : (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view

  %lhs_f32 = call @conversions.scaled_f4_to_f32(%lhs, %lhs_scale_fixed, %block_size) : (!hal.buffer_view, !hal.buffer_view, i64) -> !hal.buffer_view
  %rhs_f32 = call @conversions.scaled_f4_to_f32(%rhs, %rhs_scale_fixed, %block_size) : (!hal.buffer_view, !hal.buffer_view, i64) -> !hal.buffer_view

  %transpose_rhs = arith.constant 1 : i32
  %acc = util.null : !hal.buffer_view
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs_f32, %rhs_f32, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}

}
