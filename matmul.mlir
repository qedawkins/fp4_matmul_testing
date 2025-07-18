// Quantized mxfp4 example with:
// M = 1024
// N = 1024
// Block size = 32

!lhs = f4E2M1FN
!rhs = f4E2M1FN
!out = f4E2M1FN

!scale_ty = f8E8M0FNU

!Bin_size = tensor<1024x32x16xi8>
!Ain_size = tensor<1024x32x16xi8>
!out_cast_size = tensor<1024x32x16xi8>

// Input types.  M    Ko Kb
!A_size = tensor<1024x32x32x!lhs>
//               N    Ko Kb
!B_size = tensor<1024x32x32x!rhs>
//               M    N
!C_size = tensor<1024x1024xf32>
// Expand N for scaling for the next matmul.
!out_C_size = tensor<1024x32x32xf32>
// Truncate to the quantized type.
!out_size = tensor<1024x32x32x!out>

// Scale types.    M    K / 32 = Kb
!A_scales = tensor<1024x32x!scale_ty>
//                 N    K / 32 = Kb
!B_scales = tensor<1024x32x!scale_ty>
//                 M    N / 32 = Next matmul Kb
!C_scales = tensor<1024x32x!scale_ty>

// 4 loops in this example:
// M/N = typical M/N dimensions
// K broken up into two dimensions, Ko[uter] and Kb[lock]
// As many M/N/K dimensions as needed are allowed with two requirements:
//  1. There is always at least one distinct Kblock dimension that is NOT
//  present in EITHER scales input.
//  2. The Kblock dimension is at least the per-thread vector width of the
//  target instruction (typically 32).
//
//  We could eventually support Kblock sizes misaligned to the vector width,
//  however it's unlikely to ever be good. (support meaning fast path).

// Indexing maps for LHS and RHS. Expected to restrict to transpose B for
// initial bringup to avoid difficulties with sub-byte loads.
// i.e. vectorization is a * requirement *.
#lhs_map = affine_map<(M, N, Ko, Kb) -> (M, Ko, Kb)>
#rhs_map = affine_map<(M, N, Ko, Kb) -> (N, Ko, Kb)>
#scale_m = affine_map<(M, N, Ko, Kb) -> (M, Ko)>
#scale_n = affine_map<(M, N, Ko, Kb) -> (N, Ko)>
#out_map = affine_map<(M, N, Ko, Kb) -> (M, N)>
func.func @matmul(
    %Ain: !Ain_size, %Bin: !Bin_size, %A_scales: !A_scales, %B_scales: !B_scales, %C_scales: !C_scales) -> !out_cast_size {
  %A = iree_tensor_ext.bitcast %Ain : !Ain_size -> !A_size
  %B = iree_tensor_ext.bitcast %Bin : !Bin_size -> !B_size
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : !C_size
  %C = linalg.fill ins(%cst : f32) outs(%empty : !C_size) -> !C_size
  %0 = linalg.generic {
    indexing_maps = [#lhs_map, #rhs_map, #scale_m, #scale_n, #out_map],
    iterator_types = ["parallel", "parallel", "reduction", "reduction"]
  } ins(%A, %B, %A_scales, %B_scales : !A_size, !B_size, !A_scales, !B_scales) outs(%C : !C_size) {
  ^bb0(%a: !lhs, %b: !rhs, %a_scale: !scale_ty, %b_scale: !scale_ty, %out: f32):
    %1 = arith.scaling_extf %a, %a_scale : !lhs, !scale_ty to f32
    %2 = arith.scaling_extf %b, %b_scale : !rhs, !scale_ty to f32
    %3 = arith.mulf %1, %2 : f32
    %4 = arith.addf %out, %3 : f32
    linalg.yield %4 : f32
  } -> !C_size
  %expand = tensor.expand_shape %0 [[0], [1, 2]] output_shape [1024, 32, 32] : !C_size into !out_C_size
  %E = tensor.empty() : !out_size
  %1 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1)>,
      affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]}
  ins(%expand, %C_scales : !out_C_size, !C_scales) outs(%E : !out_size) {
  ^bb0(%in: f32, %c_scale: !scale_ty, %out: !out):
    // Expect more complicated quantization logic here. For the first example
    // just truncate.
    %trunc = arith.scaling_truncf %in, %c_scale : f32, !scale_ty to !out
    linalg.yield %trunc : !out
  } -> !out_size
  %2 = iree_tensor_ext.bitcast %1 : !out_size -> !out_cast_size
  return %2 : !out_cast_size
}
