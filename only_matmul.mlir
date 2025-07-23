// Quantized mxfp4 example with:
// M = 64
// N = 53248
// K = 16384
// Block size = 32

!lhs = f4E2M1FN
!rhs = f4E2M1FN
!out = f4E2M1FN

!scale_ty = f8E8M0FNU

!Ain_size = tensor<64x8192xi8>
!Bin_size = tensor<53248x8192xi8>

// Input types.  M    Ko Kb
!A_size = tensor<64x512x32x!lhs>
//               N    Ko Kb
!B_size = tensor<53248x512x32x!rhs>
//               M    N
!C_size = tensor<64x53248xf32>

// Scale types.    M    K / 32 = Kb
!A_scales = tensor<64x512x!scale_ty>
!Ain_scales = tensor<64x512xi8>
//                 N    K / 32 = Kb
!B_scales = tensor<53248x512x!scale_ty>
!Bin_scales = tensor<53248x512xi8>

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
    %Ain: !Ain_size, %Bin: !Bin_size, %Ain_scales: !Ain_scales, %Bin_scales: !Bin_scales) -> !C_size {
  %A = iree_tensor_ext.bitcast %Ain : !Ain_size -> !A_size
  %B = iree_tensor_ext.bitcast %Bin : !Bin_size -> !B_size
  %A_scales = iree_tensor_ext.bitcast %Ain_scales : !Ain_scales -> !A_scales
  %B_scales = iree_tensor_ext.bitcast %Bin_scales : !Bin_scales -> !B_scales
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
  return %0 : !C_size
}
