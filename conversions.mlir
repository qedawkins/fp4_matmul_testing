builtin.module @conversions {

util.func private @scaled_f4_to_f32_impl(%arg0: tensor<?x?x?xi8>, %arg1: tensor<?x?xi8>) -> tensor<?x?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %o = tensor.dim %arg0, %c0 : tensor<?x?x?xi8>
  %i = tensor.dim %arg0, %c1 : tensor<?x?x?xi8>
  %b = tensor.dim %arg0, %c2 : tensor<?x?x?xi8>

  %empty = tensor.empty(%o, %i, %b) : tensor<?x?x?xi64>
  %cvt = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1)>,
      affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%arg0, %arg1 : tensor<?x?x?xi8>, tensor<?x?xi8>) outs(%empty : tensor<?x?x?xi64>) {
  ^bb0(%in: i8, %scale: i8, %out: i64):
    %c4_i8 = arith.constant 4 : i8
    %c127_i8 = arith.constant 127 : i8
    %c2_f32 = arith.constant 2.0 : f32
    %c1_f32 = arith.constant 1.0 : f32
    %cm1_f32 = arith.constant -1.0 : f32

    %c0_i4 = arith.constant 0 : i4
    %c1_i4 = arith.constant 1 : i4
    %c2_i4 = arith.constant 2 : i4
    %s_mask = arith.constant 8 : i4
    %e_mask = arith.constant 6 : i4
    %m_mask = arith.constant 1 : i4

    %base = arith.subi %scale, %c127_i8 : i8
    %ext = arith.extsi %base : i8 to i32
    %scale_pow = math.fpowi %c2_f32, %ext : f32, i32
    %cFF = arith.constant 0xFF : i8
    %cnan_f32 = arith.constant 0x7F800001 : f32
    %sff = arith.cmpi eq, %scale, %cFF : i8
    %scale_f32 = arith.select %sff, %cnan_f32, %scale_pow : f32

    %b0 = arith.trunci %in : i8 to i4
    %s0 = arith.andi %b0, %s_mask : i4
    %e0 = arith.andi %b0, %e_mask : i4
    %m0 = arith.andi %b0, %m_mask : i4
    %es0 = arith.shrui %e0, %c1_i4 : i4

    %m0_p2 = arith.addi %m0, %c2_i4 : i4
    %ze0 = arith.cmpi eq, %e0, %c0_i4 : i4
    %fm0 = arith.select %ze0, %m0, %m0_p2 : i4
    %fm0_i32 = arith.extui %fm0 : i4 to i32
    %fm0_f32 = arith.uitofp %fm0_i32 : i32 to f32

    %eb0 = arith.select %ze0, %c1_i4, %es0 : i4
    %fe0 = arith.subi %eb0, %c2_i4 : i4
    %fe0_i32 = arith.extsi %fe0 : i4 to i32

    %fb0 = math.fpowi %c2_f32, %fe0_i32 : f32, i32
    %uf0 = arith.mulf %fm0_f32, %fb0 : f32

    %sc0 = arith.cmpi eq, %s0, %c0_i4 : i4
    %fs0 = arith.select %sc0, %c1_f32, %cm1_f32 : f32
    %f0 = arith.mulf %uf0, %fs0 : f32

    %shift = arith.shrui %in, %c4_i8 : i8
    %b1 = arith.trunci %shift : i8 to i4
    %s1 = arith.andi %b1, %s_mask : i4
    %e1 = arith.andi %b1, %e_mask : i4
    %m1 = arith.andi %b1, %m_mask : i4
    %es1 = arith.shrui %e1, %c1_i4 : i4

    %m1_p2 = arith.addi %m1, %c2_i4 : i4
    %ze1 = arith.cmpi eq, %e1, %c0_i4 : i4
    %fm1 = arith.select %ze1, %m1, %m1_p2 : i4
    %fm1_i32 = arith.extui %fm1 : i4 to i32
    %fm1_f32 = arith.uitofp %fm1_i32 : i32 to f32

    %eb1 = arith.select %ze1, %c1_i4, %es1 : i4
    %fe1 = arith.subi %eb1, %c2_i4 : i4
    %fe1_i32 = arith.extsi %fe1 : i4 to i32

    %fb1 = math.fpowi %c2_f32, %fe1_i32 : f32, i32
    %uf1 = arith.mulf %fm1_f32, %fb1 : f32

    %sc1 = arith.cmpi eq, %s1, %c0_i4 : i4
    %fs1 = arith.select %sc1, %c1_f32, %cm1_f32 : f32
    %f1 = arith.mulf %uf1, %fs1 : f32

    %float0 = arith.mulf %f0, %scale_f32 : f32
    %float1 = arith.mulf %f1, %scale_f32 : f32

    %lower = arith.bitcast %float0 : f32 to i32
    %lower_ext = arith.extui %lower : i32 to i64

    %c32_i64 = arith.constant 32 : i64
    %upper = arith.bitcast %float1 : f32 to i32
    %upper_ext = arith.extui %upper : i32 to i64
    %upper_shift = arith.shli %upper_ext, %c32_i64 : i64

    %c0_i64 = arith.constant 0 : i64
    %o0 = arith.ori %c0_i64, %lower_ext : i64
    %o1 = arith.ori %o0, %upper_shift : i64

    linalg.yield %o1 : i64
  } -> tensor<?x?x?xi64>

  %b2 = arith.muli %b, %c2 : index
  %unpacked = iree_tensor_ext.bitcast %cvt : tensor<?x?x?xi64>{%o, %i, %b} -> tensor<?x?x?xf32>{%o, %i, %b2}
  util.return %unpacked : tensor<?x?x?xf32>
}

util.func public @scaled_f4_to_f32(%arg0: tensor<?x?xi8>, %arg1: tensor<?x?xi8>, %block_size: i64) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %o = tensor.dim %arg0, %c0 : tensor<?x?xi8>
  %i = tensor.dim %arg0, %c1 : tensor<?x?xi8>
  %block_index = arith.index_cast %block_size : i64 to index
  %block_over_2 = arith.divui %block_index, %c2 : index
  %i_div = arith.divui %i, %block_over_2 : index
  %unblocked = flow.tensor.reshape %arg0 : tensor<?x?xi8>{%o, %i} -> tensor<?x?x?xi8>{%o, %i_div, %block_over_2}
  %0 = util.call @scaled_f4_to_f32_impl(%unblocked, %arg1) : (tensor<?x?x?xi8>, tensor<?x?xi8>) -> tensor<?x?x?xf32>
  %i_times_2 = arith.muli %i, %c2 : index
  %blocked = flow.tensor.reshape %0 : tensor<?x?x?xf32>{%o, %i_div, %block_index} -> tensor<?x?xf32>{%o, %i_times_2}
  util.return %blocked : tensor<?x?xf32>
}

util.func public @avoid_nan_scale(%arg0: tensor<?x?xi8>) -> tensor<?x?xi8> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %o = tensor.dim %arg0, %c0 : tensor<?x?xi8>
  %i = tensor.dim %arg0, %c1 : tensor<?x?xi8>

  %empty = tensor.empty(%o, %i) : tensor<?x?xi8>
  %cvt = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0 : tensor<?x?xi8>) outs(%empty : tensor<?x?xi8>) {
  ^bb0(%scale: i8, %out: i8):
    %c6 = arith.constant 6 : i8
    %c127 = arith.constant 127 : i8
    %upper = arith.shrui %scale, %c6 : i8
    %fixed_scale = arith.addi %upper, %c127 : i8
    linalg.yield %fixed_scale : i8
  } -> tensor<?x?xi8>
  util.return %cvt : tensor<?x?xi8>
}

}
