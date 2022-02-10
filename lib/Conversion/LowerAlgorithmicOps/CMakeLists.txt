add_mlir_conversion_library(TorchMLIRLowerAlgorithmicOps
  LowerAlgorithmicOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/torch-mlir/Conversion/LowerAlgorithmicOps

  DEPENDS
  TorchMLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRLinalg
  MLIRMath
  TorchMLIRTorchDialect
  TorchMLIRTMTensorDialect
)

torch_mlir_target_includes(TorchMLIRLowerAlgorithmicOps)
