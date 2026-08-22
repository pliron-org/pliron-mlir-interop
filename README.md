# pliron-mlir-interop

IR translation between pliron and MLIR.

Design Goal: The target IR is textual.

 - Pliron -> MLIR should not require MLIR (or any C++ dependency)
 - MLIR -> Pliron should not require Pliron (or any Rust dependency)
