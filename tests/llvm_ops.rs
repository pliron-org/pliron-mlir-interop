// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron-mlir-interop contributors

//! Translation of the pliron LLVM dialect's ops to MLIR.

mod common;

use core::num::NonZero;

use expect_test::expect;
use pliron::{
    attribute::AttrObj,
    basic_block::BasicBlock,
    builtin::{
        attributes::{IntegerAttr, StringAttr},
        op_interfaces::{
            CallOpCallable, IsTerminatorInterface, OneResultInterface, SingleBlockRegionInterface,
            SymbolOpInterface,
        },
        ops::ModuleOp,
        types::{FP32Type, IntegerType, Signedness},
    },
    context::{Context, Ptr},
    linked_list::ContainsLinkedList,
    op::{Op, op_impls},
    operation::Operation,
    region::Region,
    r#type::TypeHandle,
    utils::apint::APInt,
    value::Value,
};
use pliron_llvm::{
    attributes::{
        AtomicOrderingAttr, AtomicRmwKindAttr, FCmpPredicateAttr, FastmathFlags, FastmathFlagsAttr,
        ICmpPredicateAttr, IntegerOverflowFlagsAttr, LinkageAttr,
    },
    op_interfaces::{
        AlignableOpInterface, BinArithOp, CastOpInterface, CastOpWithNNegInterface,
        FloatBinArithOpWithFastMathFlags, IntBinArithOpWithOverflowFlag,
    },
    ops::{
        AShrOp, AddOp, AddrSpaceCastOp, AddressOfOp, AllocaOp, AndOp, AtomicCmpxchgOp,
        AtomicLoadOp, AtomicRmwOp, AtomicStoreOp, BitcastOp, BlockAddressOp, BlockTagOp, BrOp,
        CallIntrinsicOp, CallOp, CondBrOp, ConstantOp, ExtractElementOp, ExtractValueOp, FAddOp,
        FCmpOp, FDivOp, FMulOp, FNegOp, FPExtOp, FPToSIOp, FPToUIOp, FPTruncOp, FRemOp, FSubOp,
        FenceOp, FreezeOp, FuncOp, GepIndex, GetElementPtrOp, GlobalOp, ICmpOp, IndirectBrOp,
        InlineAsmOp, InsertElementOp, InsertValueOp, IntToPtrOp, LShrOp, LoadOp, MulOp, OrOp,
        PoisonOp, PtrToIntOp, ReturnOp, SDivOp, SExtOp, SIToFPOp, SRemOp, SelectOp, ShlOp,
        ShuffleVectorOp, StoreOp, SubOp, SwitchCase, SwitchOp, TruncOp, UDivOp, UIToFPOp, URemOp,
        UndefOp, UnreachableOp, VAArgOp, XorOp, ZExtOp, ZeroOp,
    },
    types::{
        FuncType, PointerType, StructLayout, StructType, VectorType, VectorTypeKind, VoidType,
    },
};

/// An `llvm.func` under construction, in a module of its own.
///
/// Ops are appended to its entry block, whose arguments are `arg_types`;
/// [TestFunc::finish] then translates the whole module.
struct TestFunc {
    module: ModuleOp,
    func: FuncOp,
    entry: Ptr<BasicBlock>,
}

impl TestFunc {
    fn new(ctx: &mut Context, arg_types: Vec<TypeHandle>) -> Self {
        let void_ty = VoidType::get(ctx).to_handle();
        let func_ty = FuncType::get(ctx, void_ty, arg_types, false);
        let module = ModuleOp::new(ctx, "m".try_into().unwrap());
        let func = FuncOp::new(ctx, "f".try_into().unwrap(), func_ty);
        module.append_operation(ctx, func.get_operation(), 0);
        let entry = func.get_or_create_entry_block(ctx);
        TestFunc {
            module,
            func,
            entry,
        }
    }

    fn arg(&self, ctx: &Context, idx: usize) -> Value {
        self.entry.deref(ctx).arguments().nth(idx).unwrap()
    }

    /// The function's body region, to hang further blocks off.
    fn region(&self, ctx: &Context) -> Ptr<Region> {
        self.func.get_operation().deref(ctx).get_region(0)
    }

    /// Build an op with `build`, append it to the entry block, and return its single result.
    fn push<O: Op>(&self, ctx: &mut Context, build: impl FnOnce(&mut Context) -> O) -> Value {
        let op = build(ctx).get_operation();
        op.insert_at_back(self.entry, ctx);
        op.deref(ctx)
            .results()
            .next()
            .unwrap_or_else(|| panic!("op has no result"))
    }

    /// [TestFunc::push] for an op with no result.
    fn push_no_result<O: Op>(&self, ctx: &mut Context, build: impl FnOnce(&mut Context) -> O) {
        build(ctx).get_operation().insert_at_back(self.entry, ctx);
    }

    /// Terminate the entry block and translate the whole module.
    fn finish(self, ctx: &mut Context) -> String {
        let terminated = self.entry.deref(ctx).get_tail().is_some_and(|tail| {
            op_impls::<dyn IsTerminatorInterface>(&*Operation::get_op_dyn(tail, ctx))
        });
        if !terminated {
            let ret = ReturnOp::new(ctx, None);
            ret.get_operation().insert_at_back(self.entry, ctx);
        }
        common::to_mlir(ctx, &self.module.get_operation())
    }
}

fn int_ty(ctx: &Context, width: u32) -> TypeHandle {
    IntegerType::get(ctx, width, Signedness::Signless).into()
}

fn int_attr(ctx: &Context, width: u32, value: u64) -> IntegerAttr {
    IntegerAttr::new(
        IntegerType::get(ctx, width, Signedness::Signless),
        APInt::from_u64(value, NonZero::new(width as usize).unwrap()),
    )
}

#[test]
fn integer_arithmetic() {
    let ctx = &mut Context::new();
    let i32_ty = int_ty(ctx, 32);
    let func = TestFunc::new(ctx, vec![i32_ty, i32_ty]);
    let (a, b) = (func.arg(ctx, 0), func.arg(ctx, 1));

    let nsw_nuw = IntegerOverflowFlagsAttr {
        nsw: true,
        nuw: true,
    };
    func.push(ctx, |ctx| AddOp::new_with_overflow_flag(ctx, a, b, nsw_nuw));
    func.push(ctx, |ctx| SubOp::new(ctx, a, b));
    func.push(ctx, |ctx| MulOp::new(ctx, a, b));
    func.push(ctx, |ctx| ShlOp::new(ctx, a, b));
    func.push(ctx, |ctx| UDivOp::new(ctx, a, b));
    func.push(ctx, |ctx| SDivOp::new(ctx, a, b));
    func.push(ctx, |ctx| URemOp::new(ctx, a, b));
    func.push(ctx, |ctx| SRemOp::new(ctx, a, b));
    func.push(ctx, |ctx| AndOp::new(ctx, a, b));
    func.push(ctx, |ctx| OrOp::new(ctx, a, b));
    func.push(ctx, |ctx| XorOp::new(ctx, a, b));
    func.push(ctx, |ctx| LShrOp::new(ctx, a, b));
    func.push(ctx, |ctx| AShrOp::new(ctx, a, b));
    func.push(ctx, |ctx| ICmpOp::new(ctx, ICmpPredicateAttr::SLT, a, b));

    let printed = func.finish(ctx);
    expect![[r#"
        "builtin.module"() <{sym_name = "m"}> ({
          ^block1v1:
            "llvm.func"() <{function_type = !llvm.func<!llvm.void (i32, i32)>, linkage = #llvm.linkage<external>, sym_name = "f"}> ({
              ^entry_block2v1(%v0: i32, %v1: i32):
                %v2 = "llvm.add"(%v0, %v1) <{overflowFlags = #llvm.overflow<nsw, nuw>}> : (i32, i32) -> (i32)
                %v3 = "llvm.sub"(%v0, %v1) : (i32, i32) -> (i32)
                %v4 = "llvm.mul"(%v0, %v1) : (i32, i32) -> (i32)
                %v5 = "llvm.shl"(%v0, %v1) : (i32, i32) -> (i32)
                %v6 = "llvm.udiv"(%v0, %v1) : (i32, i32) -> (i32)
                %v7 = "llvm.sdiv"(%v0, %v1) : (i32, i32) -> (i32)
                %v8 = "llvm.urem"(%v0, %v1) : (i32, i32) -> (i32)
                %v9 = "llvm.srem"(%v0, %v1) : (i32, i32) -> (i32)
                %v10 = "llvm.and"(%v0, %v1) : (i32, i32) -> (i32)
                %v11 = "llvm.or"(%v0, %v1) : (i32, i32) -> (i32)
                %v12 = "llvm.xor"(%v0, %v1) : (i32, i32) -> (i32)
                %v13 = "llvm.lshr"(%v0, %v1) : (i32, i32) -> (i32)
                %v14 = "llvm.ashr"(%v0, %v1) : (i32, i32) -> (i32)
                %v15 = "llvm.icmp"(%v0, %v1) <{predicate = 2 : i64}> : (i32, i32) -> (i1)
                "llvm.return"() : () -> ()
            }) : () -> ()
        }) : () -> ()"#]].assert_eq(&printed);
    common::expect_mlir_opt_output(
        &printed,
        expect![[r#"
        module @m {
          llvm.func @f(%arg0: i32, %arg1: i32) {
            %0 = llvm.add %arg0, %arg1 overflow<nsw, nuw> : i32
            %1 = llvm.sub %arg0, %arg1 : i32
            %2 = llvm.mul %arg0, %arg1 : i32
            %3 = llvm.shl %arg0, %arg1 : i32
            %4 = llvm.udiv %arg0, %arg1 : i32
            %5 = llvm.sdiv %arg0, %arg1 : i32
            %6 = llvm.urem %arg0, %arg1 : i32
            %7 = llvm.srem %arg0, %arg1 : i32
            %8 = llvm.and %arg0, %arg1 : i32
            %9 = llvm.or %arg0, %arg1 : i32
            %10 = llvm.xor %arg0, %arg1 : i32
            %11 = llvm.lshr %arg0, %arg1 : i32
            %12 = llvm.ashr %arg0, %arg1 : i32
            %13 = llvm.icmp "slt" %arg0, %arg1 : i32
            llvm.return
          }
        }

    "#]],
    );
}

#[test]
fn float_arithmetic() {
    let ctx = &mut Context::new();
    let f32_ty: TypeHandle = FP32Type::get(ctx).into();
    let func = TestFunc::new(ctx, vec![f32_ty, f32_ty]);
    let (a, b) = (func.arg(ctx, 0), func.arg(ctx, 1));

    let fast = FastmathFlagsAttr(FastmathFlags::NNAN | FastmathFlags::NINF);
    func.push(ctx, |ctx| FAddOp::new_with_fast_math_flags(ctx, a, b, fast));
    func.push(ctx, |ctx| FSubOp::new(ctx, a, b));
    func.push(ctx, |ctx| FMulOp::new(ctx, a, b));
    func.push(ctx, |ctx| FDivOp::new(ctx, a, b));
    func.push(ctx, |ctx| FRemOp::new(ctx, a, b));
    func.push(ctx, |ctx| {
        FNegOp::new_with_fast_math_flags(ctx, a, FastmathFlagsAttr::default())
    });
    func.push(ctx, |ctx| FCmpOp::new(ctx, FCmpPredicateAttr::OEQ, a, b));

    let printed = func.finish(ctx);
    expect![[r#"
        "builtin.module"() <{sym_name = "m"}> ({
          ^block1v1:
            "llvm.func"() <{function_type = !llvm.func<!llvm.void (f32, f32)>, linkage = #llvm.linkage<external>, sym_name = "f"}> ({
              ^entry_block2v1(%v0: f32, %v1: f32):
                %v2 = "llvm.fadd"(%v0, %v1) <{fastmathFlags = #llvm.fastmath<nnan, ninf>}> : (f32, f32) -> (f32)
                %v3 = "llvm.fsub"(%v0, %v1) : (f32, f32) -> (f32)
                %v4 = "llvm.fmul"(%v0, %v1) : (f32, f32) -> (f32)
                %v5 = "llvm.fdiv"(%v0, %v1) : (f32, f32) -> (f32)
                %v6 = "llvm.frem"(%v0, %v1) : (f32, f32) -> (f32)
                %v7 = "llvm.fneg"(%v0) <{fastmathFlags = #llvm.fastmath<none>}> : (f32) -> (f32)
                %v8 = "llvm.fcmp"(%v0, %v1) <{predicate = 1 : i64}> : (f32, f32) -> (i1)
                "llvm.return"() : () -> ()
            }) : () -> ()
        }) : () -> ()"#]].assert_eq(&printed);
    common::expect_mlir_opt_output(
        &printed,
        expect![[r#"
        module @m {
          llvm.func @f(%arg0: f32, %arg1: f32) {
            %0 = llvm.fadd %arg0, %arg1 {fastmathFlags = #llvm.fastmath<nnan, ninf>} : f32
            %1 = llvm.fsub %arg0, %arg1 : f32
            %2 = llvm.fmul %arg0, %arg1 : f32
            %3 = llvm.fdiv %arg0, %arg1 : f32
            %4 = llvm.frem %arg0, %arg1 : f32
            %5 = llvm.fneg %arg0 : f32
            %6 = llvm.fcmp "oeq" %arg0, %arg1 : f32
            llvm.return
          }
        }

    "#]],
    );
}

#[test]
fn casts() {
    let ctx = &mut Context::new();
    let (i16_ty, i32_ty, i64_ty) = (int_ty(ctx, 16), int_ty(ctx, 32), int_ty(ctx, 64));
    let f32_ty: TypeHandle = FP32Type::get(ctx).into();
    let f64_ty: TypeHandle = pliron::builtin::types::FP64Type::get(ctx).into();
    let ptr_ty: TypeHandle = PointerType::get(ctx, 0).into();
    let ptr1_ty: TypeHandle = PointerType::get(ctx, 1).into();

    let func = TestFunc::new(ctx, vec![i32_ty, f32_ty, ptr_ty]);
    let (i, fl, p) = (func.arg(ctx, 0), func.arg(ctx, 1), func.arg(ctx, 2));

    func.push(ctx, |ctx| BitcastOp::new(ctx, i, f32_ty));
    func.push(ctx, |ctx| IntToPtrOp::new(ctx, i, ptr_ty));
    func.push(ctx, |ctx| PtrToIntOp::new(ctx, p, i64_ty));
    func.push(ctx, |ctx| AddrSpaceCastOp::new(ctx, p, ptr1_ty));
    func.push(ctx, |ctx| SExtOp::new(ctx, i, i64_ty));
    func.push(ctx, |ctx| ZExtOp::new(ctx, i, i64_ty));
    func.push(ctx, |ctx| ZExtOp::new_with_nneg(ctx, i, i64_ty, true));
    func.push(ctx, |ctx| TruncOp::new(ctx, i, i16_ty));
    func.push(ctx, |ctx| FPExtOp::new(ctx, fl, f64_ty));
    func.push(ctx, |ctx| FPTruncOp::new(ctx, fl, f32_ty));
    func.push(ctx, |ctx| FPToSIOp::new(ctx, fl, i32_ty));
    func.push(ctx, |ctx| FPToUIOp::new(ctx, fl, i32_ty));
    func.push(ctx, |ctx| SIToFPOp::new(ctx, i, f32_ty));
    func.push(ctx, |ctx| UIToFPOp::new_with_nneg(ctx, i, f32_ty, true));

    let printed = func.finish(ctx);
    expect![[r#"
        "builtin.module"() <{sym_name = "m"}> ({
          ^block1v1:
            "llvm.func"() <{function_type = !llvm.func<!llvm.void (i32, f32, !llvm.ptr)>, linkage = #llvm.linkage<external>, sym_name = "f"}> ({
              ^entry_block2v1(%v0: i32, %v1: f32, %v2: !llvm.ptr):
                %v3 = "llvm.bitcast"(%v0) : (i32) -> (f32)
                %v4 = "llvm.inttoptr"(%v0) : (i32) -> (!llvm.ptr)
                %v5 = "llvm.ptrtoint"(%v2) : (!llvm.ptr) -> (i64)
                %v6 = "llvm.addrspacecast"(%v2) : (!llvm.ptr) -> (!llvm.ptr<1>)
                %v7 = "llvm.sext"(%v0) : (i32) -> (i64)
                %v8 = "llvm.zext"(%v0) : (i32) -> (i64)
                %v9 = "llvm.zext"(%v0) <{nonNeg = unit}> : (i32) -> (i64)
                %v10 = "llvm.trunc"(%v0) : (i32) -> (i16)
                %v11 = "llvm.fpext"(%v1) : (f32) -> (f64)
                %v12 = "llvm.fptrunc"(%v1) : (f32) -> (f32)
                %v13 = "llvm.fptosi"(%v1) : (f32) -> (i32)
                %v14 = "llvm.fptoui"(%v1) : (f32) -> (i32)
                %v15 = "llvm.sitofp"(%v0) : (i32) -> (f32)
                %v16 = "llvm.uitofp"(%v0) <{nonNeg = unit}> : (i32) -> (f32)
                "llvm.return"() : () -> ()
            }) : () -> ()
        }) : () -> ()"#]].assert_eq(&printed);
    common::expect_mlir_opt_output(
        &printed,
        expect![[r#"
        module @m {
          llvm.func @f(%arg0: i32, %arg1: f32, %arg2: !llvm.ptr) {
            %0 = llvm.bitcast %arg0 : i32 to f32
            %1 = llvm.inttoptr %arg0 : i32 to !llvm.ptr
            %2 = llvm.ptrtoint %arg2 : !llvm.ptr to i64
            %3 = llvm.addrspacecast %arg2 : !llvm.ptr to !llvm.ptr<1>
            %4 = llvm.sext %arg0 : i32 to i64
            %5 = llvm.zext %arg0 : i32 to i64
            %6 = llvm.zext nneg %arg0 : i32 to i64
            %7 = llvm.trunc %arg0 : i32 to i16
            %8 = llvm.fpext %arg1 : f32 to f64
            %9 = llvm.fptrunc %arg1 : f32 to f32
            %10 = llvm.fptosi %arg1 : f32 to i32
            %11 = llvm.fptoui %arg1 : f32 to i32
            %12 = llvm.sitofp %arg0 : i32 to f32
            %13 = llvm.uitofp nneg %arg0 : i32 to f32
            llvm.return
          }
        }

    "#]],
    );
}

#[test]
fn memory() {
    let ctx = &mut Context::new();
    let i32_ty = int_ty(ctx, 32);
    let ptr_ty: TypeHandle = PointerType::get(ctx, 0).into();
    let pair_ty: TypeHandle =
        StructType::get_unnamed(ctx, (vec![i32_ty, i32_ty], StructLayout::Unpacked)).into();

    let func = TestFunc::new(ctx, vec![i32_ty, ptr_ty]);
    let (n, p) = (func.arg(ctx, 0), func.arg(ctx, 1));

    let slot = func.push(ctx, |ctx| {
        let alloca = AllocaOp::new(ctx, i32_ty, n);
        alloca.set_alignment(ctx, 8);
        alloca
    });

    let loaded = func.push(ctx, |ctx| {
        let load = LoadOp::new(ctx, slot, i32_ty);
        load.set_alignment(ctx, 4);
        load
    });
    func.push_no_result(ctx, |ctx| StoreOp::new(ctx, loaded, slot));

    func.push(ctx, |ctx| {
        GetElementPtrOp::new(
            ctx,
            p,
            vec![GepIndex::Value(n), GepIndex::Constant(1)],
            pair_ty,
        )
    });

    func.push(ctx, |ctx| {
        AtomicRmwOp::new(
            ctx,
            p,
            loaded,
            AtomicRmwKindAttr::Add,
            AtomicOrderingAttr::Monotonic,
            None,
        )
    });
    func.push(ctx, |ctx| {
        AtomicCmpxchgOp::new(
            ctx,
            p,
            loaded,
            loaded,
            AtomicOrderingAttr::AcqRel,
            AtomicOrderingAttr::Monotonic,
            Some("agent".to_string()),
        )
    });
    func.push_no_result(ctx, |ctx| {
        FenceOp::new(ctx, AtomicOrderingAttr::SeqCst, Some("agent".to_string()))
    });

    let atomically_loaded = func.push(ctx, |ctx| {
        let atomic_load = AtomicLoadOp::new(ctx, p, i32_ty, AtomicOrderingAttr::Acquire, None);
        atomic_load.set_alignment(ctx, 4);
        atomic_load
    });
    func.push_no_result(ctx, |ctx| {
        let atomic_store =
            AtomicStoreOp::new(ctx, atomically_loaded, p, AtomicOrderingAttr::Release, None);
        atomic_store.set_alignment(ctx, 4);
        atomic_store
    });

    let printed = func.finish(ctx);
    expect![[r#"
        "builtin.module"() <{sym_name = "m"}> ({
          ^block1v1:
            "llvm.func"() <{function_type = !llvm.func<!llvm.void (i32, !llvm.ptr)>, linkage = #llvm.linkage<external>, sym_name = "f"}> ({
              ^entry_block2v1(%v0: i32, %v1: !llvm.ptr):
                %v2 = "llvm.alloca"(%v0) <{elem_type = i32, alignment = 8 : i64}> : (i32) -> (!llvm.ptr)
                %v3 = "llvm.load"(%v2) <{alignment = 4 : i64}> : (!llvm.ptr) -> (i32)
                "llvm.store"(%v3, %v2) : (i32, !llvm.ptr) -> ()
                %v4 = "llvm.getelementptr"(%v1, %v0) <{elem_type = !llvm.struct<(i32, i32)>, rawConstantIndices = array<i32: -2147483648, 1>}> : (!llvm.ptr, i32) -> (!llvm.ptr)
                %v5 = "llvm.atomicrmw"(%v1, %v3) <{bin_op = 1 : i64, ordering = 2 : i64}> : (!llvm.ptr, i32) -> (i32)
                %v6 = "llvm.cmpxchg"(%v1, %v3, %v3) <{failure_ordering = 2 : i64, success_ordering = 6 : i64, syncscope = "agent"}> : (!llvm.ptr, i32, i32) -> (!llvm.struct<(i32, i1)>)
                "llvm.fence"() <{ordering = 7 : i64, syncscope = "agent"}> : () -> ()
                %v7 = "llvm.load"(%v1) <{alignment = 4 : i64, ordering = 4 : i64}> : (!llvm.ptr) -> (i32)
                "llvm.store"(%v7, %v1) <{alignment = 4 : i64, ordering = 5 : i64}> : (i32, !llvm.ptr) -> ()
                "llvm.return"() : () -> ()
            }) : () -> ()
        }) : () -> ()"#]].assert_eq(&printed);
    common::expect_mlir_opt_output(
        &printed,
        expect![[r#"
        module @m {
          llvm.func @f(%arg0: i32, %arg1: !llvm.ptr) {
            %0 = llvm.alloca %arg0 x i32 {alignment = 8 : i64} : (i32) -> !llvm.ptr
            %1 = llvm.load %0 {alignment = 4 : i64} : !llvm.ptr -> i32
            llvm.store %1, %0 : i32, !llvm.ptr
            %2 = llvm.getelementptr %arg1[%arg0, 1] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<(i32, i32)>
            %3 = llvm.atomicrmw add %arg1, %1 monotonic : !llvm.ptr, i32
            %4 = llvm.cmpxchg %arg1, %1, %1 syncscope("agent") acq_rel monotonic : !llvm.ptr, i32
            llvm.fence syncscope("agent") seq_cst
            %5 = llvm.load %arg1 atomic acquire {alignment = 4 : i64} : !llvm.ptr -> i32
            llvm.store %5, %arg1 atomic release {alignment = 4 : i64} : i32, !llvm.ptr
            llvm.return
          }
        }

    "#]],
    );
}

#[test]
fn aggregates_and_vectors() {
    let ctx = &mut Context::new();
    let i32_ty = int_ty(ctx, 32);
    let i1_ty = int_ty(ctx, 1);
    let f32_ty: TypeHandle = FP32Type::get(ctx).into();
    let struct_ty: TypeHandle =
        StructType::get_unnamed(ctx, (vec![i32_ty, f32_ty], StructLayout::Unpacked)).into();
    let vec_ty: TypeHandle = VectorType::get(ctx, i32_ty, 4, VectorTypeKind::Fixed).into();

    let func = TestFunc::new(ctx, vec![i32_ty, i1_ty, struct_ty, vec_ty]);
    let (i, cond, agg, vec) = (
        func.arg(ctx, 0),
        func.arg(ctx, 1),
        func.arg(ctx, 2),
        func.arg(ctx, 3),
    );

    func.push(ctx, |ctx| InsertValueOp::new(ctx, agg, i, vec![0]));
    func.push(ctx, |ctx| ExtractValueOp::new(ctx, agg, vec![1]).unwrap());
    func.push(ctx, |ctx| InsertElementOp::new(ctx, vec, i, i));
    func.push(ctx, |ctx| ExtractElementOp::new(ctx, vec, i));
    func.push(ctx, |ctx| {
        ShuffleVectorOp::new(ctx, vec, vec, vec![3, 2, 1, 0])
    });
    func.push(ctx, |ctx| SelectOp::new(ctx, cond, i, i));
    func.push(ctx, |ctx| FreezeOp::new(ctx, i));

    let printed = func.finish(ctx);
    expect![[r#"
        "builtin.module"() <{sym_name = "m"}> ({
          ^block1v1:
            "llvm.func"() <{function_type = !llvm.func<!llvm.void (i32, i1, !llvm.struct<(i32, f32)>, vector<4 x i32>)>, linkage = #llvm.linkage<external>, sym_name = "f"}> ({
              ^entry_block2v1(%v0: i32, %v1: i1, %v2: !llvm.struct<(i32, f32)>, %v3: vector<4 x i32>):
                %v4 = "llvm.insertvalue"(%v2, %v0) <{position = array<i64: 0>}> : (!llvm.struct<(i32, f32)>, i32) -> (!llvm.struct<(i32, f32)>)
                %v5 = "llvm.extractvalue"(%v2) <{position = array<i64: 1>}> : (!llvm.struct<(i32, f32)>) -> (f32)
                %v6 = "llvm.insertelement"(%v3, %v0, %v0) : (vector<4 x i32>, i32, i32) -> (vector<4 x i32>)
                %v7 = "llvm.extractelement"(%v3, %v0) : (vector<4 x i32>, i32) -> (i32)
                %v8 = "llvm.shufflevector"(%v3, %v3) <{mask = array<i32: 3, 2, 1, 0>}> : (vector<4 x i32>, vector<4 x i32>) -> (vector<4 x i32>)
                %v9 = "llvm.select"(%v1, %v0, %v0) : (i1, i32, i32) -> (i32)
                %v10 = "llvm.freeze"(%v0) : (i32) -> (i32)
                "llvm.return"() : () -> ()
            }) : () -> ()
        }) : () -> ()"#]].assert_eq(&printed);
    common::expect_mlir_opt_output(
        &printed,
        expect![[r#"
        module @m {
          llvm.func @f(%arg0: i32, %arg1: i1, %arg2: !llvm.struct<(i32, f32)>, %arg3: vector<4xi32>) {
            %0 = llvm.insertvalue %arg0, %arg2[0] : !llvm.struct<(i32, f32)> 
            %1 = llvm.extractvalue %arg2[1] : !llvm.struct<(i32, f32)> 
            %2 = llvm.insertelement %arg0, %arg3[%arg0 : i32] : vector<4xi32>
            %3 = llvm.extractelement %arg3[%arg0 : i32] : vector<4xi32>
            %4 = llvm.shufflevector %arg3, %arg3 [3, 2, 1, 0] : vector<4xi32> 
            %5 = llvm.select %arg1, %arg0, %arg0 : i1, i32
            %6 = llvm.freeze %arg0 : i32
            llvm.return
          }
        }

    "#]],
    );
}

#[test]
fn constants() {
    let ctx = &mut Context::new();
    let i32_ty = int_ty(ctx, 32);
    let ptr_ty: TypeHandle = PointerType::get(ctx, 0).into();
    let func = TestFunc::new(ctx, vec![]);

    let value: AttrObj = Box::new(int_attr(ctx, 32, 42));
    func.push(ctx, |ctx| ConstantOp::new(ctx, value));
    // MLIR has dedicated ops for these, rather than a constant attribute.
    func.push(ctx, |ctx| {
        ConstantOp::new(ctx, Box::new(pliron_llvm::attributes::ZeroAttr(ptr_ty)))
    });
    func.push(ctx, |ctx| {
        ConstantOp::new(ctx, Box::new(pliron_llvm::attributes::UndefAttr(i32_ty)))
    });
    func.push(ctx, |ctx| {
        ConstantOp::new(ctx, Box::new(pliron_llvm::attributes::PoisonAttr(i32_ty)))
    });
    func.push(ctx, |ctx| UndefOp::new(ctx, i32_ty));
    func.push(ctx, |ctx| PoisonOp::new(ctx, i32_ty));
    func.push(ctx, |ctx| ZeroOp::new(ctx, ptr_ty));

    let printed = func.finish(ctx);
    expect![[r#"
        "builtin.module"() <{sym_name = "m"}> ({
          ^block1v1:
            "llvm.func"() <{function_type = !llvm.func<!llvm.void ()>, linkage = #llvm.linkage<external>, sym_name = "f"}> ({
              ^entry_block2v1:
                %v0 = "llvm.mlir.constant"() <{value = 42 : i32}> : () -> (i32)
                %v1 = "llvm.mlir.zero"() : () -> (!llvm.ptr)
                %v2 = "llvm.mlir.undef"() : () -> (i32)
                %v3 = "llvm.mlir.poison"() : () -> (i32)
                %v4 = "llvm.mlir.undef"() : () -> (i32)
                %v5 = "llvm.mlir.poison"() : () -> (i32)
                %v6 = "llvm.mlir.zero"() : () -> (!llvm.ptr)
                "llvm.return"() : () -> ()
            }) : () -> ()
        }) : () -> ()"#]].assert_eq(&printed);
    common::expect_mlir_opt_output(
        &printed,
        expect![[r#"
        module @m {
          llvm.func @f() {
            %0 = llvm.mlir.constant(42 : i32) : i32
            %1 = llvm.mlir.zero : !llvm.ptr
            %2 = llvm.mlir.undef : i32
            %3 = llvm.mlir.poison : i32
            %4 = llvm.mlir.undef : i32
            %5 = llvm.mlir.poison : i32
            %6 = llvm.mlir.zero : !llvm.ptr
            llvm.return
          }
        }

    "#]],
    );
}

#[test]
fn control_flow() {
    let ctx = &mut Context::new();
    let i1_ty = int_ty(ctx, 1);
    let i32_ty = int_ty(ctx, 32);
    let func = TestFunc::new(ctx, vec![i1_ty, i32_ty]);
    let (cond, n) = (func.arg(ctx, 0), func.arg(ctx, 1));

    let region = func.region(ctx);
    let with_arg = BasicBlock::new(ctx, Some("with_arg".try_into().unwrap()), vec![i32_ty]);
    let joined = BasicBlock::new(ctx, Some("joined".try_into().unwrap()), vec![]);
    let done = BasicBlock::new(ctx, Some("done".try_into().unwrap()), vec![]);
    with_arg.insert_at_back(region, ctx);
    joined.insert_at_back(region, ctx);
    done.insert_at_back(region, ctx);

    func.push_no_result(ctx, |ctx| BrOp::new(ctx, with_arg, vec![n]));

    let cond_br = CondBrOp::new(ctx, cond, joined, vec![], done, vec![]);
    cond_br.get_operation().insert_at_back(with_arg, ctx);

    let switch = SwitchOp::new(
        ctx,
        n,
        done,
        vec![],
        vec![
            SwitchCase {
                value: int_attr(ctx, 32, 0),
                dest: done,
                dest_opds: vec![],
            },
            SwitchCase {
                value: int_attr(ctx, 32, 7),
                dest: with_arg,
                dest_opds: vec![n],
            },
        ],
    );
    switch.get_operation().insert_at_back(joined, ctx);

    UnreachableOp::new(ctx)
        .get_operation()
        .insert_at_back(done, ctx);

    let printed = func.finish(ctx);
    expect![[r#"
        "builtin.module"() <{sym_name = "m"}> ({
          ^block1v1:
            "llvm.func"() <{function_type = !llvm.func<!llvm.void (i1, i32)>, linkage = #llvm.linkage<external>, sym_name = "f"}> ({
              ^entry_block2v1(%v0: i1, %v1: i32):
                "llvm.br"(%v1) [^with_arg_block3v1] : (i32) -> ()
              ^with_arg_block3v1(%v2: i32):
                "llvm.cond_br"(%v0) [^joined_block4v1, ^done_block5v1] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
              ^joined_block4v1:
                "llvm.switch"(%v1, %v1) [^done_block5v1, ^done_block5v1, ^with_arg_block3v1] <{case_operand_segments = array<i32: 0, 1>, operandSegmentSizes = array<i32: 1, 0, 1>, case_values = dense<[0, 7]> : vector<2xi32>}> : (i32, i32) -> ()
              ^done_block5v1:
                "llvm.unreachable"() : () -> ()
            }) : () -> ()
        }) : () -> ()"#]].assert_eq(&printed);
    common::expect_mlir_opt_output(
        &printed,
        expect![[r#"
        module @m {
          llvm.func @f(%arg0: i1, %arg1: i32) {
            llvm.br ^bb1(%arg1 : i32)
          ^bb1(%0: i32):  // 2 preds: ^bb0, ^bb2
            llvm.cond_br %arg0, ^bb2, ^bb3
          ^bb2:  // pred: ^bb1
            llvm.switch %arg1 : i32, ^bb3 [
              0: ^bb3,
              7: ^bb1(%arg1 : i32)
            ]
          ^bb3:  // 3 preds: ^bb1, ^bb2, ^bb2
            llvm.unreachable
          }
        }

    "#]],
    );
}

#[test]
fn indirect_branches() {
    let ctx = &mut Context::new();
    let i32_ty = int_ty(ctx, 32);
    let ptr_ty: TypeHandle = PointerType::get(ctx, 0).into();
    let func = TestFunc::new(ctx, vec![ptr_ty, i32_ty]);
    let (p, n) = (func.arg(ctx, 0), func.arg(ctx, 1));

    let region = func.region(ctx);
    let tagged = BasicBlock::new(ctx, Some("tagged".try_into().unwrap()), vec![]);
    let other = BasicBlock::new(ctx, Some("other".try_into().unwrap()), vec![i32_ty]);
    tagged.insert_at_back(region, ctx);
    other.insert_at_back(region, ctx);

    func.push(ctx, |ctx| {
        BlockAddressOp::new(ctx, "f".try_into().unwrap(), 1, 0)
    });
    func.push_no_result(ctx, |ctx| {
        IndirectBrOp::new(ctx, p, vec![(tagged, vec![]), (other, vec![n])])
    });

    BlockTagOp::new(ctx, 1)
        .get_operation()
        .insert_at_back(tagged, ctx);
    UnreachableOp::new(ctx)
        .get_operation()
        .insert_at_back(tagged, ctx);
    UnreachableOp::new(ctx)
        .get_operation()
        .insert_at_back(other, ctx);

    let printed = func.finish(ctx);
    expect![[r#"
        "builtin.module"() <{sym_name = "m"}> ({
          ^block1v1:
            "llvm.func"() <{function_type = !llvm.func<!llvm.void (!llvm.ptr, i32)>, linkage = #llvm.linkage<external>, sym_name = "f"}> ({
              ^entry_block2v1(%v0: !llvm.ptr, %v1: i32):
                %v3 = "llvm.blockaddress"() <{block_addr = #llvm.blockaddress<function = @f, tag = <id = 1>>}> : () -> (!llvm.ptr)
                "llvm.indirectbr"(%v0, %v1) [^tagged_block3v1, ^other_block4v1] <{indbr_operand_segments = array<i32: 0, 1>}> : (!llvm.ptr, i32) -> ()
              ^tagged_block3v1:
                "llvm.blocktag"() <{tag = #llvm.blocktag<id = 1>}> : () -> ()
                "llvm.unreachable"() : () -> ()
              ^other_block4v1(%v2: i32):
                "llvm.unreachable"() : () -> ()
            }) : () -> ()
        }) : () -> ()"#]].assert_eq(&printed);
    common::expect_mlir_opt_output(
        &printed,
        expect![[r#"
        module @m {
          llvm.func @f(%arg0: !llvm.ptr, %arg1: i32) {
            %0 = llvm.blockaddress <function = @f, tag = <id = 1>> : !llvm.ptr
            llvm.indirectbr %arg0 : !llvm.ptr, [
            ^bb1,
            ^bb2(%arg1 : i32)
            ]
          ^bb1:  // pred: ^bb0
            llvm.blocktag <id = 1>
            llvm.unreachable
          ^bb2(%1: i32):  // pred: ^bb0
            llvm.unreachable
          }
        }

    "#]],
    );
}

#[test]
fn calls() {
    let ctx = &mut Context::new();
    let i32_ty = int_ty(ctx, 32);
    let ptr_ty: TypeHandle = PointerType::get(ctx, 0).into();
    let void_ty = VoidType::get(ctx).to_handle();

    let func = TestFunc::new(ctx, vec![i32_ty, ptr_ty]);
    let (n, p) = (func.arg(ctx, 0), func.arg(ctx, 1));

    // Declarations for the direct calls to name.
    let returns_i32 = FuncType::get(ctx, i32_ty, vec![i32_ty], false);
    let returns_void = FuncType::get(ctx, void_ty, vec![i32_ty], false);
    let variadic = FuncType::get(ctx, i32_ty, vec![ptr_ty], true);
    for (name, ty) in [
        ("takes_i32", returns_i32),
        ("returns_nothing", returns_void),
        ("printf_like", variadic),
    ] {
        let decl = FuncOp::new(ctx, name.try_into().unwrap(), ty);
        func.module.append_operation(ctx, decl.get_operation(), 0);
    }

    func.push(ctx, |ctx| {
        CallOp::new(
            ctx,
            CallOpCallable::Direct("takes_i32".try_into().unwrap()),
            returns_i32,
            vec![n],
        )
    });
    // A void call has no MLIR result.
    func.push_no_result(ctx, |ctx| {
        CallOp::new(
            ctx,
            CallOpCallable::Direct("returns_nothing".try_into().unwrap()),
            returns_void,
            vec![n],
        )
    });
    func.push(ctx, |ctx| {
        CallOp::new(
            ctx,
            CallOpCallable::Direct("printf_like".try_into().unwrap()),
            variadic,
            vec![p],
        )
    });
    func.push(ctx, |ctx| {
        CallOp::new(ctx, CallOpCallable::Indirect(p), returns_i32, vec![n])
    });

    let smax_ty = FuncType::get(ctx, i32_ty, vec![i32_ty, i32_ty], false);
    func.push(ctx, |ctx| {
        CallIntrinsicOp::new(
            ctx,
            StringAttr::new("llvm.smax.i32".to_string()),
            smax_ty,
            vec![n, n],
        )
    });

    func.push(ctx, |ctx| {
        InlineAsmOp::new(ctx, i32_ty, vec![n], "nop", "=r,r", false)
    });
    func.push_no_result(ctx, |ctx| {
        InlineAsmOp::new(ctx, void_ty, vec![], "nop", "", true)
    });

    func.push(ctx, |ctx| VAArgOp::new(ctx, p, i32_ty));

    let printed = func.finish(ctx);
    expect![[r#"
        "builtin.module"() <{sym_name = "m"}> ({
          ^block1v1:
            "llvm.func"() <{function_type = !llvm.func<!llvm.void (i32, !llvm.ptr)>, linkage = #llvm.linkage<external>, sym_name = "f"}> ({
              ^entry_block2v1(%v0: i32, %v1: !llvm.ptr):
                %v2 = "llvm.call"(%v0) <{op_bundle_sizes = array<i32>, operandSegmentSizes = array<i32: 1, 0>, callee = @takes_i32}> : (i32) -> (i32)
                "llvm.call"(%v0) <{op_bundle_sizes = array<i32>, operandSegmentSizes = array<i32: 1, 0>, callee = @returns_nothing}> : (i32) -> ()
                %v4 = "llvm.call"(%v1) <{op_bundle_sizes = array<i32>, operandSegmentSizes = array<i32: 1, 0>, callee = @printf_like, var_callee_type = !llvm.func<i32 (!llvm.ptr, ...)>}> : (!llvm.ptr) -> (i32)
                %v5 = "llvm.call"(%v1, %v0) <{op_bundle_sizes = array<i32>, operandSegmentSizes = array<i32: 2, 0>}> : (!llvm.ptr, i32) -> (i32)
                %v6 = "llvm.call_intrinsic"(%v0, %v0) <{intrin = "llvm.smax.i32", op_bundle_sizes = array<i32>, operandSegmentSizes = array<i32: 2, 0>}> : (i32, i32) -> (i32)
                %v7 = "llvm.inline_asm"(%v0) <{asm_string = "nop", constraints = "=r,r"}> : (i32) -> (i32)
                "llvm.inline_asm"() <{asm_string = "nop", constraints = ""}> {llvm_convergent = unit} : () -> ()
                %v9 = "llvm.va_arg"(%v1) : (!llvm.ptr) -> (i32)
                "llvm.return"() : () -> ()
            }) : () -> ()
            "llvm.func"() <{function_type = !llvm.func<i32 (i32)>, linkage = #llvm.linkage<external>, sym_name = "takes_i32"}> ({}) : () -> ()
            "llvm.func"() <{function_type = !llvm.func<!llvm.void (i32)>, linkage = #llvm.linkage<external>, sym_name = "returns_nothing"}> ({}) : () -> ()
            "llvm.func"() <{function_type = !llvm.func<i32 (!llvm.ptr, ...)>, linkage = #llvm.linkage<external>, sym_name = "printf_like"}> ({}) : () -> ()
        }) : () -> ()"#]].assert_eq(&printed);
    common::expect_mlir_opt_output(
        &printed,
        expect![[r#"
        module @m {
          llvm.func @f(%arg0: i32, %arg1: !llvm.ptr) {
            %0 = llvm.call @takes_i32(%arg0) : (i32) -> i32
            llvm.call @returns_nothing(%arg0) : (i32) -> ()
            %1 = llvm.call @printf_like(%arg1) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
            %2 = llvm.call %arg1(%arg0) : !llvm.ptr, (i32) -> i32
            %3 = llvm.call_intrinsic "llvm.smax.i32"(%arg0, %arg0) : (i32, i32) -> i32
            %4 = llvm.inline_asm "nop", "=r,r" %arg0 : (i32) -> i32
            llvm.inline_asm {llvm_convergent} "nop", ""  : () -> ()
            %5 = llvm.va_arg %arg1 : (!llvm.ptr) -> i32
            llvm.return
          }
          llvm.func @takes_i32(i32) -> i32
          llvm.func @returns_nothing(i32)
          llvm.func @printf_like(!llvm.ptr, ...) -> i32
        }

    "#]],
    );
}

#[test]
fn globals() {
    let ctx = &mut Context::new();
    let i32_ty = int_ty(ctx, 32);
    let module = ModuleOp::new(ctx, "m".try_into().unwrap());

    // A global with a simple initializer value.
    let simple = GlobalOp::new(ctx, "counter".try_into().unwrap(), i32_ty);
    simple.set_attr_llvm_global_linkage(ctx, LinkageAttr::InternalLinkage);
    simple.set_initializer_value(ctx, Box::new(int_attr(ctx, 32, 7)));
    simple.set_alignment(ctx, 4);
    module.append_operation(ctx, simple.get_operation(), 0);

    // A global with an initializer region.
    let with_region = GlobalOp::new(ctx, "computed".try_into().unwrap(), i32_ty);
    module.append_operation(ctx, with_region.get_operation(), 0);
    let init_region = with_region.add_initializer_region(ctx);
    let init_block = init_region.deref(ctx).get_head().unwrap();
    let value = ConstantOp::new(ctx, Box::new(int_attr(ctx, 32, 3)));
    value.get_operation().insert_at_back(init_block, ctx);
    let ret = ReturnOp::new(ctx, Some(value.get_result(ctx)));
    ret.get_operation().insert_at_back(init_block, ctx);

    // A declaration: no initializer at all.
    let declared = GlobalOp::new(ctx, "elsewhere".try_into().unwrap(), i32_ty);
    module.append_operation(ctx, declared.get_operation(), 0);

    // And something that refers to one of them.
    let void_ty = VoidType::get(ctx).to_handle();
    let func_ty = FuncType::get(ctx, void_ty, vec![], false);
    let func = FuncOp::new(ctx, "use".try_into().unwrap(), func_ty);
    module.append_operation(ctx, func.get_operation(), 0);
    let entry = func.get_or_create_entry_block(ctx);
    let addr = AddressOfOp::new(ctx, "counter".try_into().unwrap(), 0);
    addr.get_operation().insert_at_back(entry, ctx);
    ReturnOp::new(ctx, None)
        .get_operation()
        .insert_at_back(entry, ctx);

    let printed = common::to_mlir(ctx, &module.get_operation());
    expect![[r#"
        "builtin.module"() <{sym_name = "m"}> ({
          ^block1v1:
            "llvm.mlir.global"() <{addr_space = 0 : i32, global_type = i32, linkage = #llvm.linkage<internal>, sym_name = "counter", alignment = 4 : i64, value = 7 : i32}> ({}) : () -> ()
            "llvm.mlir.global"() <{addr_space = 0 : i32, global_type = i32, linkage = #llvm.linkage<external>, sym_name = "computed"}> ({
              ^entry_block2v1:
                %v0 = "llvm.mlir.constant"() <{value = 3 : i32}> : () -> (i32)
                "llvm.return"(%v0) : (i32) -> ()
            }) : () -> ()
            "llvm.mlir.global"() <{addr_space = 0 : i32, global_type = i32, linkage = #llvm.linkage<external>, sym_name = "elsewhere"}> ({}) : () -> ()
            "llvm.func"() <{function_type = !llvm.func<!llvm.void ()>, linkage = #llvm.linkage<external>, sym_name = "use"}> ({
              ^entry_block3v1:
                %v1 = "llvm.mlir.addressof"() <{global_name = @counter}> : () -> (!llvm.ptr)
                "llvm.return"() : () -> ()
            }) : () -> ()
        }) : () -> ()"#]].assert_eq(&printed);
    common::expect_mlir_opt_output(
        &printed,
        expect![[r#"
        module @m {
          llvm.mlir.global internal @counter(7 : i32) {addr_space = 0 : i32, alignment = 4 : i64} : i32
          llvm.mlir.global external @computed() {addr_space = 0 : i32} : i32 {
            %0 = llvm.mlir.constant(3 : i32) : i32
            llvm.return %0 : i32
          }
          llvm.mlir.global external @elsewhere() {addr_space = 0 : i32} : i32
          llvm.func @use() {
            %0 = llvm.mlir.addressof @counter : !llvm.ptr
            llvm.return
          }
        }

    "#]],
    );
}

/// A `llvm.func` with no body is a declaration in both dialects.
#[test]
fn function_declaration() {
    let ctx = &mut Context::new();
    let i32_ty = int_ty(ctx, 32);
    let module = ModuleOp::new(ctx, "m".try_into().unwrap());
    let func_ty = FuncType::get(ctx, i32_ty, vec![i32_ty], false);
    let decl = FuncOp::new(ctx, "declared".try_into().unwrap(), func_ty);
    decl.set_attr_llvm_function_linkage(ctx, LinkageAttr::ExternalWeakLinkage);
    module.append_operation(ctx, decl.get_operation(), 0);
    let _ = decl.get_symbol_name(ctx);

    let printed = common::to_mlir(ctx, &module.get_operation());
    expect![[r#"
        "builtin.module"() <{sym_name = "m"}> ({
          ^block1v1:
            "llvm.func"() <{function_type = !llvm.func<i32 (i32)>, linkage = #llvm.linkage<extern_weak>, sym_name = "declared"}> ({}) : () -> ()
        }) : () -> ()"#]].assert_eq(&printed);
    common::expect_mlir_opt_output(
        &printed,
        expect![[r#"
        module @m {
          llvm.func extern_weak @declared(i32) -> i32
        }

    "#]],
    );
}
