// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron-mlir-interop contributors

//! # Pliron -> Textual MLIR
//!
//! Translation is interface-driven. Every `Op`, `Type` and `Attribute`
//! in the IR being translated must implement [ToMlirOp], [ToMlirType] and
//! [ToMlirAttr] respectively.
//!
//! The primary entry point is [`MlirPrinter`], which can be built with
//! `Ptr<Operation>`, `TypeHandle` or `dyn Attribute`.
//! [`MlirPrinter`] implements [Display], which is the final path to printing
//! the output MLIR text.
//!
//! ## An Example
//!
//! Printing an empty [`ModuleOp`](pliron::builtin::ops::ModuleOp) to MLIR text:
//!
//! ```
//! use pliron::{builtin::ops::ModuleOp, context::Context, op::Op};
//! use pliron_mlir_interop::MlirPrinter;
//!
//! let ctx = &mut Context::new();
//! let module = ModuleOp::new(ctx, "a_module".try_into().unwrap());
//!
//! let printed = MlirPrinter::new(ctx, &module.get_operation()).to_string();
//! assert_eq!(
//!     printed,
//!     r#""builtin.module"() <{sym_name = "a_module"}> ({
//!   ^block1v1:
//! }) : () -> ()"#
//! );
//! ```
//!
//! See [`MlirPrinter`] documentation for handling translation errors.
//!
//! ## Printing an Op: Control vs Convenience
//!
//! By manually implementing [ToMlirOp], dialect authors have full control over
//! how the MLIR equivalent must be printed.
//!
//! To simplify common cases, two convenience utilities are provided:
//!
//! 1. The default implementation of [ToMlirOp] just calls [print_generic_op] with
//!    `Op::get_opid()` as the mnemonic. This however may not be sufficient,
//!    especially when attributes need translating to MLIR's properties or
//!    discardable attributes, or when their keys differ from their MLIR equivalent.
//! 2. The [GenericOp](printers::GenericOp) builder, which allows customizing certain
//!    aspects of the translation / printing.

use crate::printers::{print_attr, print_generic_op, print_op, print_type};
use pliron::{
    attribute::Attribute,
    context::{Context, Ptr},
    derive::{attr_interface, op_interface, type_interface},
    op::Op,
    operation::Operation,
    printable::State,
    result::{Error as PlironError, Result},
    r#type::{Type, TypeHandle},
};
use std::{
    cell::Cell,
    fmt::{self, Display},
};
use thiserror::Error;

pub mod builtin;
pub mod llvm;
pub mod printers;

#[derive(Debug, Error)]
pub enum Error {
    #[error("Error formatting to MLIR text")]
    Fmt(#[from] fmt::Error),
    #[error("Op `{0}` does not implement ToMlirOp")]
    MissingOpTranslation(String),
    #[error("Type `{0}` does not implement ToMlirType")]
    MissingTypeTranslation(String),
    #[error("Attribute `{0}` does not implement ToMlirAttr")]
    MissingAttrTranslation(String),
    #[error("Cannot translate to MLIR: {0}")]
    Untranslatable(String),
}

/// A pliron [Type] that can be converted to MLIR.
#[type_interface]
pub trait ToMlirType {
    /// Print `self` as MLIR type syntax (e.g. `i32`, `!llvm.ptr`).
    fn to_mlir(&self, ctx: &Context, state: &State, f: &mut fmt::Formatter<'_>) -> Result<()>;

    fn verify(_ty: &dyn Type, _ctx: &Context) -> Result<()>
    where
        Self: Sized,
    {
        Ok(())
    }
}

/// A pliron [Attribute] that can be converted to MLIR.
#[attr_interface]
pub trait ToMlirAttr {
    /// Print `self` as MLIR attribute syntax (e.g. `42 : i64`, `"foo"`).
    fn to_mlir(&self, ctx: &Context, state: &State, f: &mut fmt::Formatter<'_>) -> Result<()>;

    fn verify(_attr: &dyn Attribute, _ctx: &Context) -> Result<()>
    where
        Self: Sized,
    {
        Ok(())
    }
}

/// A pliron [Op] that can be converted to MLIR.
///
/// The default implementation calls [print_generic_op]
/// with `Self::get_opid()` as the mnemonic.
///
/// A manual implementation (override) will typically use
/// [GenericOp](printers::GenericOp).
#[op_interface]
pub trait ToMlirOp {
    /// Print `self` (including any nested regions) as MLIR operation syntax.
    fn to_mlir(&self, ctx: &Context, state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        print_generic_op(
            ctx,
            self.get_operation(),
            &self.get_opid().to_string(),
            state,
            f,
        )
    }

    fn verify(_op: &dyn Op, _ctx: &Context) -> Result<()>
    where
        Self: Sized,
    {
        Ok(())
    }
}

/// IR entities that can be printed to MLIR text.
pub trait MlirPrinterT {
    fn mlir_print(&self, ctx: &Context, state: &State, f: &mut fmt::Formatter<'_>) -> Result<()>;
}

/// A convenience type that implements [Display] for [MlirPrinterT].
///
/// Upon failure, the error is stored internally and can be consumed using [Self::take_error].
///
/// Example:
///
/// ```
/// use pliron::{context::Context, input_err_noloc, printable::{State, Printable}, result::Result};
/// use pliron_mlir_interop::{MlirPrinter, MlirPrinterT};
/// use std::fmt::{self, Write};
///
/// /// An demo type that implements [MlirPrinterT].
/// struct MyEntity(bool);
///
/// impl MlirPrinterT for MyEntity {
///     fn mlir_print(
///         &self,
///         _ctx: &Context,
///         _state: &State,
///         f: &mut fmt::Formatter<'_>,
///     ) -> Result<()> {
///        if self.0 {
///            write!(f, "my.entity")?;
///        } else {
///            return input_err_noloc!("Some error");
///        }
///        Ok(())
///     }
/// }
///
/// let ctx = Context::new();
/// let printer = MlirPrinter::new(&ctx, &MyEntity(true));
///
/// let mut out = String::new();
/// match write!(&mut out, "{printer}") {
///    Ok(()) => {
///        println!("{out}");
///    }
///    Err(fmt::Error) => {
///        // Get the real error from the printer.
///        let err = printer.take_error().expect("Printing failed, so an error must be set");
///        eprintln!("{}", err.disp(&ctx));
///    }
/// }
/// ```
pub struct MlirPrinter<'a, T: MlirPrinterT + ?Sized> {
    entity: &'a T,
    ctx: &'a Context,
    state: State,
    error: Cell<Option<PlironError>>,
}

impl<'a, T: MlirPrinterT + ?Sized> MlirPrinter<'a, T> {
    /// Create a new [MlirPrinter]
    pub fn new(ctx: &'a Context, entity: &'a T) -> Self {
        Self {
            entity,
            ctx,
            state: State::default(),
            error: Cell::new(None),
        }
    }

    /// If there was a failure, consume the error.
    pub fn take_error(&self) -> Option<PlironError> {
        self.error.take()
    }
}

impl<'a, T: MlirPrinterT + ?Sized> Display for MlirPrinter<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.entity
            .mlir_print(self.ctx, &self.state, f)
            .map_err(|e| {
                self.error.set(Some(e));
                fmt::Error
            })
    }
}

impl MlirPrinterT for Ptr<Operation> {
    fn mlir_print(&self, ctx: &Context, state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        print_op(ctx, *self, state, f)
    }
}

impl MlirPrinterT for TypeHandle {
    fn mlir_print(&self, ctx: &Context, state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        print_type(ctx, *self, state, f)
    }
}

impl MlirPrinterT for dyn Attribute {
    fn mlir_print(&self, ctx: &Context, state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        print_attr(ctx, self, state, f)
    }
}
