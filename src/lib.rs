// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron-mlir-interop contributors

//! Pliron -> Textual MLIR
//!
//! Translation is interfaces driven. Every [Op], [Type] and [Attribute]
//! in the IR being translated must implement [ToMlirOp], [ToMlirType] and
//! [ToMlirAttr] respectively.
//!
//! The primary entry point is [op_to_mlir_string].
//!
//! If you don't want to build an entire [String] and want to implement your own
//! [Display](std::fmt::Display) object, you can use the functions in [printers].

use crate::printers::{print_generic_op, print_op};
use pliron::{
    attribute::Attribute,
    context::{Context, Ptr},
    derive::{attr_interface, op_interface, type_interface},
    op::Op,
    operation::Operation,
    printable::State,
    result::{Error as PlironError, Result},
    r#type::Type,
};
use std::{cell::Cell, fmt};
use thiserror::Error;

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
/// with `Self::get_opid()` as the mnemonic. Override [Self::to_mlir] as necessary.
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

/// Translate an [Operation] (and everything nested within it) to MLIR text.
///
/// This is a wrapper around `print_op` that converts errors suitably.
pub fn op_to_mlir_string(ctx: &Context, op: Ptr<Operation>) -> Result<String> {
    use fmt::Write as _;

    struct Printer<'a> {
        ctx: &'a Context,
        op: Ptr<Operation>,
        state: State,
        error: Cell<Option<PlironError>>,
    }

    impl fmt::Display for Printer<'_> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            print_op(self.ctx, self.op, &self.state, f).map_err(|e| {
                self.error.set(Some(e));
                fmt::Error
            })
        }
    }

    let printer = Printer {
        ctx,
        op,
        state: State::default(),
        error: Cell::new(None),
    };

    let mut buf = String::new();
    match write!(buf, "{printer}") {
        Ok(()) => Ok(buf),
        Err(_) => Err(printer
            .error
            .into_inner()
            .expect("Printer::fmt failed without recording an error")),
    }
}
