// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron-mlir-interop contributors

//! Pliron -> textual MLIR printers.

use std::fmt;

use pliron::{
    attribute::{Attribute, attr_cast},
    basic_block::BasicBlock,
    common_traits::Named,
    context::{Context, Ptr},
    input_err, input_err_noloc,
    linked_list::ContainsLinkedList,
    location::Located,
    op::op_cast,
    operation::Operation,
    printable::State,
    region::Region,
    result::Result,
    r#type::{TypeHandle, Typed, type_cast},
};

use crate::{Error, ToMlirAttr, ToMlirOp, ToMlirType};

/// Dispatch to `op`'s [ToMlirOp] implementation.
///
/// Fails if `op`'s concrete [pliron::op::Op] type does not implement [ToMlirOp].
pub fn print_op(
    ctx: &Context,
    op: Ptr<Operation>,
    state: &State,
    f: &mut fmt::Formatter<'_>,
) -> Result<()> {
    let op_obj = Operation::get_op_dyn(op, ctx);
    let Some(conv) = op_cast::<dyn ToMlirOp>(op_obj.as_ref()) else {
        let loc = op.deref(ctx).loc();
        return input_err!(
            loc,
            Error::MissingOpTranslation(op_obj.get_opid().to_string())
        );
    };
    conv.to_mlir(ctx, state, f)
}

/// Dispatch to `ty`'s [ToMlirType] implementation.
///
/// Fails if `ty`'s concrete [pliron::type::Type] does not implement [ToMlirType].
pub fn print_type(
    ctx: &Context,
    ty: TypeHandle,
    state: &State,
    f: &mut fmt::Formatter<'_>,
) -> Result<()> {
    let ty_ref = ty.deref(ctx);
    let Some(conv) = type_cast::<dyn ToMlirType>(&*ty_ref) else {
        return input_err_noloc!(Error::MissingTypeTranslation(
            ty_ref.get_type_id().to_string()
        ));
    };
    conv.to_mlir(ctx, state, f)
}

/// Dispatch to `attr`'s [ToMlirAttr] implementation.
///
/// Fails if `attr`'s concrete [Attribute] does not implement [ToMlirAttr].
pub fn print_attr(
    ctx: &Context,
    attr: &dyn Attribute,
    state: &State,
    f: &mut fmt::Formatter<'_>,
) -> Result<()> {
    let Some(conv) = attr_cast::<dyn ToMlirAttr>(attr) else {
        return input_err_noloc!(Error::MissingAttrTranslation(
            attr.get_attr_id().to_string()
        ));
    };
    conv.to_mlir(ctx, state, f)
}

/// Print `op` using MLIR's generic operation syntax:
///
/// ```text
/// operation         ::= op-result-list? generic-operation
/// generic-operation ::= string-literal `(` value-use-list? `)` successor-list?
///                       region-list? dictionary-attribute?
///                       `:` function-type
/// ```
///
/// e.g.,
///
/// ```text
/// %r0, %r1 = "mnemonic"(%o0, %o1) [^s0, ^s1] ({ ... }) {attr = val} : (t0, t1) -> (t2, t3)
/// ```
///
/// Operand/result/successor/region structure and naming come from `op`'s underlying [Operation].
pub fn print_generic_op(
    ctx: &Context,
    op: Ptr<Operation>,
    mnemonic: &str,
    state: &State,
    f: &mut fmt::Formatter<'_>,
) -> Result<()> {
    let operation = op.deref(ctx);

    if operation.get_num_results() > 0 {
        for (i, res) in operation.results().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "%{}", res.unique_name(ctx))?;
        }
        write!(f, " = ")?;
    }

    write!(f, "\"{mnemonic}\"(")?;
    for (i, opd) in operation.operands().enumerate() {
        if i > 0 {
            write!(f, ", ")?;
        }
        write!(f, "%{}", opd.unique_name(ctx))?;
    }
    write!(f, ")")?;

    if operation.get_num_successors() > 0 {
        write!(f, " [")?;
        for (i, succ) in operation.successors().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "^{}", succ.deref(ctx).unique_name(ctx))?;
        }
        write!(f, "]")?;
    }

    if operation.num_regions() > 0 {
        write!(f, " (")?;
        for (i, region) in operation.regions().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            print_region(ctx, region, state, f)?;
        }
        write!(f, ")")?;
    }

    if !operation.attributes.0.is_empty() {
        write!(f, " {{")?;
        for (i, (key, val)) in operation.attributes.0.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{key} = ")?;
            print_attr(ctx, &**val, state, f)?;
        }
        write!(f, "}}")?;
    }

    write!(f, " : (")?;
    for (i, ty) in operation.operand_types(ctx).enumerate() {
        if i > 0 {
            write!(f, ", ")?;
        }
        print_type(ctx, ty, state, f)?;
    }
    write!(f, ") -> (")?;
    for (i, ty) in operation.result_types().enumerate() {
        if i > 0 {
            write!(f, ", ")?;
        }
        print_type(ctx, ty, state, f)?;
    }
    write!(f, ")")?;

    Ok(())
}

/// Print `region` as `{ block* }`, per MLIR's `region ::= '{' block* '}'`.
///
/// Every block (including the entry block) is printed with an explicit label.
fn print_region(
    ctx: &Context,
    region: Ptr<Region>,
    state: &State,
    f: &mut fmt::Formatter<'_>,
) -> Result<()> {
    write!(f, "{{")?;
    state.push_indent();
    for block in region.deref(ctx).iter(ctx) {
        print_newline_indent(state, f)?;
        print_block(ctx, block, state, f)?;
    }
    state.pop_indent();
    print_newline_indent(state, f)?;
    write!(f, "}}")?;
    Ok(())
}

/// Print `block` as `^label(args): operation*`.
fn print_block(
    ctx: &Context,
    block: Ptr<BasicBlock>,
    state: &State,
    f: &mut fmt::Formatter<'_>,
) -> Result<()> {
    {
        let block_ref = block.deref(ctx);
        write!(f, "^{}", block_ref.unique_name(ctx))?;
        if block_ref.get_num_arguments() > 0 {
            write!(f, "(")?;
            for (i, arg) in block_ref.arguments().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "%{}: ", arg.unique_name(ctx))?;
                print_type(ctx, arg.get_type(ctx), state, f)?;
            }
            write!(f, ")")?;
        }
        write!(f, ":")?;
    }

    state.push_indent();
    for op in block.deref(ctx).iter(ctx) {
        print_newline_indent(state, f)?;
        print_op(ctx, op, state, f)?;
    }
    state.pop_indent();
    Ok(())
}

fn print_newline_indent(state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
    write!(
        f,
        "\n{:width$}",
        "",
        width = state.current_indent() as usize
    )?;
    Ok(())
}
