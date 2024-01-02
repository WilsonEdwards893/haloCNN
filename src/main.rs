mod testcircuit;
mod matrix;

use std::env;

use testcircuit::{ConvCircuit, run_conv_test};
use halo2_proofs::{
    arithmetic::Field,
    circuit::{layouter::TableLayouter, Cell, Value, Chip, Layouter},
    dev::{MockProver, VerifyFailure},
    pasta::Fp,
    plonk::{
        Advice, Assignment, Circuit, Column, ConstraintSystem, Error, Expression, Fixed,
        Selector
    },
    poly::Rotation,
};

mod convcircuit;

fn main() {
    env::set_var("RUST_BACKTRACE", "full");
    run_conv_test(1);
    
}