mod matrix;
use std::env;

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
use halo2_test::lenet5;
use lenet5::run_lenet5_test;

fn main() {
    env::set_var("RUST_BACKTRACE", "1");
    //run_conv_test(1, 1);
    run_lenet5_test();
}