mod circuit;
mod matrix;

use circuit::{ConvCircuit, run_conv_test};
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


fn main() {
    
    run_conv_test(1);
    
}