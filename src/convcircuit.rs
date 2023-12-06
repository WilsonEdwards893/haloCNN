use halo2_proofs::{circuit::{Value, SimpleFloorPlanner}, plonk::{Circuit, ConstraintSystem},arithmetic::Field};
use crate::matrix::Matrix;
use halo2_test::convchip::{ConvChip, ConvChipConfig};
/// The full circuit implementation.
///
/// In this struct we store the private input variables. We use `Option<F>` because
/// they won't have any value during key generation. During proving, if any of these
/// were `None` we would get an error.
#[derive(Default)]
struct ConvCircuit<F: Field> {
    constant: F,
    input:Matrix<Value<F>>,
    kernel: Matrix<Value<F>>,
    bias: Vec<Value<F>>,
}

