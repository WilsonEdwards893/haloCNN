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

impl<F: Field> Circuit<F> for ConvCircuit<F> {
    type Config = ConvChipConfig;

    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        // We create the two advice columns that FieldChip uses for I/O.
        let advice = [meta.advice_column(), meta.advice_column(), meta.advice_column(), meta.advice_column()];

        // We also need an instance column to store public inputs.
        let instance = meta.instance_column();
        
        // Create a fixed column to load constants.
        let constant = meta.fixed_column();

        // 
        ConvChip::configure(meta, advice, instance, constant, );
    }

    fn synthesize(&self, config: Self::Config, layouter: impl halo2_proofs::circuit::Layouter<F>) -> Result<(), halo2_proofs::plonk::Error> {
        todo!()
    }
}