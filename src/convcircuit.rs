use halo2_proofs::{circuit::{Value, SimpleFloorPlanner}, plonk::{Circuit, ConstraintSystem, Error},arithmetic::Field};
use crate::matrix::{Matrix, shape};
use halo2_test::convchip::{ConvChip, ConvChipConfig, ConvInstructions,self};
/// The full circuit implementation.
///
/// In this struct we store the private input variables. We use `Option<F>` because
/// they won't have any value during key generation. During proving, if any of these
/// were `None` we would get an error.
#[derive(Default)]
struct ConvCircuit<F: Field> {
    constant: F,
    input:Matrix<Value<F>>,
    filter: Matrix<Value<F>>,
    bias: Vec<Value<F>>,
}

impl<F: Field> Circuit<F> for ConvCircuit<F> {
    // the config of Convolution layer
    type Config = ConvChipConfig;

    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        // We create the four advice columns that ConvChip uses for I/O.
        let advice = [meta.advice_column();4];

        // We also need an instance column to store public inputs.
        let instance = meta.instance_column();
 
        // Create a fixed column to load constants.
        let constant = meta.fixed_column();
        
        ConvChip::configure(meta, advice, instance, constant, 100)
    }

    fn synthesize(&self, 
        config: Self::Config, 
        mut layouter: impl halo2_proofs::circuit::Layouter<F>) -> Result<(), Error> {

        let conv_chip = ConvChip::<F>::construct(config);
        // Load our private values into the circuit.
        let input = conv_chip.load_matrix(layouter.namespace(|| "load input"), 0, &self.input)?;
        let kernel = conv_chip.load_matrix(layouter.namespace(|| "load filter"), 1, &self.filter)?;
        let bias = conv_chip.load_bias(layouter.namespace(|| "load bias"), &self.bias)?;
        
        let output = conv_chip.conv(layouter.namespace(|| "conv operation"), input, kernel, bias)?;
        
        // Expose the result as a public input to the circuit.
        conv_chip.expose_public(layouter.namespace(|| "expose output"), output)
    }
}