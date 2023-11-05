use std::{marker::PhantomData, os::windows::prelude::FileExt};

//stride = 1, padding = 0
use halo2_proofs::{
    arithmetic::Field,
    circuit::{layouter::TableLayouter, Cell, Value, Chip, Layouter},
    dev::{MockProver, VerifyFailure},
    pasta::Fp,
    plonk::{
        Advice, Assignment, Circuit, Column, ConstraintSystem, Error, Expression, Fixed,
        Selector, Instance
    },
    poly::Rotation,
};

use Matrix::Matrix;
trait ConvInstructions<F: Field>: Chip<F> {
    // Use a vector to represent a matrix
    type Matrix;
    // Loads input
    fn load_input(&self, layouter: impl Layouter<F>, input: Self::Matrix);

    // Loads kernel and bias matrix
    fn load_param(&self, layouter: impl Layouter<F>, kernel: Self::Matrix, bias: Self::Matrix) -> Result<(), Error>;

    // Returns `ouput = input * kernel + bias`.
    fn conv(
        &self,
        layouter: impl Layouter<F>,
        input: Self::Matrix,
        kernel: Self::Matrix,
        bias: Self::Matrix,
    ) -> Result<Self::Matrix, Error>;

    /// Exposes a matrix as a public input to the circuit.
    fn expose_public(
        &self,
        layouter: impl Layouter<F>,
        ouput: Self::Matrix,
        row: usize,
    ) -> Result<(), Error>;
}

/// The chip that will implement our instructions! Chips store their own
/// config, as well as type markers if necessary.
struct ConvChip<F: Field> {
    config: ConvChipConfig,
    _marker: PhantomData<F>,
}

impl<F: Field> Chip<F> for ConvChip<F> {
    type Config = ConvChipConfig;
    type Loaded = ();

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn loaded(&self) -> &Self::Loaded {
        &()
    }
}

#[derive(Clone, Debug)]
struct ConvChipConfig {
    //input, kernel and bias
    advice: [Column<Advice>; 3],
    // This is the public input (instance) column.
    instance: Column<Instance>,
    constant: Column<Fixed>,
    s_conv: Selector,
}

impl<F: Field> ConvChip<F> {
    fn construct(config: <Self as Chip<F>>::Config) -> Self {
        Self {
            config,
            _marker: PhantomData,
        }
    }

    fn configure(
        meta: &mut ConstraintSystem<F>,
        advice: [Column<Advice>; 3],
        instance: Column<Instance>,
        constant: Column<Fixed>,
    ) -> <Self as Chip<F>>::Config {
        // Create an instance column for output
        let instance = meta.instance_column();

        // Create a fixed column for constants
        let constant = meta.fixed_column();

        let s_conv = meta.selector();

        meta.create_gate("conv", |meta| {

            // Query the variables from the columns
            let x = meta.query_advice(advice[0], Rotation::cur()); // Input image
            let w = meta.query_advice(advice[1], Rotation::cur()); // Kernel matrix
            let b = meta.query_advice(advice[2], Rotation::cur()); // Bias vector
            let y = meta.query_instance(instance, Rotation::cur()); // Output image
            let one = meta.query_fixed(constant); // Constant one

            // Query the selector
            let s_conv = meta.query_selector(s_conv);

            // Construct the expressions for the convolution formula and constraint
            // y[i,j] = sum(x[i+k,j+l] * w[k,l] + b[i,j]) for k,l in [0,n-1]
            let xw = x * w; // x[i+k,j+l] * w[k,l]
            let rp = RunningProduct::new(meta, xw); // Running product of xw
            let sum = rp.product() + b; // sum(x[i+k,j+l] * w[k,l]) + b[i,j]
            let res = sum - y; // sum(x[i+k,j+l] * w[k,l]) + b[i,j] - y[i,j]

            // Return the polynomial constraint
            // If s_conv is enabled, then res must be zero, otherwise it can be anything
            vec![s_conv * res]
        });

        ConvChipConfig {
            advice,
            instance,
            constant,
            s_conv,
        }
    }
}

impl<F: Field> ConvInstructions<F> for ConvChip<F> {
 // Specify the associated type
 type Matrix = Vec<Vec<F>>;

}