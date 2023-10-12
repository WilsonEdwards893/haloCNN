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

trait ConvInstructions<F: Field>: Chip<F> {
    type Num;
    // Use a vector to represent a matrix
    type Matrix;
    // Loads input
    fn load_input(&self, layouter: impl Layouter<F>, input: Matrix);

    // Loads kernel and bias matrix
    fn load_param(&self, layouter: impl Layouter<F>, kernel: Matrix, bias: Matrix) -> Result<(), Error>;

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
    config: ConvConfig,
    _marker: PhantomData<F>,
}

impl<F: Field> Chip<F> for ConvChip<F> {
    type Config = ConvConfig;
    type Loaded = ();

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn loaded(&self) -> &Self::Loaded {
        &()
    }
}

