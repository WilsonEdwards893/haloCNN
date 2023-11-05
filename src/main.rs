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

trait MatrixInstructions<F: Field>: Chip<F> {
    /// Variable representing a matrix.
    type Matrix;
    // Variable representing a cell.
    type Cell;
    // Loads a matrix into the circuit as a private input.
    fn load_private(&self, layouter: impl Layouter<F>, m: Vec<Vec<Value<F>>>) -> Result<Self::Matrix, Error>;
    // Loads a matrix into the circuit as a public input.
    fn load_public(&self, layouter: impl Layouter<F>, m: Vec<Vec<Value<F>>>) -> Result<Self::Matrix, Error>;
    // Converts a matrix into a vector.
    fn to_vector(&self, layouter: impl Layouter<F>, m: Self::Matrix) -> Result<Vec<Self::Cell>, Error>;
    // Converts a vector into a matrix.
    fn to_matrix(&self, layouter: impl Layouter<F>, v: Vec<Self::Cell>) -> Result<Self::Matrix, Error>;
    // Returns the maximum value of a vector.
    fn max(&self, layouter: impl Layouter<F>, v: Vec<Self::Cell>) -> Result<Self::Cell, Error>;
}

pub struct MatrixConfig {
    // The column where the matrix elements are stored.
    pub element_col: Column<Advice>,
    // The column where the matrix indices are stored (optional).
    pub index_col: Option<Column<Fixed>>,
    // The selector for the conversion gate.
    pub conversion_selector: Selector,
    // The selector for the max gate.
    pub max_selector: Selector,
    // The selector for the output gate.
    pub output_selector: Selector,
}

pub struct MatrixLoaded {
    // The copy constraint for the element column.
    pub element_copy: Copy,
}

pub struct MatrixChip<F: Field> {
    config: MatrixConfig,
}

impl<F: Field> Chip<F> for MatrixChip<F> {
    type Config = MatrixConfig;
    type Loaded = MatrixLoaded;
    
    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn loaded(&self) -> &Self::Loaded {
        &()
    }
}



fn main() {
    
}