use std::{marker::PhantomData};
//stride = 1, padding = 0
use halo2_proofs::{
    arithmetic::Field,
    circuit::{layouter::TableLayouter, Cell, Value, Chip, Layouter, AssignedCell},
    dev::{MockProver, VerifyFailure},
    pasta::Fp,
    plonk::{
        Advice, Assignment, Circuit, Column, ConstraintSystem, Error, Expression, Fixed,
        Selector, Instance
    },
    poly::Rotation,
};

pub use crate::matrix::Matrix;
use crate::matrix::{rows, shape};

trait ConvInstructions<F: Field>: Chip<F> {
    type Num;
    // Loads input
    fn load_matrix(&self, layouter: impl Layouter<F>, index:usize, input: &Matrix<Value<F>>)-> Result<Matrix<Value<F>>, Error>;

    // Loads kernel and bias matrix
    fn load_bias(&self, layouter: impl Layouter<F>, bias: Vec<Value<F>>) -> Result<(), Error>;

    // Returns `ouput = input * kernel + bias`.
    fn conv(
        &self,
        layouter: impl Layouter<F>,
        input: Matrix<Value<F>>,
        kernel: Matrix<Value<F>>,
        bias: Vec<Value<F>>,
    ) -> Result<Matrix<Value<F>>, Error>;

    // Exposes a matrix as a public input to the circuit.
    fn expose_public(
        &self,
        layouter: impl Layouter<F>,
        ouput: Matrix<F>,
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
        n: usize //shape of kernel
    ) -> <Self as Chip<F>>::Config {
        meta.enable_equality(instance);
        meta.enable_constant(constant);
        for column in &advice {
            meta.enable_equality(*column);
        }
        let s_conv = meta.selector();
        
        meta.create_gate("conv", |meta| {
            // Query the selector
            let s_conv = meta.query_selector(s_conv);

            // Query the variables from the columns
            let x = meta.query_advice(advice[0], Rotation::cur()); // Input matrix
            let w = meta.query_advice(advice[1], Rotation::cur()); // Kernel matrix
            let b = meta.query_advice(advice[2], Rotation::cur()); // Bias vector
            let y = meta.query_instance(instance, Rotation::cur()); // Output matrix

            // Construct the expressions for the convolution formula and constraint
            // y[i,j] = sum(x[i+k,j+l] * w[k,l] + b[i,j]) for k,l in [0,n-1]
            let mut res = b;

            for k in 0..n {
                for l in 0..n {
                    // value of x[i+k,j+l]
                    let x_ikjl = meta.query_advice(
                        advice[0],
                        Rotation((k * n + l) as i32),
                    );
                    // value of w[k,l]
                    let w_kl = meta.query_advice(
                        advice[1],
                        Rotation((k * n + l) as i32),
                    );
                    // sum x[i+k,j+l] * w[k,l]
                    res = res + x_ikjl * w_kl;
                }
            }

            // Return the polynomial constraint
            // If s_conv is enabled, then y[i,j]-sum(x[i+k,j+l] * w[k,l] + b[i,j]) must be zero, otherwise it can be anything
            vec![s_conv * (y - res)]
        });

        ConvChipConfig {
            advice,
            instance,
            constant,
            s_conv,
        }
    }

}

/// A variable representing a number.
#[derive(Clone)]
struct Number<F: Field>(AssignedCell<F, F>);

impl<F: Field> ConvInstructions<F> for ConvChip<F> {
    type Num = Number<F>;

    // load matrix
    fn load_matrix(&self, mut layouter: impl Layouter<F>, index: usize, input: &Matrix<Value<F>>)-> Result<Matrix<Value<F>>, Error> {
        
        let config = self.config();

        // acquire the row and column of a matrix
        let (row, col) = shape(input);
        
        // create a vector for assigned vallue
        let mut values = Vec::new();

         // assign input values to the corresponding advice columns
        let _ = layouter.assign_region(
            || "load input",
            |mut region| {
                // 遍历每一行
                for i in 0..row {
                    // 遍历每一列
                    for j in 0..col {
                        // 获取 input[i][j] 的值
                        let value = input[i][j];
                        // 分配 value 到当前单元格
                        let _ = region
                        .assign_advice(
                            || format!("input[{}][{}]", i, j),
                            config.advice[index],
                            i * col + j, // offset of current cell
                            || value,
                        );

                        values.push(value);
                    }
                }
                Ok(())
            },
        );
        // turn vector to a matrix
        let matrix = values
        .chunks(col)
        .map(|chunk| chunk.to_vec())
        .collect::<Matrix<Value<F>>>();

        // return matrix
        Ok(matrix)
    }

    fn load_bias(&self, mut layouter: impl Layouter<F>, bias: Vec<Value<F>>) -> Result<(), Error> {
        let config = self.config();

        
    }

    fn conv(
        &self,
        layouter: impl Layouter<F>,
        input: Matrix<Value<F>>,
        kernel: Matrix<Value<F>>,
        bias: Vec<Value<F>>,
    ) -> Result<Matrix<Value<F>>, Error> {
        let config = self.config();


    }

    fn expose_public(
        &self,
        layouter: impl Layouter<F>,
        ouput: Matrix<F>,
        row: usize,
    ) -> Result<(), Error> {
        let config = self.config();

        layouter.constrain_instance(num.0.cell(), config.instance, row)
    }

}