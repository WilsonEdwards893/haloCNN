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

// followed by the example at https://zcash.github.io/halo2/user/simple-example.html
trait ConvInstructions<F: Field>: Chip<F> {
    type Num;
    // Loads input
    fn load_matrix(&self, layouter: impl Layouter<F>, index:usize, input: &Matrix<Value<F>>)-> Result<Matrix<Self::Num>, Error>;

    // Loads kernel and bias matrix
    fn load_bias(&self, layouter: impl Layouter<F>, bias: Vec<Value<F>>) -> Result<Vec<Self::Num>, Error>;

    // Returns `ouput = input * kernel + bias`.
    fn conv(
        &self,
        layouter: impl Layouter<F>,
        input: Matrix<Self::Num>,
        kernel: Matrix<Self::Num>,
        bias: Vec<Self::Num>,
    ) -> Result<Matrix<Self::Num>, Error>;

    // Exposes a matrix as a public input to the circuit.
    fn expose_public(
        &self,
        layouter: impl Layouter<F>,
        ouput: Matrix<Self::Num>
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
    advice: [Column<Advice>; 4],
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
        advice: [Column<Advice>; 4],
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
    fn load_matrix(&self, mut layouter: impl Layouter<F>, index: usize, input: &Matrix<Value<F>>)-> Result<Matrix<Self::Num>, Error> {
        
        let config = self.config();

        // acquire the row and column of a matrix
        let (row, col) = shape(input);
        
        // create a vector for assigned vallue
        let mut values = Vec::new();

        // assign input values to the corresponding advice columns
        let _ = layouter.assign_region(
            || "load input",
            |mut region| {
                // iterate over row
                for i in 0..row {
                    // iterate over column
                    for j in 0..col {
                        // acquire value of input[i][j]
                        let value = input[i][j];
                        // assgin value to current cell
                        let ret = region
                        .assign_advice(
                            || format!("input[{}][{}]", i, j),
                            config.advice[index],
                            i * col + j, // offset of current cell
                            || value,
                        ).map(Number)
                        ;

                        values.push(ret.unwrap());
                    }
                }
                Ok(())
            },
        );
        // turn vector to a matrix
        let matrix = values
        .chunks(col)
        .map(|chunk| chunk.to_vec())
        .collect::<Matrix<Self::Num>>();

        // return matrix
        Ok(matrix)
    }

    fn load_bias(&self, mut layouter: impl Layouter<F>, bias: Vec<Value<F>>) -> Result<Vec<Self::Num>, Error> {
        let config = self.config();

        // acquire the length of the bias vector
        let len = bias.len();

        let mut values = Vec::new();

        // assign bias values to the corresponding advice columns
        let _ = layouter.assign_region(
            || "load bias",
            |mut region| {
                // iterate over the bias vector
                for i in 0..len {
                    // get the bias[i] value
                    let value = bias[i];
                    // assign value to the current cell
                    let ret = region
                    .assign_advice(
                        || format!("bias[{}]", i),
                        config.advice[2], // use the third advice column
                        i, // offset of current cell
                        || value,
                    ).map(Number);
                    // add the assigned value to the vector
                    values.push(ret.unwrap());
                }
                Ok(())
            },
        );
        Ok(values)
    }

    fn conv(
        &self,
        mut layouter: impl Layouter<F>,
        input: Matrix<Self::Num>,
        kernel: Matrix<Self::Num>,
        bias: Vec<Self::Num>
    ) -> Result<Matrix<Self::Num>, Error> {
        let config = self.config();

        // acquire the shape of the input and kernel matrices
        let (input_row, input_col) = shape(&input);
        let (kernel_row, kernel_col) = shape(&kernel);

        // check if the bias vector has the same length as the kernel row
        // if bias.len() != kernel_row {
            //return Err(Error::Synthesis);
        // }

        let output_row = input_row - kernel_row + 1;
        let output_col = input_col - kernel_col + 1;
        // create a vector for the output values
        let mut values = Vec::new();

        // assign output values to the corresponding advice columns
        let _ = layouter.assign_region(
            || "assign output",
            |mut region| {
                // iterate over the input matrix with a sliding window of the kernel size
                for i in 0..(input_row - kernel_row + 1) {
                    for j in 0..(input_col - kernel_col + 1) {
                        // create a vector for the dot product values
                        let mut dot_products = Vec::new();

                        // iterate over the kernel matrix
                        for k in 0..kernel_row {
                            for l in 0..kernel_col {
                                // get the input[i+k][j+l] and kernel[k][l] values
                                let input_value = &input[i+k][j+l];
                                let kernel_value = &kernel[k][l];

                                // multiply the input and kernel values and add them to the dot product vector
                                let dot_product = input_value.0.value().cloned() * kernel_value.0.value();
                                dot_products.push(dot_product);
                            }
                        }

                        // sum up the dot product values
                        let sum = dot_products.iter().fold(Value::known(Field::ZERO), |acc, x| acc + x);

                        // add the bias value corresponding to the current channel
                        let output_value = sum + bias[0].0.value();

                        // assign output value to the current cell
                        let ret = region
                        .assign_advice(
                            || format!("output[{}][{}]", i, j),
                            config.advice[3], // use the third advice column
                            i * output_col + j, // offset of current cell
                            || output_value,
                        ).map(Number);

                         // add the output value to the output vector
                         values.push(ret.unwrap());
                    }
                }
                Ok(())
            },
        );

    // turn the output vector into a matrix
    let output_matrix = values
    .chunks(output_col)
    .map(|chunk| chunk.to_vec())
    .collect::<Matrix<Self::Num>>();
    // return the output matrix
    Ok(output_matrix)

}

    fn expose_public(
        &self,
        mut layouter: impl Layouter<F>,
        input: Matrix<Self::Num>,
    ) -> Result<(), Error> {
        let config = self.config();
        // acquire the shape of the output matrix
        let (input_row, input_col) = shape(&input);
         // iterate over the output matrix
        for i in 0..input_row {
            for j in 0..input_col {
                // get the output[i][j] value
                let value = &input[i][j];
                // constrain value to the current cell
                let _ = layouter.constrain_instance(
                    value.0.cell(),
                    config.instance, // use the j-th instance column
                    i * input_row + j, // offset of current cell
                );
            }
        }
        Ok(())
    }

}