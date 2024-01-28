use std::{cell, marker::PhantomData};
use ff::PrimeField;
//stride = 1, padding = 0
use halo2_proofs::{
    
    arithmetic::Field, circuit::{AssignedCell, Cell, Chip, Layouter, Value}, pasta::Fp, plonk::{
        Advice, Assigned, Column, ConstraintSystem, Constraints, Error, Expression, Fixed, FixedQuery, Instance, Selector
    }, poly::Rotation
};
use crate::lenet5::LeNetChipConfig;
pub use crate::matrix::Matrix;
use crate::matrix::{rows, shape};

// followed by the example at https://zcash.github.io/halo2/user/simple-example.html
pub trait ConvInstructions<F: Field>: Chip<F> {
    type Num;
    // Loads input from advice column
    fn load_matrix_from_advice(&self, layouter: impl Layouter<F>, index:usize, offset:usize, input: &Matrix<Value<F>>)-> Result<Matrix<Self::Num>, Error>;
    // Loads input from instance column
    fn load_matrix_from_instance(&self, layouter: impl Layouter<F>,  index:usize, offset:usize, input: &Matrix<Value<F>>)-> Result<Matrix<Self::Num>, Error>;
    // Loads kernel and bias matrix
    fn load_bias(&self, layouter: impl Layouter<F>, bias: &Vec<Value<F>>) -> Result<Vec<Self::Num>, Error>;
    // Returns `ouput = input * kernel + bias`.
    fn conv(
        &self,
        layouter: impl Layouter<F>,
        input: Matrix<Self::Num>,
        kernel: Matrix<Self::Num>,
        bias: Vec<Self::Num>,
        input_offset:usize
    ) -> Result<Matrix<Self::Num>, Error>;

    // Exposes a matrix as a public input to the circuit.
    fn expose_public(
        &self,
        layouter: impl Layouter<F>,
        ouput: Matrix<Self::Num>,
        offset:usize
    ) -> Result<(), Error>;
   /*  fn expose_public_from_hash(
        &self, 
        layouter: impl Layouter<F>, 
        input:Number<F>, 
        offset: usize) -> Result<(), Error>; */
}

/// The chip that will implement our instructions! Chips store their own
/// config, as well as type markers if necessary.
pub struct ConvChip<F: Field> {
    config: LeNetChipConfig,
    _marker: PhantomData<F>,
}

impl<F: Field> Chip<F> for ConvChip<F> {
    type Config = LeNetChipConfig;
    type Loaded = ();

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn loaded(&self) -> &Self::Loaded {
        &()
    }
}

/* #[derive(Clone, Debug)]
pub struct ConvChipConfig {
    //input, kernel and bias
    advice: [Column<Advice>; 4],
    // This is the public input (instance) column.
    instance: Column<Instance>,
    constant: Column<Fixed>,
    selector: Selector,
}
 */
impl<F: Field> ConvChip<F> {
    pub fn construct(config: <Self as Chip<F>>::Config) -> Self {
        Self {
            config,
            _marker: PhantomData,
        }
    }

    pub fn configure(
        meta: &mut ConstraintSystem<F>,
        advice: [Column<Advice>; 4],
        instance: Column<Instance>,
        constant: Column<Fixed>,
        n: usize, //shape of input matrix
        k: usize, //shape of kernel
    ) -> <Self as Chip<F>>::Config {
        meta.enable_equality(instance);
        meta.enable_constant(constant);
        for column in &advice {
            meta.enable_equality(*column);
        }
        let selector = meta.selector();
        
        meta.create_gate("conv", |meta| {
            // Query the selector
            let s_conv = meta.query_selector(selector);

            // Query the variables from the columns
            let x = meta.query_advice(advice[0], Rotation::cur()); // Input matrix
            let w = meta.query_advice(advice[1], Rotation::cur()); // Kernel matrix
            let b = meta.query_advice(advice[2], Rotation::cur()); // Bias vector
            let y = meta.query_instance(instance, Rotation::next()); // Output matrix
            
            // Construct the expressions for the convolution formula and constraint
            // y[i,j] = sum(x[i+k,j+l] * w[k,l] + b[i,j]) for k,l in [0,k-1]
            
            // Create a vector to store the res values
            let mut res_vec = Vec::new();

            // Calculate the output size
            let m = n - k + 1;

            for i in 0..m {
                for j in 0..m {
                    // sum x[i+k,j+l] * w[k,l] + b[i,j]
                    let mut res_ij = b.clone();
                    for k in 0..k {
                        for l in 0..k {
                            // value of x[i+k,j+l]
                            let x_ikjl = meta.query_advice(
                                advice[0],
                                Rotation(((i + k) * n + j + l) as i32),
                            );
                            // value of w[k,l]
                            let w_kl = meta.query_advice(
                                advice[1],
                                Rotation((k * k + l) as i32),
                            );
                            // sum x[i+k,j+l] * w[k,l]
                            res_ij = res_ij + x_ikjl * w_kl;
                        }
                    }
                    // insert res_ij into the vector
                    res_vec.push(res_ij);
                }
            }

            // Return the polynomial constraint
            // If s_conv is enabled, then y[i,j]-sum(x[i+k,j+l] * w[k,l] + b[i,j]) must be zero, otherwise it can be anything
            // Use y and each element of the res_vec as constraints, and check if they are zero
            Constraints::with_selector(s_conv,  
                (0..m * m).map(|i| {
                    let y_i = meta.query_instance(instance, Rotation(i as i32));
                    let res_i = &res_vec[i];
                    y_i - res_i.clone()
                }).collect::<Vec<_>>()
            )
        });

        LeNetChipConfig {
            advice,
            instance,
            constant,
            selector,
        }
    }

}

    
/// A variable representing a number.
#[derive(Debug,Clone)]
pub struct Number<F:Field>(pub AssignedCell<F, F>);

impl<F: Field> ConvInstructions<F> for ConvChip<F> {
    type Num = Number<F>;

    // load matrix from advice column
    fn load_matrix_from_advice(&self, mut layouter: impl Layouter<F>, index: usize, offset: usize, input: &Matrix<Value<F>>)-> Result<Matrix<Self::Num>, Error> {
        
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
                        // enable selector
                        let cell_offset = i * col + j + offset;
                        // config.selector.enable(&mut region, offset);
                        // assgin value to current cell
                        let ret = region
                        .assign_advice(
                            || format!("input[{}][{}]", i, j),
                            config.advice[index],
                            cell_offset, // offset of current cell
                            || value,
                        ).map(Number);
                        // add the result to vector
                        values.push(ret.unwrap());
                    }
                }
                Ok(())
            },
        );
        
        // turn vector to a matrix
        let matrix = values
        .split_off(row * col)
        .chunks(col)
        .map(|chunk| chunk.to_vec())
        .collect::<Matrix<Self::Num>>();

        // return matrix
        Ok(matrix)
    }

    fn load_bias(&self, mut layouter: impl Layouter<F>, bias: &Vec<Value<F>>) -> Result<Vec<Self::Num>, Error> {
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

        let bias_vec = values.split_off(len);
        Ok(bias_vec)
    }
    
    fn load_matrix_from_instance(&self, mut layouter: impl Layouter<F>, index: usize,offset: usize, input: &Matrix<Value<F>>)-> Result<Matrix<Self::Num>, Error>{
        
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
                        // enable selector
                        let cell_offset = i * col + j + offset;
                        // config.selector.enable(&mut region, offset);
                        // assgin value to current cell
                        let ret = region
                        .assign_advice_from_instance(
                            || format!("input[{}][{}]", i, j),
                            config.instance,
                            cell_offset, // offset of current cell
                            config.advice[index],
                            offset
                        ).map(Number);
                        // add the result to vector
                        values.push(ret.unwrap());
                    }
                }
                Ok(())
            },
        );
        
        // turn vector to a matrix
        let matrix = values
        .split_off(row * col)
        .chunks(col)
        .map(|chunk| chunk.to_vec())
        .collect::<Matrix<Self::Num>>();

        // return matrix
        Ok(matrix)
    }

    fn conv(
        &self,
        mut layouter: impl Layouter<F>,
        input: Matrix<Self::Num>,
        kernel: Matrix<Self::Num>,
        bias: Vec<Self::Num>,
        input_offset: usize
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

                        let offset = i * output_col + j + input_offset;

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
                        // to do:
                        // apply activation function
           
                        // assign output value to the current cell
                        let ret = region
                        .assign_advice(
                            || format!("output[{}][{}]", i, j),
                            config.advice[3], // use the 4th advice column
                            offset, // offset of current cell
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
        .split_off(output_col * output_row)
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
        offset: usize
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
                    i * input_row + j + offset, // offset of current cell
                );
            }
        }
        Ok(())
    }

    /* fn expose_public_from_hash(
        &self,
        mut layouter: impl Layouter<F>,
        input: Number<F>,
        offset: usize
    ) -> Result<(), Error> {
        
        //let value = Value::known(input);
        let config = self.config();
        //let v = Expression::Constant(value).try_into();
        //let f_value = Expression::Constant(value);

        //let number_cell:  AssignedCell<F,F>;
        //let number_cell = AssignedCell {
            //value: value,
            //cell: region.assign_advice(|| "cell", column, 0, || Ok(fp))?,
            //_marker: PhantomData::default(),
        //};
        //let f_2:Value<F> = value.into();
        //let num = Number<F>;
        layouter.constrain_instance(input.0.cell(), config.instance,offset)
        
    } */
}
