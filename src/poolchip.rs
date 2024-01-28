use std::marker::PhantomData;
use ff::PrimeField;
//stride = 1, padding = 0
use halo2_proofs::{
    arithmetic::Field, circuit::{Value, Chip, Layouter, AssignedCell}, pasta::Fp, plonk::{
        Advice, Column, ConstraintSystem, Constraints, Error, Expression, Fixed, Instance, Selector
    }, poly::Rotation
};
pub use crate::matrix::Matrix;
use crate::{convchip::Number, matrix::shape,lenet5::LeNetChipConfig};
// A trait for implementing pooling layer in CNN using zcash/halo2 library
pub trait PoolInstructions<F: Field>: Chip<F> {
    type Num;
    // Loads input from instance column
    fn load_matrix_from_instance(&self, layouter: impl Layouter<F>, index:usize, offset:usize, input: &Matrix<Value<F>>)-> Result<Matrix<Self::Num>, Error>;
    // Loads constant from fixed column
    fn load_constant(&self, layouter: impl Layouter<F>, index:usize, constant: F)-> Result<Self::Num, Error>;
    // Returns `output = pool(input)`, where pool is a pooling function such as max or average
    fn pool(
        &self,
        layouter: impl Layouter<F>,
        input: Matrix<Self::Num>,
        pool_size_raw: usize,
        pool_size: Self::Num,
        index: usize,
        offset: usize
    ) -> Result<Matrix<Self::Num>, Error>;
    
    // Exposes a matrix as a public input to the circuit.
    fn expose_public(
        &self,
        layouter: impl Layouter<F>,
        output: Matrix<Self::Num>,
        offset:usize
    ) -> Result<(), Error>;
}

pub struct PoolChip<F: Field> {
    config: LeNetChipConfig,
    _marker: PhantomData<F>,
}

impl<F: Field> Chip<F> for PoolChip<F> {
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
pub struct LeNetChipConfig {
    advice: [Column<Advice>; 4],
    // This is the public input (instance) column.
    instance: Column<Instance>,
    constant: Column<Fixed>,
    selector: Selector,
} */

impl <F:Field> PoolChip<F> {
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
        k: usize, //shape of pool size
    ) -> <Self as Chip<F>>::Config 
    where F: Field {
        meta.enable_equality(instance);
        meta.enable_constant(constant);
        for column in &advice {
            meta.enable_equality(*column);
        }   
        let selector = meta.selector();

        meta.create_gate("avg_pool", |meta| {
            // Query the selector
            let s_pool = meta.query_selector(selector);
    
            // Query the variables from the columns
            let x = meta.query_advice(advice[0], Rotation::cur()); // Input matrix
            let y = meta.query_instance(instance, Rotation::next()); // Output matrix
            let exp_k = meta.query_fixed(constant);
            // Construct the expressions for the average pooling formula and constraint
            // y[i,j] = mean(x[i*k:i*k+k,j*k:j*k+k]) for i,j in [0,n/k-1]
    
            // Create a vector to store the res values
            let mut res_vec = Vec::new();

            // Calculate the output size
            let m = n / k;

            for i in 0..m {
                for j in 0..m {
                    // sum x[i*k:i*k+k,j*k:j*k+k]
                    let mut res_ij = Expression::Constant(F::ZERO);
                    for p in 0..k {
                        for q in 0..k {
                            // value of x[i*k+p,j*k+q]
                            let x_ikpjkq = meta.query_advice(
                                advice[0],
                                Rotation(((i * k + p) * n + j * k + q) as i32),
                            );
                            // sum x[i*k:i*k+k,j*k:j*k+k]
                            res_ij = res_ij + x_ikpjkq;
                        }
                    }

                    // divide by k*k
                    // insert res_ij into the vector
                    res_vec.push(res_ij);
                }
            }

            Constraints::with_selector(s_pool,  
                (0..m * m).map(|i| {
                    let ksqr = exp_k.clone() * exp_k.clone();
                    let y_i = meta.query_instance(instance, Rotation(i as i32));
                    let res_i = &res_vec[i];
                    y_i * ksqr - res_i.clone()
                }).collect::<Vec<_>>()
            )
        });

        LeNetChipConfig {
            advice,
            instance,
            selector,
            constant
        }
    }
}

impl<F: Field> PoolInstructions<F> for PoolChip<F> {

    type Num = Number<F>;

    fn load_matrix_from_instance(&self, mut layouter: impl Layouter<F>, index: usize, offset: usize, input: &Matrix<Value<F>>)-> Result<Matrix<Self::Num>, Error>{
        
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

    fn load_constant(&self, mut layouter: impl Layouter<F>, index:usize, constant: F)-> Result<Self::Num, Error> {
        let config = self.config();

        layouter.assign_region(
            || "load constant",
            |mut region| {
                region
                    .assign_advice_from_constant(|| "constant value", config.advice[index],  0, constant)
                    .map(Number)
            },
        )
    }

    fn pool(
            &self,
            mut layouter: impl Layouter<F>,
            input: Matrix<Self::Num>,
            pool_size_raw: usize,
            pool_size: Self::Num,
            index: usize,
            offset: usize
    ) -> Result<Matrix<Self::Num>, Error> {
    let config = self.config();
    let mut values = Vec::new();
    
    let (input_row, input_col) = shape(&input);
    let output_size = input_row / pool_size_raw;
    let ksqr = pool_size.0.value_field().square();
    let ksqr_inv = ksqr.invert().evaluate();
    // assign output values to the corresponding advice columns
    let _ = layouter.assign_region(
        || "assign output",
        |mut region| {
            // iterate over the input matrix with a sliding window of the pool size
            for i in 0..output_size {
                for j in 0..output_size {
                    // create a vector for the sum values
                    let mut sums = Vec::new();

                    let offset = i * input_col + j + offset;

                    // iterate over the pool window
                    for p in 0..pool_size_raw {
                        for q in 0..pool_size_raw {
                            // get the input[i*pool_size+p][j*pool_size+q] value
                            let input_value = &input[i*pool_size_raw+p][j*pool_size_raw+q];

                            // add the input value to the sum vector
                            let sum = input_value.0.value().cloned();
                            sums.push(sum);
                        }
                    }

                    // sum up the sum values
                    let sum = sums.iter().fold(Value::known(Field::ZERO), |acc, x| acc + x);


                    // divide by pool_size*pool_size
                    let output_value = sum * ksqr_inv;
                    // to do:
                    // apply activation function
       
                    // assign output value to the current cell
                    let ret = region
                    .assign_advice(
                        || format!("output[{}][{}]", i, j),
                        config.advice[index], // use the specified advice column
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
    .split_off(output_size * output_size)
    .chunks(output_size)
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
}