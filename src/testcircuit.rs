use halo2_proofs::{
    circuit::{Value, SimpleFloorPlanner}, 
    plonk::{Circuit, ConstraintSystem, Error},
    arithmetic::Field ,pasta::Fp, 
    dev::MockProver};
use rand::Rng;
use crate::matrix::{Matrix, shape};
use halo2_test::convchip::{ConvChip, ConvChipConfig, ConvInstructions,self};
use std::time::Instant;

/// The full circuit implementation.
///
/// In this struct we store the private input variables. We use `Option<F>` because
/// they won't have any value during key generation. During proving, if any of these
/// were `None` we would get an error.
#[derive(Default)]
pub struct ConvCircuit<F: Field> {
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
        
        ConvChip::configure(meta, advice, instance, constant, 25)
    }
    
    fn synthesize(&self, 
        config: Self::Config, 
        mut layouter: impl halo2_proofs::circuit::Layouter<F>) -> Result<(), Error> {

        let conv_chip = ConvChip::<F>::construct(config);
        // Load our private values into the circuit.
        let input = conv_chip.load_matrix_from_advice(layouter.namespace(|| "load input"), 0, &self.input)?;
        let kernel = conv_chip.load_matrix_from_advice(layouter.namespace(|| "load filter"), 1, &self.filter)?;
        let bias = conv_chip.load_bias(layouter.namespace(|| "load bias"), &self.bias)?;
        
        let output = conv_chip.conv(layouter.namespace(|| "conv operation"), input, kernel, bias)?;
        
        // Expose the result as a public input to the circuit.
        conv_chip.expose_public(layouter.namespace(|| "expose output"), output)
    }
}

use std::vec::Vec;

fn raw_conv(input: Vec<Vec<u64>>, kernel: Vec<Vec<u64>>, bias: Vec<u64>) -> Vec<Vec<u64>> {
    let input_size = input.len();
    let kernel_size = kernel.len();

    let output_size = input_size - kernel_size + 1;

    let mut output: Vec<Vec<u64>> = Vec::new();

    for i in 0..output_size {
        let mut output_row: Vec<u64> = Vec::new();
        for j in 0..output_size {
            let mut sum = 0;
            for k in 0..kernel_size {
                for l in 0..kernel_size {
                    sum += input[i + k][j + l] * kernel[k][l];
                }
            }
            output_row.push(sum + bias[0]);
        }
        output.push(output_row);
    }
    output
}

pub fn run_conv_test(filter_num: usize) {
    // The number of rows in our circuit cannot exceed 2^k
    // and the input set of MNIST and CIFAR10 is 32*32
    let k = 11;
    let constant = Fp::from(7);

    // Create a random number generator
    let mut rng = rand::thread_rng();
    let mut input: Matrix<Value<Fp>> = Vec::new();
    let mut raw_input= Vec::new();

    let input_size = 28;
    // Use a loop to assign values to each row of the matrix
    for _ in 0..input_size {
        // Create an empty matrix let mut matrix
        let mut row: Vec<Value<Fp>> = Vec::new();
        let mut raw_row = Vec::new();
        // Use a loop to assign values to each element of the vector
        for _ in 0..input_size {
            // Generate a random integer from 0 to 255
            let x = rng.gen_range(0..255);
            // Use Fp::from(x) to convert the integer to a finite field element
            let y = Fp::from(x);
            raw_row.push(x);
            // Add the element to the vector
            row.push(Value::known(y));
        }
        // Add vector to the matrix
        raw_input.push(raw_row);
        input.push(row);
    }

    // generate a filter in the same way
    let mut filter: Matrix<Value<Fp>> = Vec::new();
    let mut raw_filter= Vec::new();
    for _ in 0..5 {
        let mut row: Vec<Value<Fp>> = Vec::new();
        let mut raw_filter_row  = Vec::new();
        for _ in 0..5 {
            let x = rng.gen_range(0..255);
            raw_filter_row.push(x);
            let y = Fp::from(x);
            row.push(Value::known(y));
        }
        // 将向量添加到filter中
        raw_filter.push(raw_filter_row);
        filter.push(row);
    }

    let mut bias: Vec<Value<Fp>> = Vec::new();
    let mut raw_bias = Vec::new();
    raw_bias.push(0);
    bias.push(Value::known(Fp::from(0)));



    let circuit = ConvCircuit {
        input,
        filter,
        bias,
    };

    let output_size = input_size - 5 + 1;
    let mut output: Vec<Fp> = Vec::new();
    let raw_output = raw_conv(raw_input, raw_filter, raw_bias);
    
    let iter = raw_output.into_iter();

    for row in iter {
        let iter = row.into_iter();
        for elem in iter {
            let value = Fp::from(elem);
            output.push(value);
        }
    }

    //output.push(constant);
    //output[0] += Fp::one();
    let start_1 = Instant::now();
    let prover = MockProver::run(k, &circuit, vec![output]).unwrap();
    let end_1 = Instant::now();

    let duration_1 = end_1 - start_1;
    println!("The proof took {:?}", duration_1);

    let start_2 = Instant::now();
    prover.verify();
    let end_2 = Instant::now();

    let duration_2 = end_2 - start_2;
    println!("The verification took {:?}", duration_2);

}


