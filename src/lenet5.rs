use halo2_proofs::{
    circuit::{self, SimpleFloorPlanner, Value}, 
    plonk::{Advice, Circuit, Column, ConstraintSystem, Error, Fixed, Instance, Selector},
    arithmetic::Field ,pasta::{group::ff::PrimeField, Fp, Eq}, 
    dev::{CircuitCost, MockProver}};
use rand::{rngs::OsRng, Rng};
use crate::{matrix::{Matrix}, utils::generate_matrix};
pub use halo2_proofs::dev::cost;
use crate::{convchip::{ConvChip, ConvInstructions}, poolchip::{PoolChip, PoolInstructions}};
use std::time::Instant;
pub use crate::{convchip::Number, utils};

#[derive(Default)]
pub struct LeNet5 <F: PrimeField> {
    input:Matrix<Value<F>>,
    filter: Vec<Matrix<Value<F>>>,
    bias: Vec<Value<F>>,
    constant: Vec<F>,
}

#[derive(Clone, Debug)]
pub struct LeNetChipConfig {
    //input, kernel and bias
    pub advice: [Column<Advice>; 4],
    // This is the public input (instance) column.
    pub instance: Column<Instance>,
    pub constant: Column<Fixed>,
    pub selector: Selector,
}

impl<F: PrimeField> Circuit<F> for LeNet5<F> {
    type Config = LeNetChipConfig;

    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        todo!()
    }

    fn synthesize(&self, config: Self::Config, mut layouter: impl circuit::Layouter<F>) -> Result<(), Error> {

        let conv_chip = ConvChip::<F>::construct(config.clone());
        let pool_chip = PoolChip::<F>::construct(config.clone());

        // Load our private values into the circuit.
        let input_1 = conv_chip.load_matrix_from_advice(layouter.namespace(|| "load input"), 0, 0, &self.input)?;

        let mut kernel_1 = Vec::new();
        for i in 0..6 {
            kernel_1.push(conv_chip.load_matrix_from_advice(layouter.namespace(|| "load filter"), 1, (i-1)*25,&self.filter[i])?);
        }
        let bias = conv_chip.load_bias(layouter.namespace(|| "load bias"), &self.bias)?;

        let mut output_1 = Vec::new();
        for i in 0..6 {
            let output = conv_chip.conv(layouter.namespace(|| "conv operation"), input_1.clone(), kernel_1[i].clone(), bias.clone(), 0)?;
            output_1.push(output);
        }

        let pool_size = pool_chip.load_constant(layouter.namespace(|| "get pool size"), 1, self.constant[0]).unwrap();
        let mut output_2 = Vec::new();
        for j in 0..6 {
            let output = pool_chip.pool(layouter.namespace(|| "pool operation"), output_1[j].clone(), 2, pool_size.clone(), 0,0).unwrap();
            output_2.push(output)
        }


        //let output_2 = pool_chip.pool(layouter, output_1[1], 2, pool_size, 0,0);
        pool_chip.expose_public(layouter.namespace(|| "expose output"), output_2[0].clone(), 0);
        pool_chip.expose_public(layouter.namespace(|| "expose output"), output_2[1].clone(), 14*14);
        pool_chip.expose_public(layouter.namespace(|| "expose output"), output_2[2].clone(), 2*14*14);
        pool_chip.expose_public(layouter.namespace(|| "expose output"), output_2[3].clone(), 3*14*14);
        pool_chip.expose_public(layouter.namespace(|| "expose output"), output_2[4].clone(), 4*14*14);
        pool_chip.expose_public(layouter.namespace(|| "expose output"), output_2[5].clone(), 5*14*14)
    }
}

pub fn run_lenet5_test() {

    let k = 16;
    let input_size = 32;
    let mut public_output = Vec::new();
    let mut constant_vec =  Vec::new();
    let constant = Fp::from(2);
    constant_vec.push(constant);

    for i in 0..1176 {
        public_output.push(constant);
    }

    let (input, raw_input) = generate_matrix(1, input_size, input_size);
    let (filter,raw_filter) = generate_matrix(6, 5, 5);
    
    let circuit_input = &input[0];
    let mut bias = Vec::new();
    bias.push(Value::known(Fp::from(0)));

    let circuit = LeNet5 {
        input: circuit_input.to_vec(),
        filter: filter,
        bias: bias,
        constant: constant_vec,
    };

    let start_1 = Instant::now();
    let prover = MockProver::run(k, &circuit, vec![public_output]).unwrap();
    let end_1 = Instant::now();

    let duration_1 = end_1 - start_1;
    println!("The proof took {:?}", duration_1);

    let start_2 = Instant::now();
    prover.verify();
    let end_2 = Instant::now();

    let duration_2 = end_2 - start_2;
    println!("The verification took {:?}", duration_2);

    let proof_cost = CircuitCost::<Eq, LeNet5<Fp>>::measure(k, &circuit).proof_size(1);

    //let sets = proof_size::point_sets;
    println!("The proof size is {:?}", proof_cost);
}