use halo2_proofs::{circuit::Value, pasta::Fp};
use rand::Rng;
use crate::matrix::Matrix;

pub fn generate_matrix(filter_num: usize, row_size: usize, col_size: usize) -> (Vec<Matrix<Value<Fp>>>, Vec<Matrix<u64>>) { // change the function name and add two parameters for row size and column size
    // Create a random number generator
    let mut rng = rand::thread_rng();
    // create a vector of matrices for filters
    let mut filter: Vec<Matrix<Value<Fp>>> = Vec::new();
    // create a vector of matrices for raw values
    let mut raw_filter: Vec<Matrix<u64>> = Vec::new();
    // loop over filter_num
    for _ in 0..filter_num {
        // create a matrix for one filter
        let mut filter_matrix: Matrix<Value<Fp>> = Vec::new();
        // create a matrix for one raw value
        let mut raw_matrix: Matrix<u64> = Vec::new();
        for _ in 0..row_size { // use row_size instead of 5
            // create a row for one filter
            let mut filter_row: Vec<Value<Fp>> = Vec::new();
            // create a row for one raw value
            let mut raw_row: Vec<u64> = Vec::new();
            for _ in 0..col_size { // use col_size instead of 5
                // generate a random value
                let x = rng.gen_range(0..255);
                let y = Fp::from(x);
                // add the value to the row
                filter_row.push(Value::known(y));
                // add the value to the raw row
                raw_row.push(x);
            }
            // add the row to the matrix
            filter_matrix.push(filter_row);
            // add the raw row to the raw matrix
            raw_matrix.push(raw_row);
        }
        // add the matrix to the vector
        filter.push(filter_matrix);
        // add the raw matrix to the raw vector
        raw_filter.push(raw_matrix);
    }
    // return the tuple of vectors
    (filter, raw_filter)
}