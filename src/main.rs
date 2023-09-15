use halo2_proofs::{
    arithmetic::Field,
    circuit::{layouter::TableLayouter, Cell, Chip, Layouter},
    dev::{MockProver, VerifyFailure},
    pasta::Fp,
    plonk::{
        Advice, Assignment, Circuit, Column, ConstraintSystem, Error, Expression, Fixed,
        Selector
    },
    poly::Rotation,
};

struct CircuitConfig {
    // 定义advice列
    x_col: Column<Advice>,
    k_col: Column<Advice>,
    y_col: Column<Advice>,
    q_col: Column<Advice>,
    //定义fix列
    t_col: Column<Fixed>,
    //定义permutation
    perm: Permutation

}

fn main() {
    
}
