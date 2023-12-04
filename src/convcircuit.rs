use ff::Field;
use halo2_proofs::circuit::Value;


/// The full circuit implementation.
///
/// In this struct we store the private input variables. We use `Option<F>` because
/// they won't have any value during key generation. During proving, if any of these
/// were `None` we would get an error.
#[derive(Default)]
struct ConvCircuit<F: Field> {
    constant: F,
    a: Value<F>,
    b: Value<F>,
}

