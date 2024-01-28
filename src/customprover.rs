use ff::Field;
use halo2_proofs::{circuit, pasta::EqAffine, plonk::
{create_proof,verify_proof, keygen_pk, keygen_vk, Circuit}, poly::commitment::Params, 
transcript::{Blake2bWrite, Challenge255}};
use rand::rngs::OsRng;

struct Custom_Prover {
} 

impl Custom_Prover{
    //fn create_custom_proof<ConcreteCircuit: Circuit<F:Field>> (&self, circuit:&ConcreteCircuit, ){
        //let params: Params<EqAffine> = Params::new(3);
        //let vk = keygen_vk(&params, &self.circuit);
        //let pk = keygen_pk(&params, vk, &self.circuit);
        //let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);

        //let proof = create_proof(&params,
        //&pk,
        //&[self.circuit, self.circuit],
        //&[],
        //OsRng,
        //&mut transcript,
        //);


    //}

    //fn verify_custom_proof(){
    //}
}