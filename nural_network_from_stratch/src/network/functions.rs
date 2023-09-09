use crate::network::types::ActivationFunction;
use crate::network::types::WeightInitializer;
use rand::Rng;

pub const ReLU: ActivationFunction = |x| if x > 0.0 { x } else { 0.0 };
pub const SigMoid: ActivationFunction = |x| 1.0 / (1.0 + (-x).exp());

pub fn create_random_weight_initializer<const LOWER_BOUND: i64, const UPPER_BOUND: i64>(
) -> WeightInitializer {
    || {
        let mut rng = rand::thread_rng();
        rng.gen_range((LOWER_BOUND as f64)..(UPPER_BOUND as f64))
    }
}
