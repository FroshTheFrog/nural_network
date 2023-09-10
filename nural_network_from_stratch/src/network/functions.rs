use crate::network::types::ActivationFunction;
use crate::network::types::WeightInitializer;
use rand::Rng;

pub fn create_random_weight_initializer<const LOWER_BOUND: i64, const UPPER_BOUND: i64>(
) -> WeightInitializer {
    || {
        let mut rng = rand::thread_rng();
        rng.gen_range((LOWER_BOUND as f64)..(UPPER_BOUND as f64))
    }
}

pub const ReLU: fn(f64) -> f64 = |x| if x > 0.0 { x } else { 0.0 };
pub const Sigmoid: fn(f64) -> f64 = |x| 1.0 / (1.0 + (-x).exp());

impl<F> ActivationFunction for F
where
    F: Fn(f64) -> f64,
{
    fn activate(&self, x: f64) -> f64 {
        self(x)
    }

    fn derivative(&self, x: f64) -> f64 {
        let h = 1e-7;
        (self(x + h) - self(x)) / h
    }
}
