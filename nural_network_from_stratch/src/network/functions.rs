use crate::network::derivatives::WeightInitializer;
use rand::Rng;

const H: f64 = 1e-7;

pub trait ActivationFunction {
    fn activate(&self, x: f64) -> f64;
    fn derivative(&self, x: f64) -> f64;
}

pub trait CostFunction {
    fn cost(&self, output: &[f64], expected: &[f64]) -> f64;
    fn derivative(&self, output: &[f64], expected: &[f64]) -> Vec<f64>;
}

pub fn create_random_weight_initializer<const LOWER_BOUND: i64, const UPPER_BOUND: i64>(
) -> WeightInitializer {
    || {
        let mut rng = rand::thread_rng();
        rng.gen_range((LOWER_BOUND as f64)..(UPPER_BOUND as f64))
    }
}

pub struct Relu;

impl ActivationFunction for Relu {
    fn activate(&self, x: f64) -> f64 {
        if x > 0.0 {
            x
        } else {
            0.0
        }
    }

    fn derivative(&self, x: f64) -> f64 {
        if x > 0.0 {
            1.0
        } else {
            0.0
        }
    }
}

pub const SIGMOID: fn(f64) -> f64 = |x| 1.0 / (1.0 + (-x).exp());

pub struct MeanSquaredError;

impl CostFunction for MeanSquaredError {
    fn cost(&self, output: &[f64], expected: &[f64]) -> f64 {
        output
            .iter()
            .zip(expected)
            .map(|(o, e)| (o - e).powi(2))
            .sum::<f64>()
            / output.len() as f64
    }

    fn derivative(&self, output: &[f64], expected: &[f64]) -> Vec<f64> {
        output
            .iter()
            .zip(expected)
            .map(|(o, e)| 2.0 * (o - e))
            .collect()
    }
}

impl<F> ActivationFunction for F
where
    F: Fn(f64) -> f64,
{
    fn activate(&self, x: f64) -> f64 {
        self(x)
    }

    fn derivative(&self, x: f64) -> f64 {
        (self(x + H) - self(x)) / H
    }
}

impl<F> CostFunction for F
where
    F: Fn(&[f64], &[f64]) -> f64,
{
    fn cost(&self, output: &[f64], expected: &[f64]) -> f64 {
        self(output, expected)
    }

    fn derivative(&self, output: &[f64], expected: &[f64]) -> Vec<f64> {
        let mut result = Vec::with_capacity(output.len());
        for i in 0..output.len() {
            let mut output_plus_h = output.to_vec();
            output_plus_h[i] += H;
            result.push((self(&output_plus_h, expected) - self(output, expected)) / H);
        }
        result
    }
}
