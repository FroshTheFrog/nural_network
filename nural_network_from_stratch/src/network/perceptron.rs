use crate::network::types::WeightInitializer;

pub struct Perceptron {
    weights: Vec<f64>,
    bias: f64,
}

impl Perceptron {
    pub fn new(weight_initializer: WeightInitializer, size: usize) -> Self {
        let weights = (0..size).map(|_| weight_initializer()).collect();
        let bias = weight_initializer();

        Self { weights, bias }
    }

    pub fn feed_forward(&self, inputs: &[f64]) -> f64 {
        self.weights
            .iter()
            .zip(inputs)
            .fold(self.bias, |acc, (w, i)| acc + w * i)
    }

    pub fn get_size(&self) -> usize {
        self.weights.len()
    }
}
