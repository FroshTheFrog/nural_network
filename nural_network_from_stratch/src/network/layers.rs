use crate::network::perceptron::Perceptron;
use crate::network::types::{ActivationFunction, Layer, WeightInitializer};

pub struct NeuralLayer<'a> {
    perceptrons: Vec<Perceptron>,
    activation_function: &'a dyn ActivationFunction,
}

impl<'a> NeuralLayer<'a> {
    pub fn new(
        activation_function: &'a dyn ActivationFunction,
        weight_initializer: WeightInitializer,
        input_size: usize,
        output_size: usize,
    ) -> Self {
        let perceptrons = (0..output_size)
            .map(|_| Perceptron::new(weight_initializer, input_size))
            .collect();

        Self {
            perceptrons,
            activation_function,
        }
    }
}

impl Layer for NeuralLayer<'_> {
    fn get_input_size(&self) -> usize {
        self.perceptrons[0].get_size()
    }

    fn get_output_size(&self) -> usize {
        self.perceptrons.len()
    }

    fn feed_forward(&self, inputs: &[f64]) -> Vec<f64> {
        self.perceptrons
            .iter()
            .map(|p| self.activation_function.activate(p.feed_forward(inputs)))
            .collect()
    }
}
