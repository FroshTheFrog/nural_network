use crate::network::derivatives::WeightInitializer;
use crate::network::perceptron::Perceptron;

use super::{
    derivatives::{LayerDerivative, PerceptronDerivative},
    functions::ActivationFunction,
};

pub trait Layer {
    fn get_input_size(&self) -> usize;
    fn get_output_size(&self) -> usize;
    fn feed_forward(
        &self,
        inputs: &[f64],
        return_derivative: bool,
    ) -> (Vec<f64>, Option<LayerDerivative>);

    fn update(&mut self, derivative: &LayerDerivative, learning_rate: f64);
}

pub struct NeuralLayer<'a> {
    perceptrons: Vec<Perceptron<'a>>,
}

impl<'a> NeuralLayer<'a> {
    pub fn new(
        activation_function: &'a dyn ActivationFunction,
        weight_initializer: WeightInitializer,
        input_size: usize,
        output_size: usize,
    ) -> Self {
        let perceptrons = (0..output_size)
            .map(|_| Perceptron::new(weight_initializer, input_size, activation_function))
            .collect();

        Self { perceptrons }
    }
}

impl Layer for NeuralLayer<'_> {
    fn get_input_size(&self) -> usize {
        self.perceptrons[0].get_size()
    }

    fn get_output_size(&self) -> usize {
        self.perceptrons.len()
    }

    fn feed_forward(
        &self,
        inputs: &[f64],
        return_derivative: bool,
    ) -> (Vec<f64>, Option<LayerDerivative>) {
        let results: Vec<(f64, Option<PerceptronDerivative>)> = self
            .perceptrons
            .iter()
            .map(|p| p.feed_forward(inputs, return_derivative))
            .collect();

        let (outputs, derivatives): (Vec<f64>, Vec<Option<PerceptronDerivative>>) =
            results.into_iter().unzip();

        if return_derivative {
            (outputs, Some(derivatives.into_iter().flatten().collect()))
        } else {
            (outputs, None)
        }
    }

    fn update(&mut self, derivative: &LayerDerivative, learning_rate: f64) {
        self.perceptrons
            .iter_mut()
            .zip(derivative.iter())
            .for_each(|(p, g)| p.update(g, learning_rate));
    }
}
