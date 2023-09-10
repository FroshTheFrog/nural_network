use crate::network::types::ActivationFunction;
use crate::network::types::WeightInitializer;

use super::types::PerceptronDerivative;

pub struct Perceptron<'a> {
    activation_function: &'a dyn ActivationFunction,
    weights: Vec<f64>,
    bias: f64,
}

impl<'a> Perceptron<'a> {
    pub fn new(
        weight_initializer: WeightInitializer,
        size: usize,
        activation_function: &'a dyn ActivationFunction,
    ) -> Self {
        let weights = (0..size).map(|_| weight_initializer()).collect();
        let bias = weight_initializer();

        Self {
            weights,
            bias,
            activation_function,
        }
    }

    pub fn feed_forward(
        &self,
        inputs: &[f64],
        return_derivative: bool,
    ) -> (f64, Option<PerceptronDerivative>) {
        let before_activation = self
            .weights
            .iter()
            .zip(inputs)
            .fold(self.bias, |acc, (w, i)| acc + w * i);

        let result = self.activation_function.activate(before_activation);

        if return_derivative {
            let derivative = self.derivative(inputs, before_activation);
            (result, Some(derivative))
        } else {
            (result, None)
        }
    }

    pub fn get_size(&self) -> usize {
        self.weights.len()
    }

    pub fn update(&mut self, derivative: &PerceptronDerivative, learning_rate: f64) {
        self.weights
            .iter_mut()
            .zip(derivative.derivatives_w.iter())
            .for_each(|(w, dw)| *w -= learning_rate * dw);

        self.bias -= learning_rate * derivative.derivative_b;
    }

    fn derivative(&self, inputs: &[f64], z: f64) -> PerceptronDerivative {
        let derivative_at_z = self.activation_function.derivative(z);

        let derivatives_w = inputs
            .iter()
            .map(|&input| derivative_at_z * input)
            .collect();

        let derivative_b = derivative_at_z;

        PerceptronDerivative {
            derivatives_w,
            derivative_b,
        }
    }
}
