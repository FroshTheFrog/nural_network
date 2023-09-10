use crate::network::layers::Layer;

use super::{
    derivatives::{LayerDerivative, NetworkDerivative},
    functions::CostFunction,
};

type NetworkGradient = NetworkDerivative;

pub struct Network<'a> {
    layers: Vec<&'a mut dyn Layer>,
}

impl<'a> Network<'a> {
    pub fn new(layers: Vec<&'a mut dyn Layer>) -> Self {
        assert!(
            Self::verify_layer_dimensions(&layers),
            "Invalid layer dimensions"
        );
        Self { layers }
    }

    fn verify_layer_dimensions(layers: &Vec<&'a mut dyn Layer>) -> bool {
        layers
            .windows(2)
            .all(|w| w[0].get_output_size() == w[1].get_input_size())
    }

    pub fn feed_forward(
        &self,
        input: Vec<f64>,
        return_derivative: bool,
    ) -> (Vec<f64>, Option<NetworkDerivative>) {
        let mut derivative: NetworkDerivative = Vec::with_capacity(self.layers.len());
        let mut input = input;

        for layer in self.layers.iter() {
            let (output, layer_derivative) = layer.feed_forward(&input, return_derivative);
            input = output;
            if return_derivative {
                derivative.push(layer_derivative.unwrap());
            }
        }

        if return_derivative {
            (input, Some(derivative))
        } else {
            (input, None)
        }
    }

    pub fn train(
        &mut self,
        input: Vec<f64>,
        expected: Vec<f64>,
        cost_function: &dyn CostFunction,
        learning_rate: f64,
    ) -> f64 {
        let (output, derivative_option) = self.feed_forward(input, true);

        let mut unwrapped_derivative = derivative_option.unwrap();
        let cost_derivative = cost_function.derivative(&output, &expected);

        Self::calculate_gradient(&mut unwrapped_derivative, &cost_derivative);

        self.update(unwrapped_derivative, learning_rate);

        cost_function.cost(&output, &expected)
    }

    pub fn calculate_gradient(
        unwrapped_derivative: &mut NetworkDerivative,
        cost_derivative: &[f64],
    ) {
        unwrapped_derivative.reverse();

        let mut backprop_error = cost_derivative.to_vec();

        for layer_derivative in unwrapped_derivative.iter_mut() {
            Self::multiply_layer_derivative_by_vector(layer_derivative, &backprop_error);

            backprop_error = Self::calculate_next_backprop_error(layer_derivative, &backprop_error);
        }

        unwrapped_derivative.reverse();
    }

    pub fn multiply_layer_derivative_by_vector(
        derivative: &mut LayerDerivative,
        values: &Vec<f64>,
    ) {
        derivative
            .iter_mut()
            .zip(values.iter())
            .for_each(|(perceptron_derivative, value)| {
                perceptron_derivative.multiply(*value);
            });
    }

    pub fn calculate_next_backprop_error(
        layer_derivative: &LayerDerivative,
        current_backprop_error: &[f64],
    ) -> Vec<f64> {
        let mut next_backprop_error: Vec<f64> = vec![0.0; layer_derivative[0].derivatives_w.len()];

        for (i, perceptron_derivative) in layer_derivative.iter().enumerate() {
            for (j, weight_derivative) in perceptron_derivative.derivatives_w.iter().enumerate() {
                next_backprop_error[j] += current_backprop_error[i] * weight_derivative;
            }
        }

        next_backprop_error
    }

    fn update(&mut self, gradient: NetworkGradient, learning_rate: f64) {
        self.layers
            .iter_mut()
            .zip(gradient.iter())
            .for_each(|(layer, layer_gradient)| layer.update(layer_gradient, learning_rate));
    }
}
