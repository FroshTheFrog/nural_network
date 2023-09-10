use crate::network::layers::Layer;

use super::{derivatives::NetworkDerivative, functions::CostFunction};

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
        let (output, derivative) = self.feed_forward(input, true);

        let unwrapped_derivative = derivative.unwrap();
        let cost_derivative = cost_function.derivative(&output, &expected);

        let mut gradient: NetworkGradient = Vec::with_capacity(self.layers.len());

        // TODO

        gradient.reverse();
        self.update(gradient, learning_rate);

        cost_function.cost(&output, &expected)
    }

    fn update(&mut self, gradient: NetworkGradient, learning_rate: f64) {
        self.layers
            .iter_mut()
            .zip(gradient.iter())
            .for_each(|(layer, layer_gradient)| layer.update(layer_gradient, learning_rate));
    }
}
