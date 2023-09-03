struct Perceptron {
    weights: Vec<f64>,
    bias: f64,
}

impl Perceptron {
    fn new(size: usize, weight_initializer: fn() -> f64) -> Self {
        let mut weights = Vec::with_capacity(size);
        for _ in 0..size {
            weights.push(weight_initializer());
        }

        Perceptron {
            weights,
            bias: weight_initializer(),
        }
    }
}

trait Layer {}

struct NeuralLayer {
    perceptions: Vec<Perceptron>,
    activation_function: fn(f64) -> f64,
}

impl NeuralLayer {
    fn new(
        size: usize,
        activation_function: fn(f64) -> f64,
        weight_initializer: fn() -> f64,
    ) -> Self {
        let mut perceptions: Vec<Perceptron> = Vec::with_capacity(size);
        for _ in 0..size {
            perceptions.push(Perceptron::new(size, weight_initializer));
        }

        NeuralLayer {
            perceptions: perceptions,
            activation_function: activation_function,
        }
    }
}

impl Layer for NeuralLayer {}

pub struct Network {
    layers: Vec<Box<dyn Layer>>,
}

impl Network {
    pub fn new(layers: Vec<Box<dyn Layer>>) -> Self {
        Network { layers }
    }
}
