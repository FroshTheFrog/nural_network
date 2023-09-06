struct Perceptron<const SIZE: usize> {
    weights: [f64; SIZE],
    bias: f64,
}

impl<const SIZE: usize> Perceptron<{ SIZE }> {
    fn new(weight_initializer: fn() -> f64) -> Self {
        Perceptron {
            weights: [(); SIZE].map(|_| weight_initializer()),
            bias: weight_initializer(),
        }
    }
}

trait Layer {}

struct NeuralLayer<const INPUT_SIZE: usize, const OUTPUT_SIZE: usize> {
    perceptions: [Perceptron<INPUT_SIZE>; OUTPUT_SIZE],
    activation_function: fn(f64) -> f64,
}

impl<const INPUT_SIZE: usize, const OUTPUT_SIZE: usize> NeuralLayer<INPUT_SIZE, OUTPUT_SIZE> {
    fn new(activation_function: fn(f64) -> f64, weight_initializer: fn() -> f64) -> Self {
        NeuralLayer {
            perceptions: [(); OUTPUT_SIZE].map(|_| Perceptron::new(weight_initializer)),
            activation_function: activation_function,
        }
    }
}

impl<const INPUT_SIZE: usize, const OUTPUT_SIZE: usize> Layer
    for NeuralLayer<INPUT_SIZE, OUTPUT_SIZE>
{
}

pub struct Network<const NUMBER_OF_LAYERS: usize> {
    layers: [Box<dyn Layer>; NUMBER_OF_LAYERS],
}

impl<const NUMBER_OF_LAYERS: usize> Network<NUMBER_OF_LAYERS> {
    pub fn new(layers: [Box<dyn Layer>; NUMBER_OF_LAYERS]) -> Self {
        Network { layers: layers }
    }
}
