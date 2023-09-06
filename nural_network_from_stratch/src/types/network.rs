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

trait Layer {
    fn get_input_size(&self) -> usize;
    fn get_output_size(&self) -> usize;
}

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
    fn get_input_size(&self) -> usize {
        INPUT_SIZE
    }

    fn get_output_size(&self) -> usize {
        OUTPUT_SIZE
    }
}

pub type LayerList<const NUMBER_OF_LAYERS: usize> = [Box<dyn Layer>; NUMBER_OF_LAYERS];

pub struct Network<const NUMBER_OF_LAYERS: usize> {
    layers: LayerList<NUMBER_OF_LAYERS>,
}

impl<const NUMBER_OF_LAYERS: usize> Network<NUMBER_OF_LAYERS> {
    pub fn new(layers: LayerList<NUMBER_OF_LAYERS>) -> Option<Self> {
        if Self::verify_layer_dimensions(&layers) {
            return Some(Network { layers });
        }
        None
    }

    fn verify_layer_dimensions(layers: &LayerList<NUMBER_OF_LAYERS>) -> bool {
        let mut previous_layer_output_Size = layers[0].get_output_size();

        for layer in layers.iter().skip(1) {
            let current_layer_input_size = layer.get_input_size();

            if current_layer_input_size != previous_layer_output_Size {
                return false;
            }

            previous_layer_output_Size = layer.get_output_size();
        }
        true
    }
}
