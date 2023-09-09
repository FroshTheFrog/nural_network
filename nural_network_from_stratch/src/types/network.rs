type ActivationFunction = fn(f64) -> f64;
type WeightInitializer = fn() -> f64;

struct Perceptron {
    weights: Vec<f64>,
    bias: f64,
}

impl Perceptron {
    fn new(weight_initializer: WeightInitializer, size: usize) -> Self {
        let weights = (0..size).map(|_| weight_initializer()).collect();
        let bias = weight_initializer();

        Self { weights, bias }
    }

    fn feed_forward(&self, inputs: &[f64]) -> f64 {
        self.weights
            .iter()
            .zip(inputs)
            .fold(self.bias, |acc, (w, i)| acc + w * i)
    }
}

trait Layer {
    fn get_input_size(&self) -> usize;
    fn get_output_size(&self) -> usize;
    fn feed_forward(&self, inputs: &[f64]) -> Vec<f64>;
}

struct NeuralLayer {
    perceptrons: Vec<Perceptron>,
    activation_function: ActivationFunction,
}

impl NeuralLayer {
    fn new(
        activation_function: ActivationFunction,
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

impl Layer for NeuralLayer {
    fn get_input_size(&self) -> usize {
        self.perceptrons[0].weights.len()
    }

    fn get_output_size(&self) -> usize {
        self.perceptrons.len()
    }

    fn feed_forward(&self, inputs: &[f64]) -> Vec<f64> {
        self.perceptrons
            .iter()
            .map(|p| (self.activation_function)(p.feed_forward(inputs)))
            .collect()
    }
}

pub struct Network {
    layers: Vec<Box<dyn Layer>>,
}

impl Network {
    pub fn new(layers: Vec<Box<dyn Layer>>) -> Self {
        assert!(
            Self::verify_layer_dimensions(&layers),
            "Invalid layer dimensions"
        );
        Self { layers }
    }

    fn verify_layer_dimensions(layers: &[Box<dyn Layer>]) -> bool {
        layers
            .windows(2)
            .all(|w| w[0].get_output_size() == w[1].get_input_size())
    }

    pub fn feed_forward(&self, input: Vec<f64>) -> Vec<f64> {
        self.layers
            .iter()
            .fold(input, |acc, layer| layer.feed_forward(&acc))
    }
}
