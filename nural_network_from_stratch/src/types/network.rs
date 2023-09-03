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

trait Layer {
    fn size(&self) -> usize;
}

struct NeuralLayer {
    perceptions: Vec<Perceptron>,
    activation_function: fn(f64) -> f64,
}

impl NeuralLayer {
    fn setup(
        size: usize,
        activation_function: fn(f64) -> f64,
        weight_initializer: fn() -> f64,
    ) -> impl Fn(usize) -> Self {
        move |input: usize| {
            let mut perceptions: Vec<Perceptron> = Vec::with_capacity(size);
            for _ in 0..size {
                perceptions.push(Perceptron::new(input, weight_initializer));
            }

            NeuralLayer {
                perceptions: perceptions,
                activation_function: activation_function,
            }
        }
    }
}

impl Layer for NeuralLayer {
    fn size(&self) -> usize {
        self.perceptions.len()
    }
}

pub struct Network {
    layers: Vec<Box<dyn Layer>>,
}

impl Network {
    pub fn new(input_size: usize, layers: Vec<Box<dyn Fn(usize) -> Box<dyn Layer>>>) -> Self {
        let mut compiled_layers = Vec::with_capacity(layers.len());
        let mut current_input_size = input_size;

        for layer_closure in layers.iter() {
            let compiled_layer = layer_closure(current_input_size);
            current_input_size = compiled_layer.size();
            compiled_layers.push(compiled_layer);
        }

        Network {
            layers: compiled_layers,
        }
    }
}
