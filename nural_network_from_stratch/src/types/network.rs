struct Perceptron {
    weights: Vec<f64>,
    bias: f64
}

impl Perceptron {
    fn new(size: usize, weight_initializer: fn() -> f64) -> Self {
        let mut weights = Vec::with_capacity(size);
        for _ in 0..size {
            weights.push(weight_initializer());
        }

        Perceptron {
            weights,
            bias: weight_initializer()
        }
    }
}

struct NeuralLayer {
    perceptions: Vec<Perceptron>,
    activation_function: fn(f64) -> f64
}

impl NeuralLayer {
    fn setup(size: usize, activation_function: fn(f64) -> f64, weight_initializer: fn() -> f64) -> impl Fn(usize) -> Self {
        move |input| {
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

trait Layer { }

impl Layer for NeuralLayer { }

pub struct Network {
    layers: Vec<Box<dyn Layer>>
}

impl Network {
    pub fn new(layer_sizes: Vec<usize>, activation_function: fn(f64) -> f64, weight_initializer: fn() -> f64) -> Self {
        let mut layers: Vec<NeuralLayer> = Vec::with_capacity(layer_sizes.len());
        for i in 0..layer_sizes.len() - 1 {
            layers.push(NeuralLayer::new(layer_sizes[i], layer_sizes[i + 1], activation_function, weight_initializer));
        }
    
        Network {
            layers
        }
    }
}
