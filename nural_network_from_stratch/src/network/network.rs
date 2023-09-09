use crate::network::types::Layer;

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
