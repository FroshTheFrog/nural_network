use crate::network::types::Layer;

pub struct Network<'a> {
    layers: Vec<&'a dyn Layer>,
}

impl<'a> Network<'a> {
    pub fn new(layers: Vec<&'a dyn Layer>) -> Self {
        assert!(
            Self::verify_layer_dimensions(&layers),
            "Invalid layer dimensions"
        );
        Self { layers }
    }

    fn verify_layer_dimensions(layers: &Vec<&'a dyn Layer>) -> bool {
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
