pub type ActivationFunction = fn(f64) -> f64;
pub type WeightInitializer = fn() -> f64;

pub trait Layer {
    fn get_input_size(&self) -> usize;
    fn get_output_size(&self) -> usize;
    fn feed_forward(&self, inputs: &[f64]) -> Vec<f64>;
}
