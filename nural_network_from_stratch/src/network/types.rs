pub type WeightInitializer = fn() -> f64;

pub trait ActivationFunction {
    fn activate(&self, x: f64) -> f64;
    fn derivative(&self, x: f64) -> f64;
}

pub trait CostFunction {
    fn cost(&self, output: &[f64], expected: &[f64]) -> f64;
    fn derivative(&self, output: &[f64], expected: &[f64]) -> Vec<f64>;
}

pub trait Layer {
    fn get_input_size(&self) -> usize;
    fn get_output_size(&self) -> usize;
    fn feed_forward(&self, inputs: &[f64]) -> Vec<f64>;
}
