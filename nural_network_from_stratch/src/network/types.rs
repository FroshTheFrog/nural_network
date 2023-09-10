pub type WeightInitializer = fn() -> f64;

pub struct PerceptronDerivative {
    pub derivatives_w: Vec<f64>,
    pub derivative_b: f64,
}

impl PerceptronDerivative {
    pub fn multiply(&mut self, value: f64) {
        self.derivatives_w
            .iter_mut()
            .for_each(|weight| *weight *= value);
        self.derivative_b *= value;
    }
}

pub type LayerDerivative = Vec<PerceptronDerivative>;

pub type NetworkDerivative = Vec<LayerDerivative>;

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
    fn feed_forward(
        &self,
        inputs: &[f64],
        return_derivative: bool,
    ) -> (Vec<f64>, Option<LayerDerivative>);

    fn update(&mut self, derivative: &LayerDerivative, learning_rate: f64);
}
