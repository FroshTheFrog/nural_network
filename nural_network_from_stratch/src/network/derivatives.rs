pub type WeightInitializer = fn() -> f64;

pub struct PerceptronDerivative {
    pub derivatives_w: Vec<f64>,
    pub derivative_b: f64,
}

pub type LayerDerivative = Vec<PerceptronDerivative>;

pub type NetworkDerivative = Vec<LayerDerivative>;

impl PerceptronDerivative {
    pub fn multiply(&mut self, value: f64) {
        self.derivatives_w
            .iter_mut()
            .for_each(|weight| *weight *= value);
        self.derivative_b *= value;
    }
}

pub fn multiply_layer_derivative_by_vector(derivative: &mut LayerDerivative, values: &Vec<f64>) {
    derivative
        .iter_mut()
        .zip(values.iter())
        .for_each(|(perceptron_derivative, value)| {
            perceptron_derivative.multiply(*value);
        });
}
