use network::functions::create_random_weight_initializer;
use network::functions::ReLU;
use network::layers::NeuralLayer;
use network::network::Network;
use network::types::Layer;

mod network;

fn main() {
    let layer0 = NeuralLayer::new(ReLU, create_random_weight_initializer::<-1, 1>(), 5, 5);
    let layer1 = NeuralLayer::new(ReLU, create_random_weight_initializer::<-1, 1>(), 5, 5);
    let layer2 = NeuralLayer::new(ReLU, create_random_weight_initializer::<-1, 1>(), 5, 3);

    let layers: Vec<Box<dyn Layer>> = vec![Box::new(layer0), Box::new(layer1), Box::new(layer2)];

    let test_network = Network::new(layers);
}
