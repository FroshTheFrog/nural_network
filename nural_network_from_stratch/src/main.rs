use network::functions::create_random_weight_initializer;
use network::functions::RELU;
use network::layers::NeuralLayer;
use network::network::Network;
use network::types::Layer;

use crate::network::functions::SIGMOID;

mod network;

fn main() {
    let mut layer0 = NeuralLayer::new(&RELU, create_random_weight_initializer::<-1, 1>(), 5, 5);
    let mut layer1 = NeuralLayer::new(&SIGMOID, create_random_weight_initializer::<-1, 1>(), 5, 5);
    let mut layer2 = NeuralLayer::new(&RELU, create_random_weight_initializer::<-1, 1>(), 5, 3);

    let layers: Vec<&mut dyn Layer> = vec![&mut layer0, &mut layer1, &mut layer2];

    let test_network = Network::new(layers);

    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let (output, derivative) = test_network.feed_forward(input, false);

    println!("{:?}", output);
}
