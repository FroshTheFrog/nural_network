use network::functions::create_random_weight_initializer;
use network::functions::RELU;
use network::layers::NeuralLayer;
use network::network::Network;
use network::types::Layer;

mod network;

fn main() {
    let layer0 = NeuralLayer::new(&RELU, create_random_weight_initializer::<-1, 1>(), 5, 5);
    let layer1 = NeuralLayer::new(&RELU, create_random_weight_initializer::<-1, 1>(), 5, 5);
    let layer2 = NeuralLayer::new(&RELU, create_random_weight_initializer::<-1, 1>(), 5, 3);

    let layers: Vec<&dyn Layer> = vec![&layer0, &layer1, &layer2];

    let test_network = Network::new(layers);

    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let output = test_network.feed_forward(input);

    println!("{:?}", output);
}
