use network::functions::create_random_weight_initializer;
use network::functions::Relu;
use network::layers::Layer;
use network::layers::NeuralLayer;
use network::network::Network;

use crate::network::functions::MeanSquaredError;

mod network;

fn main() {
    let mut layer0 = NeuralLayer::new(&Relu, create_random_weight_initializer::<-1, 1>(), 3, 3);
    let mut layer1 = NeuralLayer::new(&Relu, create_random_weight_initializer::<-1, 1>(), 3, 3);
    let mut layer2 = NeuralLayer::new(&Relu, create_random_weight_initializer::<-1, 1>(), 3, 2);
    let mut layer3 = NeuralLayer::new(&Relu, create_random_weight_initializer::<-1, 1>(), 2, 2);
    let mut layer4 = NeuralLayer::new(&Relu, create_random_weight_initializer::<-1, 1>(), 2, 2);

    let layers: Vec<&mut dyn Layer> = vec![
        &mut layer0,
        &mut layer1,
        &mut layer2,
        &mut layer3,
        &mut layer4,
    ];

    let mut test_network = Network::new(layers);

    let input = vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let expected = vec![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]];

    let number_of_epocs = 10;

    let learning_rate = 0.01;

    for _ in 0..number_of_epocs {
        for (input, expected) in input.iter().zip(expected.iter()) {
            let cost = test_network.train(
                input.to_vec(),
                expected.to_vec(),
                &MeanSquaredError,
                learning_rate,
            );

            println!("Cost: {}", cost);
        }
    }
}
