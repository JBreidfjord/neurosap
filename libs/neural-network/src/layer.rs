use crate::*;

crate struct Layer {
    crate neurons: Vec<Neuron>,
}

impl Layer {
    pub fn random(
        rng: &mut dyn rand::RngCore,
        input_neurons: usize,
        output_neurons: usize,
    ) -> Layer {
        let neurons = (0..output_neurons)
            .map(|_| Neuron::random(rng, input_neurons))
            .collect();

        Layer { neurons }
    }

    crate fn propagate(&self, inputs: Vec<f32>, activate: Option<bool>) -> Vec<f32> {
        self.neurons
            .iter()
            .map(|neuron| neuron.propagate(&inputs, activate))
            .collect()
    }

    pub fn from_weights(
        input_size: usize,
        output_size: usize,
        weights: &mut dyn Iterator<Item = f32>,
    ) -> Layer {
        let neurons = (0..output_size)
            .map(|_| Neuron::from_weights(input_size, weights))
            .collect();

        Layer { neurons }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod random {
        use super::*;
        use approx::assert_relative_eq;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        #[test]
        fn test() {
            let mut rng = ChaCha8Rng::from_seed(Default::default());
            let layer = Layer::random(&mut rng, 2, 2);

            assert_relative_eq!(layer.neurons[0].bias, 0.8181262);
            assert_relative_eq!(
                layer.neurons[0].weights.as_slice(),
                [-0.6255188, 0.67383957].as_ref()
            );

            assert_relative_eq!(layer.neurons[1].bias, -0.53516835);
            assert_relative_eq!(
                layer.neurons[1].weights.as_slice(),
                [0.26284897, 0.5238807].as_ref()
            );
        }
    }

    mod propagate {
        use super::*;
        use approx::assert_relative_eq;

        #[test]
        fn test() {
            let layer = Layer {
                neurons: vec![
                    Neuron {
                        weights: vec![0.25, 0.75],
                        bias: 0.0,
                    },
                    Neuron {
                        weights: vec![0.5, 0.5],
                        bias: 0.5,
                    },
                ],
            };

            let prop = layer.propagate(vec![0.3, 0.6], None);
            assert_relative_eq!(prop.as_slice(), [0.525, 0.95].as_ref());
        }
    }
}
