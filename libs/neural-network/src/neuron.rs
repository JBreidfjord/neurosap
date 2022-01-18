use crate::*;

crate struct Neuron {
    crate weights: Vec<f32>,
    crate bias: f32,
}

impl Neuron {
    pub fn random(rng: &mut dyn rand::RngCore, output_size: usize) -> Neuron {
        let weights = (0..output_size)
            .map(|_| rng.gen_range(-1.0..=1.0))
            .collect();

        let bias = rng.gen_range(-1.0..=1.0);

        Neuron { weights, bias }
    }

    crate fn propagate(&self, inputs: &Vec<f32>, activate: Option<bool>) -> f32 {
        let sum = inputs
            .iter()
            .zip(&self.weights)
            .map(|(input, weight)| weight * input)
            .sum::<f32>();

        let activate = activate.unwrap_or(true);
        if activate {
            (sum + self.bias).max(0.0)
        } else {
            sum + self.bias
        }
    }

    pub fn from_weights(output_neurons: usize, weights: &mut dyn Iterator<Item = f32>) -> Neuron {
        let bias = weights.next().expect("Not enough weights!");
        let weights = (0..output_neurons)
            .map(|_| weights.next().expect("Not enough weights!"))
            .collect();

        Neuron { weights, bias }
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
            let neuron = Neuron::random(&mut rng, 2);

            assert_relative_eq!(neuron.bias, 0.8181262);
            assert_relative_eq!(neuron.weights.as_slice(), [-0.6255188, 0.67383957].as_ref());
        }
    }

    mod propagate {
        use super::*;
        use approx::assert_relative_eq;

        #[test]
        fn test() {
            let neuron = Neuron {
                weights: vec![0.5, 0.5],
                bias: 0.5,
            };

            // Ensures ReLU activation function is used
            assert_relative_eq!(neuron.propagate(&vec![-10.0, -10.0], None), 0.0);

            // Test deactivating the activation function
            assert_relative_eq!(neuron.propagate(&vec![-10.0, -10.0], Some(false)), -9.5);

            assert_relative_eq!(neuron.propagate(&vec![1.0, 0.5], None), 1.25);
        }
    }
}
