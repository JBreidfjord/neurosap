use crate::*;

pub trait MutationMethod {
    fn mutate(&self, rng: &mut dyn rand::RngCore, child: &mut Chromosome);
}

pub struct GaussianMutation {
    // Probability of changing a gene
    rate: f32,
    // Magnitude of change
    factor: f32,
}

impl GaussianMutation {
    pub fn new(rate: f32, factor: f32) -> GaussianMutation {
        assert!(rate >= 0.0 && rate <= 1.0);

        GaussianMutation { rate, factor }
    }
}

impl MutationMethod for GaussianMutation {
    fn mutate(&self, rng: &mut dyn rand::RngCore, child: &mut Chromosome) {
        for gene in child.iter_mut() {
            let sign = if rng.gen_bool(0.5) { -1.0 } else { 1.0 };

            if rng.gen_bool(self.rate as _) {
                *gene += sign * self.factor * rng.gen::<f32>();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod gaussian {
        use super::*;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        fn actual(rate: f32, factor: f32) -> Vec<f32> {
            let mut child = vec![1.0, 2.0, 3.0, 4.0, 5.0].into_iter().collect();

            let mut rng = ChaCha8Rng::from_seed(Default::default());

            GaussianMutation::new(rate, factor).mutate(&mut rng, &mut child);

            child.into_iter().collect()
        }

        mod given_zero_rate {
            use approx::assert_relative_eq;

            fn actual(factor: f32) -> Vec<f32> {
                super::actual(0.0, factor)
            }

            mod and_zero_factor {
                use super::*;

                #[test]
                fn does_not_change_original_chromosome() {
                    let actual = actual(0.0);
                    let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0];

                    assert_relative_eq!(actual.as_slice(), expected.as_slice());
                }
            }

            mod and_nonzero_factor {
                use super::*;

                #[test]
                fn does_not_change_original_chromosome() {
                    let actual = actual(0.5);
                    let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0];

                    assert_relative_eq!(actual.as_slice(), expected.as_slice());
                }
            }
        }
        mod given_half_rate {
            use approx::assert_relative_eq;

            fn actual(factor: f32) -> Vec<f32> {
                super::actual(0.5, factor)
            }

            mod and_zero_factor {
                use super::*;

                #[test]
                fn does_not_change_original_chromosome() {
                    let actual = actual(0.0);
                    let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0];

                    assert_relative_eq!(actual.as_slice(), expected.as_slice());
                }
            }

            mod and_nonzero_factor {
                use super::*;

                #[test]
                fn slightly_changes_original_chromosome() {
                    let actual = actual(0.5);
                    let expected = vec![1.0, 1.7756249, 3.0, 4.1596804, 5.0];

                    assert_relative_eq!(actual.as_slice(), expected.as_slice());
                }
            }
        }
        mod given_max_rate {
            use approx::assert_relative_eq;

            fn actual(factor: f32) -> Vec<f32> {
                super::actual(1.0, factor)
            }

            mod and_zero_factor {
                use super::*;

                #[test]
                fn does_not_change_original_chromosome() {
                    let actual = actual(0.0);
                    let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0];

                    assert_relative_eq!(actual.as_slice(), expected.as_slice());
                }
            }

            mod and_nonzero_factor {
                use super::*;

                #[test]
                fn changes_original_chromosome() {
                    let actual = actual(0.5);
                    let expected = vec![1.4545316, 2.1162078, 2.7756248, 3.9505124, 4.638691];

                    assert_relative_eq!(actual.as_slice(), expected.as_slice());
                }
            }
        }
    }
}
