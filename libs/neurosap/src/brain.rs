use crate::*;

pub struct Brain {
    crate nn: nn::Network,
}

impl Brain {
    pub fn random(rng: &mut dyn RngCore) -> Brain {
        Brain {
            nn: nn::Network::random(rng, &Self::topology()),
        }
    }

    crate fn as_chromosome(&self) -> ga::Chromosome {
        self.nn.weights().collect()
    }

    crate fn from_chromosome(chromosome: ga::Chromosome) -> Brain {
        Brain {
            nn: nn::Network::from_weights(&Self::topology(), chromosome),
        }
    }

    fn topology() -> [nn::LayerTopology; 3] {
        [
            nn::LayerTopology {
                neurons: 51, // Input size
            },
            nn::LayerTopology {
                neurons: 128, // Hidden layer size
            },
            nn::LayerTopology {
                neurons: 62, // Output size
            },
        ]
    }
}
