use crate::*;

struct Agent {
    crate brain: Brain,
    crate fitness: f32,
}

impl Agent {
    pub fn random(rng: &mut dyn RngCore) -> Agent {
        Agent {
            brain: Brain,
            fitness: 0.0,
        }
    }

    fn new(brain: Brain, rng: &mut dyn RngCore) -> Agent {
        Agent {
            brain,
            fitness: 0.0,
        }
    }

    crate fn as_chromosome(&self) -> ga::Chromosome {
        self.brain.as_chromosome()
    }

    crate fn from_chromosome(chromosome: ga::Chromosome) -> Agent {
        Agent {
            brain: Brain::from_chromosome(chromosome),
            fitness: 0.0,
        }
    }

    pub fn fitness(&self) -> f32 {
        self.fitness
    }
}
