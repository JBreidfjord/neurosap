use crate::*;

pub struct Agent {
    crate brain: Brain,
    crate fitness: f32,
    crate finished: bool,
}

impl Agent {
    pub fn random(rng: &mut dyn RngCore) -> Agent {
        Agent {
            brain: Brain::random(rng),
            fitness: 0.0,
            finished: false,
        }
    }

    crate fn as_chromosome(&self) -> ga::Chromosome {
        self.brain.as_chromosome()
    }

    crate fn from_chromosome(chromosome: ga::Chromosome) -> Agent {
        Agent {
            brain: Brain::from_chromosome(chromosome),
            fitness: 0.0,
            finished: false,
        }
    }

    pub fn fitness(&self) -> f32 {
        self.fitness
    }

    pub fn finish(&mut self, fitness: f32) {
        self.fitness = fitness;
        self.finished = true;
    }
}
