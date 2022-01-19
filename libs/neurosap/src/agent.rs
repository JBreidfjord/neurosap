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

    crate fn from_chromosome_with_fitness(
        chromosome: ga::Chromosome,
        fitness: f32,
        finished: bool,
    ) -> Agent {
        Agent {
            brain: Brain::from_chromosome(chromosome),
            fitness,
            finished,
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

impl From<AgentArchive> for Agent {
    fn from(archive: AgentArchive) -> Self {
        Agent::from_chromosome_with_fitness(
            archive.chromosome.into_iter().collect(),
            archive.fitness,
            archive.finished,
        )
    }
}
