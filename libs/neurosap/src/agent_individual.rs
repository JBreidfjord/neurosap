use crate::*;

pub struct AgentIndividual {
    fitness: f32,
    chromosome: ga::Chromosome,
}

impl ga::Individual for AgentIndividual {
    fn create(chromosome: ga::Chromosome) -> AgentIndividual {
        AgentIndividual {
            fitness: 0.0,
            chromosome,
        }
    }

    fn chromosome(&self) -> &ga::Chromosome {
        &self.chromosome
    }

    fn fitness(&self) -> f32 {
        self.fitness
    }
}

impl AgentIndividual {
    pub fn from_agent(agent: &Agent) -> AgentIndividual {
        AgentIndividual {
            fitness: agent.fitness(),
            chromosome: agent.as_chromosome(),
        }
    }

    pub fn into_agent(self) -> Agent {
        Agent::from_chromosome(self.chromosome)
    }
}
