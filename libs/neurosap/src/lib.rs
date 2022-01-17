#![feature(crate_visibility_modifier)]

use nalgebra as na;
use rand::{Rng, RngCore};

use lib_genetic_algorithm as ga;
use lib_neural_network as nn;

use self::agent_individual::*;
pub use self::{agent::*, brain::*};

mod agent;
mod agent_individual;
mod brain;

pub struct NeuroSAP {
    population: Vec<Agent>,
    ga: ga::GeneticAlgorithm<ga::RouletteWheelSelection>,
}

impl NeuroSAP {
    pub fn random(rng: &mut dyn RngCore) -> NeuroSAP {
        let population: Vec<Agent> = (0..50).map(|_| Agent::random(rng)).collect();
        let ga = ga::GeneticAlgorithm::new(
            ga::RouletteWheelSelection::new(),
            ga::UniformCrossover::new(),
            ga::GaussianMutation::new(0.05, 0.3),
        );

        NeuroSAP { population, ga }
    }

    pub fn population(&self) -> &Vec<Agent> {
        &self.population
    }

    pub fn evolve(&mut self, rng: &mut dyn RngCore) -> bool {
        if self.population.iter().any(|agent| !agent.finished) {
            return false;
        }

        // Transform Vec<Agent> into Vec<AgentIndividual>
        let population: Vec<_> = self
            .population
            .iter()
            .map(AgentIndividual::from_agent)
            .collect();

        // Evolve population
        let new_population = self.ga.step(rng, &population);

        // Transform Vec<AgentIndividual> into Vec<Agent>
        self.population = new_population
            .iter()
            .map(AgentIndividual::into_agent)
            .collect();

        true
    }
}
