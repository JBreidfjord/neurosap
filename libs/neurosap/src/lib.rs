#![feature(crate_visibility_modifier)]

use rand::RngCore;

use std::fs::File;
use std::path::Path;

use serde::{Deserialize, Serialize};

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

#[derive(Deserialize, Serialize)]
crate struct AgentArchive {
    chromosome: Vec<f32>,
    fitness: f32,
    finished: bool,
}

impl NeuroSAP {
    pub fn random(population_count: usize, rng: &mut dyn RngCore) -> NeuroSAP {
        let population: Vec<Agent> = (0..population_count).map(|_| Agent::random(rng)).collect();
        let ga = ga::GeneticAlgorithm::new(
            ga::RouletteWheelSelection::new(),
            ga::UniformCrossover::new(),
            ga::GaussianMutation::new(0.05, 0.3),
        );

        NeuroSAP { population, ga }
    }

    pub fn step(&self, inputs: Vec<f32>) -> Vec<f32> {
        let mut outputs = Vec::<f32>::new();
        for agent in self.population.iter() {
            if agent.finished {
                continue;
            }

            outputs = agent.brain.nn.propagate(inputs);
            break;
        }

        if outputs.is_empty() {
            panic!("No agents found!");
        } else {
            outputs
        }
    }

    pub fn finish_agent(&mut self, fitness: f32) {
        for agent in self.population.iter_mut() {
            if !agent.finished {
                agent.finish(fitness);
                break;
            }
        }
    }

    pub fn evolve(&mut self, rng: &mut dyn RngCore) {
        if self.population.iter().any(|agent| !agent.finished) {
            panic!("Population not finished!");
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
            .into_iter()
            .map(|individual| individual.into_agent())
            .collect();
    }

    pub fn save(&self, filename: &str) {
        let path = Path::new(filename);
        let file = File::create(path).unwrap();
        let archives: Vec<AgentArchive> = self
            .population
            .iter()
            .map(|agent| AgentArchive::from(agent))
            .collect();

        serde_json::to_writer(file, &archives).unwrap();
    }

    pub fn population_count(&self) -> usize {
        self.population.len()
    }
}

pub fn load(filename: &str) -> NeuroSAP {
    // Load population
    let path = Path::new(filename);
    let file = File::open(path).unwrap();
    let archives: Vec<AgentArchive> = serde_json::from_reader(file).unwrap();
    let population = archives
        .into_iter()
        .map(|archive| Agent::from(archive))
        .collect();
    let ga = ga::GeneticAlgorithm::new(
        ga::RouletteWheelSelection::new(),
        ga::UniformCrossover::new(),
        ga::GaussianMutation::new(0.05, 0.3),
    );

    NeuroSAP { population, ga }
}

impl From<&Agent> for AgentArchive {
    fn from(agent: &Agent) -> Self {
        AgentArchive {
            chromosome: agent.as_chromosome().into_iter().collect(),
            fitness: agent.fitness,
            finished: agent.finished,
        }
    }
}
