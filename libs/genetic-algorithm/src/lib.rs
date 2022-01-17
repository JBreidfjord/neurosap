#![feature(type_alias_impl_trait)]
#![feature(crate_visibility_modifier)]

use rand::seq::SliceRandom;
use rand::Rng;
use std::iter::FromIterator;
use std::ops::Index;

pub use self::{chromosome::*, crossover::*, mutation::*, selection::*};

mod chromosome;
mod crossover;
mod mutation;
mod selection;

pub struct GeneticAlgorithm<S> {
    selection_method: S,
    crossover_method: Box<dyn CrossoverMethod>,
    mutation_method: Box<dyn MutationMethod>,
}

pub trait Individual {
    fn fitness(&self) -> f32;
    fn chromosome(&self) -> &Chromosome;
    fn create(chromosome: Chromosome) -> Self;
}

impl<S> GeneticAlgorithm<S>
where
    S: SelectionMethod,
{
    pub fn new(
        selection_method: S,
        crossover_method: impl CrossoverMethod + 'static,
        mutation_method: impl MutationMethod + 'static,
    ) -> GeneticAlgorithm<S> {
        GeneticAlgorithm {
            selection_method,
            crossover_method: Box::new(crossover_method),
            mutation_method: Box::new(mutation_method),
        }
    }

    pub fn step<I>(&self, rng: &mut dyn rand::RngCore, population: &[I]) -> Vec<I>
    where
        I: Individual,
    {
        assert!(!population.is_empty());

        let new_population = (0..population.len())
            .map(|_| {
                // Selection
                let parent_a = self.selection_method.select(rng, population).chromosome();
                let parent_b = self.selection_method.select(rng, population).chromosome();

                // Crossover
                let mut child = self.crossover_method.crossover(rng, parent_a, parent_b);

                // Mutation
                self.mutation_method.mutate(rng, &mut child);

                I::create(child)
            })
            .collect();

        new_population
    }

    pub fn breed<I>(&self, rng: &mut dyn rand::RngCore, parent_a: I, parent_b: I) -> I
    where
        I: Individual,
    {
        // No Selection
        // Crossover
        let mut child =
            self.crossover_method
                .crossover(rng, parent_a.chromosome(), parent_b.chromosome());

        // Mutation
        self.mutation_method.mutate(rng, &mut child);

        I::create(child)
    }
}

#[cfg(test)]
#[derive(Clone, Debug, PartialEq)]
pub enum TestIndividual {
    WithChromosome { chromosome: Chromosome },
    WithFitness { fitness: f32 },
}

#[cfg(test)]
impl TestIndividual {
    pub fn new(fitness: f32) -> TestIndividual {
        Self::WithFitness { fitness }
    }
}

#[cfg(test)]
impl Individual for TestIndividual {
    fn fitness(&self) -> f32 {
        match self {
            Self::WithChromosome { chromosome } => chromosome.iter().sum(),
            Self::WithFitness { fitness } => *fitness,
        }
    }

    fn chromosome(&self) -> &Chromosome {
        match self {
            Self::WithChromosome { chromosome } => chromosome,
            Self::WithFitness { .. } => {
                panic!("Not supported for TestIndividual::WithFitness")
            }
        }
    }

    fn create(chromosome: Chromosome) -> Self {
        Self::WithChromosome { chromosome }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod geneticalgorithm {
        use super::*;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        fn individual(genes: &[f32]) -> TestIndividual {
            let chromosome = genes.iter().cloned().collect();

            TestIndividual::create(chromosome)
        }

        #[test]
        fn test() {
            let mut rng = ChaCha8Rng::from_seed(Default::default());

            let ga = GeneticAlgorithm::new(
                RouletteWheelSelection::new(),
                UniformCrossover::new(),
                GaussianMutation::new(0.25, 1.0),
            );

            let mut population = vec![
                individual(&[0.0, 0.0, 0.0]), // fitness = 0.0
                individual(&[1.0, 1.0, 1.0]), // fitness = 3.0
                individual(&[2.0, 2.0, 2.0]), // fitness = 6.0
                individual(&[3.0, 3.0, 3.0]), // fitness = 9.0
                individual(&[4.0, 4.0, 4.0]), // fitness = 12.0
            ];

            for _ in 0..5 {
                population = ga.step(&mut rng, &population);
            }

            let expected_population = vec![
                individual(&[2.2071722, 1.443665, 4.9321785]),
                individual(&[2.0, 3.0, 3.3834975]),
                individual(&[2.0, 3.0, 2.5290549]),
                individual(&[2.2071722, 3.5167656, 3.3899784]),
                individual(&[2.0, 3.0, 2.8409562]),
            ];

            assert_eq!(population, expected_population);
        }
    }
}
