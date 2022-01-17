use crate::*;

#[derive(Clone, Debug)]
pub struct Chromosome {
    genes: Vec<f32>,
}

impl Chromosome {
    pub fn len(&self) -> usize {
        self.genes.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &f32> {
        self.genes.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut f32> {
        self.genes.iter_mut()
    }

    pub fn split_at(&self, index: usize) -> [Chromosome; 2] {
        let (left, right) = self.genes.split_at(index);
        [
            Chromosome {
                genes: left.to_vec(),
            },
            Chromosome {
                genes: right.to_vec(),
            },
        ]
    }
}

impl Index<usize> for Chromosome {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        &self.genes[index]
    }
}

impl FromIterator<f32> for Chromosome {
    fn from_iter<T: IntoIterator<Item = f32>>(iter: T) -> Chromosome {
        Chromosome {
            genes: iter.into_iter().collect(),
        }
    }
}

impl IntoIterator for Chromosome {
    type Item = f32;
    type IntoIter = impl Iterator<Item = f32>;

    fn into_iter(self) -> Self::IntoIter {
        self.genes.into_iter()
    }
}

impl PartialEq for Chromosome {
    fn eq(&self, other: &Self) -> bool {
        approx::relative_eq!(self.genes.as_slice(), other.genes.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod index {
        use super::*;

        #[test]
        fn test() {
            let chromosome = Chromosome {
                genes: vec![1.0, 2.0, 3.0],
            };

            assert_eq!(chromosome[0], 1.0);
            assert_eq!(chromosome[1], 2.0);
            assert_eq!(chromosome[2], 3.0);
        }
    }

    mod from_iterator {
        use super::*;

        #[test]
        fn test() {
            let chromosome: Chromosome = vec![1.0, 2.0, 3.0].into_iter().collect();

            assert_eq!(chromosome[0], 1.0);
            assert_eq!(chromosome[1], 2.0);
            assert_eq!(chromosome[2], 3.0);
        }
    }

    mod into_iterator {
        use super::*;

        #[test]
        fn test() {
            let chromosome = Chromosome {
                genes: vec![1.0, 2.0, 3.0],
            };

            let genes: Vec<_> = chromosome.into_iter().collect();

            assert_eq!(genes.len(), 3);
            assert_eq!(genes[0], 1.0);
            assert_eq!(genes[1], 2.0);
            assert_eq!(genes[2], 3.0);
        }
    }
}
