#![feature(crate_visibility_modifier)]

use nalgebra as na;
use rand::{Rng, RngCore};

use lib_genetic_algorithm as ga;
use lib_neural_network as nn;

pub use self::{agent::*, brain::*};

mod agent;
mod brain;
