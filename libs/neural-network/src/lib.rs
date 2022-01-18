#![feature(crate_visibility_modifier)]

use rand::Rng;

use self::{layer::*, neuron::*};
pub use self::{layer_topology::*, network::*};

mod layer;
mod layer_topology;
mod network;
mod neuron;
