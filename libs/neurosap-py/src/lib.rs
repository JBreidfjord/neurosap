use pyo3::prelude::*;

use neurosap_core as nsap;

#[pyclass(unsendable)]
struct NeuroSAP {
    ns: nsap::NeuroSAP,
}

#[pymethods]
impl NeuroSAP {
    #[new]
    fn new(population_count: usize) -> NeuroSAP {
        NeuroSAP {
            ns: nsap::NeuroSAP::random(population_count, &mut rand::thread_rng()),
        }
    }

    fn step(&self, inputs: Vec<f32>) -> Vec<f32> {
        self.ns.step(inputs)
    }

    fn evolve(&mut self) {
        self.ns.evolve(&mut rand::thread_rng());
    }

    fn finish_agent(&mut self, fitness: f32) {
        self.ns.finish_agent(fitness);
    }

    fn save(&self, filename: &str) {
        self.ns.save(filename);
    }

    fn population_count(&self) -> usize {
        self.ns.population_count()
    }
}

#[pyfunction]
fn load(filename: &str) -> NeuroSAP {
    NeuroSAP {
        ns: nsap::load(filename),
    }
}

#[pymodule]
fn neurosap(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<NeuroSAP>()?;
    m.add_function(wrap_pyfunction!(load, m)?)?;
    Ok(())
}
