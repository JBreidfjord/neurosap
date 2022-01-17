use pyo3::prelude::*;

#[pymodule]
fn neurosap(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<agent::Agent>()?;
    Ok(())
}
