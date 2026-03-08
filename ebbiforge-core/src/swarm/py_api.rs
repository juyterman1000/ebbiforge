use super::master_pipeline::SwarmEngineMaster;
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyclass]
pub struct PySwarmEngine {
    engine: SwarmEngineMaster,
}

#[pymethods]
impl PySwarmEngine {
    #[new]
    #[pyo3(signature = (n_agents=10_000_000, width=1000.0, height=1000.0))]
    pub fn new(n_agents: usize, width: f32, height: f32) -> Self {
        Self {
            engine: SwarmEngineMaster::new(n_agents, width, height),
        }
    }

    /// Advance the simulation by 1 tick.
    pub fn tick(&mut self) {
        self.engine.tick();
    }

    /// Inject pheromones into the stigmergic field.
    /// Channel 0: Resources, Channel 1: Danger, Channel 2: Trail,
    /// Channel 3: Hoarding Suppressor, Channel 4: Novelty, Channel 5: Alliance
    pub fn deposit_pheromone(&mut self, x: f32, y: f32, channel: usize, amount: f32) {
        self.engine.pheromones.deposit(x, y, channel, amount);
    }

    /// Extract macro-state metrics for analysis.
    pub fn get_macro_state(&self) -> PyObject {
        Python::with_gil(|py| {
            let dict = PyDict::new_bound(py);

            let n = self.engine.pool.n_agents as f32;

            // Mean surprise
            let surprise = self.engine.pool.surprise.as_slice();
            let sum_surprise: f32 = surprise.iter().sum();
            dict.set_item("mean_surprise", sum_surprise / n).unwrap();

            // Mean health
            let health = self.engine.pool.health.as_slice();
            let sum_health: f32 = health.iter().sum();
            dict.set_item("mean_health", sum_health / n).unwrap();

            // Agent count and tick
            dict.set_item("n_agents", self.engine.pool.n_agents).unwrap();
            dict.set_item("tick", self.engine.global_tick).unwrap();

            // Virtual memory estimate
            dict.set_item("virtual_mb", self.engine.pool.estimated_virtual_mb()).unwrap();

            dict.into()
        })
    }
}
