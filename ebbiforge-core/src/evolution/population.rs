//! Population Engine (Darwin Godel Machine)
//!
//! Manages a population of "Agent Genomes" that evolve over time.
//! Selects fittest variants based on benchmark performance.

use super::EvolutionConfig;
use pyo3::prelude::*;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use tracing::info;

/// Genetic representation of an agent
#[derive(Clone, Debug, Serialize, Deserialize)]
#[pyclass]
pub struct AgentGenome {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub generation: u32,
    #[pyo3(get, set)]
    pub system_prompt: String,
    #[pyo3(get, set)]
    pub tools: Vec<String>,
    #[pyo3(get, set)]
    pub temperature: f32,
    #[pyo3(get)]
    pub fitness_score: f32,
}

#[pymethods]
impl AgentGenome {
    #[new]
    pub fn new(id: String, system_prompt: String, tools: Vec<String>) -> Self {
        AgentGenome {
            id,
            generation: 0,
            system_prompt,
            tools,
            temperature: 0.7,
            fitness_score: 0.0,
        }
    }

    pub fn __repr__(&self) -> String {
        format!(
            "Genome(id={}, gen={}, score={:.2})",
            self.id, self.generation, self.fitness_score
        )
    }
}

/// Darwinian Evolution Engine
#[pyclass]
pub struct PopulationEngine {
    config: EvolutionConfig,
    population: Vec<AgentGenome>,
}

#[pymethods]
impl PopulationEngine {
    #[new]
    #[pyo3(signature = (config = None))]
    pub fn new(config: Option<EvolutionConfig>) -> Self {
        let cfg = config.unwrap_or_default();
        info!(
            "🧬 [PopulationEngine] Initialized (pop_size={}, mutation={:.2})",
            cfg.population_size, cfg.mutation_rate
        );

        PopulationEngine {
            config: cfg,
            population: Vec::new(),
        }
    }

    /// Initialize population from a base genome
    pub fn init_population(&mut self, base_genome: &AgentGenome) {
        self.population.clear();
        // Keep elite base
        let mut elite = base_genome.clone();
        elite.id = format!("{}_elite", base_genome.id);
        self.population.push(elite);

        // Mutate the rest
        for i in 1..self.config.population_size {
            let mut variant = base_genome.clone();
            variant.id = format!("{}_v{}", base_genome.id, i);
            self.mutate(&mut variant);
            self.population.push(variant);
        }
        info!(
            "🌱 [Population] Spawned {} variants for generation 0",
            self.config.population_size
        );
    }

    /// Run evolution step (selection and mutation)
    /// Expects population to be already scored
    pub fn evolve_generation(&mut self) -> AgentGenome {
        // 1. Sort by fitness (descending)
        self.population.sort_by(|a, b| {
            b.fitness_score
                .partial_cmp(&a.fitness_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let best = self.population[0].clone();
        info!(
            "[Evolution] Generation best: {} (score={:.2})",
            best.id, best.fitness_score
        );

        // 2. Selection & Reproduction
        let mut next_gen = Vec::new();

        // Eliteism: Keep top 1 unchanged
        let mut elite = self.population[0].clone();
        elite.generation += 1;
        next_gen.push(elite);

        // Generate rest from top 50%
        let mut rng = thread_rng();
        let parent_pool_size = (self.config.population_size + 1) / 2;

        while next_gen.len() < self.config.population_size {
            // Select random score-weighted parent
            let parent_idx = rng.gen_range(0..parent_pool_size);
            let mut child = self.population[parent_idx].clone();

            child.id = format!("gen{}_v{}", child.generation + 1, next_gen.len());
            child.generation += 1;

            // Mutate
            self.mutate(&mut child);
            next_gen.push(child);
        }

        self.population = next_gen;
        best
    }

    /// Score a genome (external benchmark callback)
    pub fn update_fitness(&mut self, genome_id: String, score: f32) {
        if let Some(genome) = self.population.iter_mut().find(|g| g.id == genome_id) {
            genome.fitness_score = score;
        }
    }

    /// Get current population
    pub fn get_population(&self) -> Vec<AgentGenome> {
        self.population.clone()
    }
}

impl PopulationEngine {
    fn mutate(&self, genome: &mut AgentGenome) {
        let mut rng = thread_rng();

        // Mutate Temperature (Hyperparameter)
        if rng.gen::<f32>() < self.config.mutation_rate {
            let change = rng.gen_range(-0.2..0.2);
            genome.temperature = (genome.temperature + change).clamp(0.0, 1.0);
        }

        // Mutate System Prompt (Strategy)
        if rng.gen::<f32>() < self.config.mutation_rate {
            let additions = vec![
                " Think step-by-step.",
                " Be concise.",
                " Double check your work.",
                " Use tools aggressively.",
                " Verify assumptions first.",
            ];
            if additions.is_empty() {
                return;
            }
            let variant = additions.choose(&mut rng).unwrap_or(&additions[0]);

            if !genome.system_prompt.contains(variant) {
                genome.system_prompt.push_str(variant);
            }
        }

        // Mutate Tools (Skillset): randomly add or remove a tool capability
        let available_tools = [
            "web_search", "code_execution", "math_solver",
            "data_analysis", "document_retrieval", "api_caller",
        ];
        if rng.gen::<f32>() < 0.15 {
            // 15% chance: toggle a random tool
            let tool = available_tools.choose(&mut rng).unwrap_or(&"web_search");
            if genome.tools.contains(&tool.to_string()) {
                genome.tools.retain(|t| t != *tool);
            } else {
                genome.tools.push(tool.to_string());
            }
        }
    }
}
