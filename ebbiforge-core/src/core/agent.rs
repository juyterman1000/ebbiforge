use pyo3::prelude::*;

/// Represents an autonomous agent with a specific persona and instructions.
///
/// `Agent` is the core unit of logic in the CogOps ecosystem, encapsulating
/// instructions, behavior, and potential handoff targets.
#[pyclass]
#[derive(Clone, Debug)]
pub struct Agent {
    /// The unique identifier for the agent
    #[pyo3(get, set)]
    pub name: String,
    /// System instructions or "persona" that guides the model's behavior
    #[pyo3(get, set)]
    pub instructions: String,
    /// Targeted potential handoff agents by name
    handoff_names: Vec<String>,
}

#[pymethods]
impl Agent {
    #[new]
    pub fn new(name: String, instructions: String) -> Self {
        Agent {
            name,
            instructions,
            handoff_names: Vec::new(),
        }
    }

    /// Appends an agent name to the list of permissible handoff targets.
    pub fn add_handoff(&mut self, agent_name: String) {
        self.handoff_names.push(agent_name);
    }

    /// Returns a list of all agents this agent is authorized to hand off to.
    pub fn get_handoffs(&self) -> Vec<String> {
        self.handoff_names.clone()
    }

    /// Parses a model action string for handoff commands (e.g., "transfer_to_Reviewer").
    ///
    /// Returns the target agent name if the handoff is valid and authorized.
    pub fn get_handoff_target(&self, action: &str) -> Option<String> {
        if action.starts_with("transfer_to_") {
            let target = action.replace("transfer_to_", "");
            if self.handoff_names.contains(&target) {
                return Some(target);
            }
        }
        None
    }
}

/// A centralized registry for managing and resolving agent personas.
#[pyclass]
pub struct AgentRegistry {
    agents: std::collections::HashMap<String, Agent>,
}

#[pymethods]
impl AgentRegistry {
    #[new]
    pub fn new() -> Self {
        AgentRegistry {
            agents: std::collections::HashMap::new(),
        }
    }

    /// Registers a new `Agent` persona into the registry.
    pub fn register(&mut self, agent: Agent) {
        self.agents.insert(agent.name.clone(), agent);
    }

    /// Retrieves an `Agent` by its unique name.
    pub fn get(&self, name: &str) -> Option<Agent> {
        self.agents.get(name).cloned()
    }

    /// Resolves a handoff action from one agent to another.
    ///
    /// Verifies that the source agent is authorized to transfer control to the target.
    pub fn resolve_handoff(&self, from_agent: &str, action: &str) -> Option<Agent> {
        if let Some(agent) = self.agents.get(from_agent) {
            if let Some(target_name) = agent.get_handoff_target(action) {
                return self.agents.get(&target_name).cloned();
            }
        }
        None
    }
}
