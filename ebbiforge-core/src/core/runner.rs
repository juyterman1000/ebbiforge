use crate::core::agent::{Agent, AgentRegistry};
use crate::core::config::EbbiforgeConfig;
use crate::core::middleware::{EbbiforgeContext, Middleware, MiddlewarePipeline};
use crate::core::tools::{execute_tool, get_tool_definitions, ToolResult};
use crate::{HistoryBuffer, TrajectoryPoint};
use pyo3::prelude::*;
use serde_json::json;
use std::sync::{Arc, Mutex, OnceLock};
use std::collections::HashMap;
use tokio::runtime::Runtime;
use tokio::task::AbortHandle;
use tracing::info;

// Prevent OS-level thread and socket exhaustion by sharing the core async I/O drivers
use std::env;
static SHARED_RUNTIME: OnceLock<Arc<Runtime>> = OnceLock::new();
static SHARED_CLIENT: OnceLock<reqwest::Client> = OnceLock::new();

fn get_shared_runtime() -> Arc<Runtime> {
    SHARED_RUNTIME.get_or_init(|| {
        Arc::new(
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .expect("Failed to create global tokio runtime")
        )
    }).clone()
}

fn get_shared_client() -> reqwest::Client {
    SHARED_CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .gzip(true)
            .brotli(true)
            .build()
            .expect("Failed to create global reqwest client")
    }).clone()
}

/// Orchestrates the lifecycle of an AI agent within the middleware pipeline.
///
/// `AgentGraph` manages agent registration, middleware injection, and the
/// execution loop for individual tasks with REAL tool use via ReAct pattern.
pub struct AgentGraph {
    pub config: EbbiforgeConfig,
    pipeline: MiddlewarePipeline,
    registry: AgentRegistry,
    pub runtime: Arc<Runtime>,
    client: reqwest::Client,
    active_tasks: Arc<Mutex<HashMap<String, AbortHandle>>>,
}

impl AgentGraph {
    pub fn new() -> Self {
        AgentGraph {
            config: EbbiforgeConfig::default(),
            pipeline: MiddlewarePipeline::new(),
            registry: AgentRegistry::new(),
            runtime: get_shared_runtime(),
            client: get_shared_client(),
            active_tasks: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Configures the graph with a custom set of options.
    pub fn with_config(config: EbbiforgeConfig) -> Self {
        AgentGraph {
            config,
            pipeline: MiddlewarePipeline::new(),
            registry: AgentRegistry::new(),
            runtime: get_shared_runtime(),
            client: get_shared_client(),
            active_tasks: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Registers a new `Agent` persona for use within the graph.
    pub fn register_agent(&mut self, agent: Agent) {
        self.registry.register(agent);
    }

    /// Attaches a middleware layer to the execution pipeline.
    pub fn use_middleware(&mut self, middleware: Box<dyn Middleware>) {
        self.pipeline.add(middleware);
    }

    /// Executes a task using the ReAct pattern with real tool use.
    ///
    /// ReAct Loop:
    /// 1. Send task + history to model with tool definitions
    /// 2. Model either returns text OR requests a tool call
    /// 3. If tool call: execute tool, add observation to history, repeat
    /// 4. If finish() called: return success with final answer
    /// 5. Max 10 iterations to prevent infinite loops
    pub async fn run_task(
        &self,
        task_id: &str,
        buffer: &HistoryBuffer,
        agent_name: Option<&str>,
    ) -> Result<EbbiforgeContext, String> {
        let display_name = agent_name.unwrap_or("DefaultAgent");
        info!(
            "▶️ [AgentGraph] Starting Task: {} (Agent: {})",
            task_id, display_name
        );

        // Initialize Context
        let prompt = agent_name
            .and_then(|n| self.registry.get(n))
            .map(|a| a.instructions.clone())
            .unwrap_or_else(|| "You are a helpful research assistant. Use the provided tools to find real information and answer questions accurately.".to_string());

        let mut ctx = EbbiforgeContext::new(task_id.to_string(), prompt.clone());

        // Copy trajectory reference
        for point in buffer.get_raw() {
            ctx.add_trajectory_point(point);
        }

        // STEP 1: Pre-Execution Hooks
        info!("🔄 [AgentGraph] Stage: Pre-Step Hooks ({})", display_name);
        self.pipeline.run_before(&mut ctx)?;

        if ctx.should_stop {
            info!("⏹️ [AgentGraph] Stopped: {:?}", ctx.stop_reason);
            return Ok(ctx);
        }

        // STEP 2: ReAct Loop with Tool Use
        info!(" [AgentGraph] Stage: ReAct Loop with Tool Use");

        let api_key =
            env::var("MODEL_API_KEY").map_err(|_| "MODEL_API_KEY not found".to_string())?;

        // Model fallback list - Gemma models first (have quota), Gemini as backup
        // Model fallback list - User requested gemini-2.5-flash
        // Model fallback list - Optimized based on live quota (gemini-3-flash has 0 usage)
        let fallback_models = vec![
            "gemini-2.0-flash", // Per User Request
            "gemma-3-27b-it", // High Quota (30 RPM) - Primary
            "gemma-3-12b-it", // High Quota (30 RPM)
            "gemini-2.1-flash-lite",
            "gemini-3-flash",
            "gemini-2.5-flash",
        ];

        let base_url = env::var("MODEL_BASE_URL")
            .unwrap_or_else(|_| "https://generativelanguage.googleapis.com/v1beta/models".to_string());
        let tool_defs = get_tool_definitions();

        let max_iterations = 15;
        let mut final_answer: Option<String> = None;
        let mut step_num = buffer.len() as u32 + 1;

        for iteration in 0..max_iterations {
            info!("   [ReAct] Iteration {}/{}", iteration + 1, max_iterations);

            // Build conversation history
            let mut contents = Vec::new();

            // System prompt
            contents.push(json!({
                "role": "user",
                "parts": [{"text": format!(
                    "{}\n\n\
                    System Instructions: {}",
                    self.config.system_prompt, prompt
                )}]
            }));
            contents.push(json!({
                "role": "model",
                "parts": [{"text": "I will use the available tools to find real information and provide accurate answers."}]
            }));

            // Add all history points
            for point in buffer.get_raw() {
                let role = match point.action.as_str() {
                    "Task" | "User" | "Observation" | "ToolResult" => "user",
                    _ => "model",
                };

                contents.push(json!({
                    "role": role,
                    "parts": [{"text": format!("[{}] {}", point.action, point.thought)}]
                }));
            }

            // Build request with tools
            // Note: Gemma models DON'T support function calling - we need two approaches

            // Try models with fallback
            let mut response_json: Option<serde_json::Value> = None;

            for model in &fallback_models {
                let url = format!("{}/{}:generateContent?key={}", base_url, model, api_key);
                info!("   [ReAct] Trying model: {}", model);

                // Gemma models don't support function calling - use text prompt instead
                let body = if model.contains("gemma") {
                    // Text-only prompt for Gemma - ask it to respond in JSON format
                    let mut gemma_contents = contents.clone();
                    gemma_contents.push(json!({
                        "role": "user",
                        "parts": [{"text": format!(
                            "You have access to these tools: web_search(query), calculate(expression), finish(answer).\n\
                            To use a tool, respond ONLY with a JSON object like:\n\
                            {{\"tool\": \"web_search\", \"args\": {{\"query\": \"NVIDIA stock price\"}}}}\n\
                            When you have the final answer, use:\n\
                            {{\"tool\": \"finish\", \"args\": {{\"answer\": \"Your final answer here\"}}}}\n\
                            Respond with ONLY the JSON object, no other text."
                        )}]
                    }));
                    json!({
                        "contents": gemma_contents,
                        "generationConfig": {
                            "maxOutputTokens": 2048
                        }
                    })
                } else {
                    // Function calling for Gemini models
                    json!({
                        "contents": contents,
                        "tools": [tool_defs],
                        "generationConfig": {
                            "maxOutputTokens": 2048
                        }
                    })
                };

                match self.client.post(&url).json(&body).send().await {
                    Ok(resp) => {
                        if resp.status().is_success() {
                            if let Ok(json) = resp.json::<serde_json::Value>().await {
                                response_json = Some(json);
                                info!("   [ReAct] Got response from {}", model);
                                break;
                            }
                        } else {
                            let error = resp.text().await.unwrap_or_default();
                            if error.contains("429") || error.contains("RESOURCE_EXHAUSTED") {
                                info!("   [ReAct] Quota exhausted for {} - Falling back immediately...", model);
                                continue;
                            } else if error.contains("Function calling is not enabled") {
                                info!("   [ReAct] {} doesn't support function calling, using text mode", model);
                                continue; // Try next model
                            } else {
                                info!(
                                    "   [ReAct] API error for {}: {}",
                                    model,
                                    &error[..error.len().min(200)]
                                );
                                continue; // Try next model
                            }
                        }
                    }
                    Err(e) => info!("   [ReAct] Request failed for {}: {}", model, e),
                }

                if response_json.is_some() {
                    break; // Break model loop
                }
            }

            let response = response_json.ok_or("All models exhausted or failed")?;

            // Parse response - check for function calls
            let candidate = &response["candidates"][0];
            let parts = &candidate["content"]["parts"];

            if let Some(parts_arr) = parts.as_array() {
                for part in parts_arr {
                    // Check for native function call (Gemini models)
                    if let Some(func_call) = part.get("functionCall") {
                        let func_name = func_call["name"].as_str().unwrap_or("");
                        let func_args = &func_call["args"];

                        info!(
                            "   [ReAct] 🔧 Tool Call (native): {}({:?})",
                            func_name, func_args
                        );

                        // Record the tool call in trajectory
                        buffer.add(TrajectoryPoint::new(
                            step_num,
                            "ToolCall".to_string(),
                            format!("{}({})", func_name, func_args),
                        ));
                        step_num += 1;

                        // Execute the tool
                        let result = execute_tool(&self.client, func_name, func_args).await;

                        match &result {
                            ToolResult::Success(output) => {
                                info!(
                                    "   [ReAct] Tool Result: {}...",
                                    &output.chars().take(100).collect::<String>()
                                );

                                // Record observation
                                buffer.add(TrajectoryPoint::new(
                                    step_num,
                                    "Observation".to_string(),
                                    output.clone(),
                                ));
                                step_num += 1;

                                // Check if this was the finish() tool
                                if func_name == "finish" {
                                    final_answer = Some(output.clone());
                                    info!("   [ReAct] 🏁 Task completed with answer!");
                                }
                            }
                            ToolResult::Error(err) => {
                                info!("   [ReAct] Tool Error: {}", err);
                                buffer.add(TrajectoryPoint::new(
                                    step_num,
                                    "ToolError".to_string(),
                                    err.clone(),
                                ));
                                step_num += 1;
                            }
                        }
                    }

                    // Check for text response - might contain JSON tool call from Gemma
                    if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                        if !text.is_empty() {
                            // Try to parse as JSON tool call (for Gemma models)
                            if let Ok(json_call) =
                                serde_json::from_str::<serde_json::Value>(text.trim())
                            {
                                if let Some(tool_name) =
                                    json_call.get("tool").and_then(|t| t.as_str())
                                {
                                    let args = json_call.get("args").cloned().unwrap_or(json!({}));

                                    info!(
                                        "   [ReAct] 🔧 Tool Call (parsed): {}({:?})",
                                        tool_name, args
                                    );

                                    // Record the tool call
                                    buffer.add(TrajectoryPoint::new(
                                        step_num,
                                        "ToolCall".to_string(),
                                        format!("{}({})", tool_name, args),
                                    ));
                                    step_num += 1;

                                    // Execute the tool
                                    let result = execute_tool(&self.client, tool_name, &args).await;

                                    match &result {
                                        ToolResult::Success(output) => {
                                            info!(
                                                "   [ReAct] Tool Result: {}...",
                                                &output.chars().take(100).collect::<String>()
                                            );

                                            buffer.add(TrajectoryPoint::new(
                                                step_num,
                                                "Observation".to_string(),
                                                output.clone(),
                                            ));
                                            step_num += 1;

                                            if tool_name == "finish" {
                                                final_answer = Some(output.clone());
                                                info!("   [ReAct] 🏁 Task completed with answer!");
                                            }
                                        }
                                        ToolResult::Error(err) => {
                                            info!("   [ReAct] Tool Error: {}", err);
                                            buffer.add(TrajectoryPoint::new(
                                                step_num,
                                                "ToolError".to_string(),
                                                err.clone(),
                                            ));
                                            step_num += 1;
                                        }
                                    }
                                    continue; // Don't also log as thought
                                }
                            }

                            // Not JSON - log as thought
                            info!(
                                "   [ReAct] 💭 Model Thought: {}...",
                                &text.chars().take(100).collect::<String>()
                            );
                            buffer.add(TrajectoryPoint::new(
                                step_num,
                                "Thought".to_string(),
                                text.to_string(),
                            ));
                            step_num += 1;
                        }
                    }
                }
            }

            // If we have a final answer, break the loop
            if final_answer.is_some() {
                break;
            }
        }

        // Update context with final result
        if let Some(answer) = &final_answer {
            ctx.final_answer = Some(answer.clone());
            if let Some(last) = buffer.last() {
                ctx.add_trajectory_point(last);
            }
            info!(
                "[AgentGraph] Task Complete - Answer: {}...",
                &answer.chars().take(100).collect::<String>()
            );
        } else {
            info!("[AgentGraph] Task ended without calling finish()");
        }

        // STEP 3: Post-Execution Hooks
        info!("🔄 [AgentGraph] Stage: Post-Step Hooks");
        self.pipeline.run_after(&mut ctx)?;

        info!("[AgentGraph] ReAct Loop Complete.");
        Ok(ctx)
    }
}

/// Python-accessible wrapper for the synchronous `AgentGraph`.
#[pyclass(name = "AgentGraphPy")]
pub struct AgentGraphPy {
    inner: AgentGraph,
}

#[pymethods]
impl AgentGraphPy {
    #[new]
    #[pyo3(signature = (config = None))]
    pub fn new(config: Option<EbbiforgeConfig>) -> Self {
        let inner = match config {
            Some(c) => AgentGraph::with_config(c),
            None => AgentGraph::new(),
        };
        AgentGraphPy { inner }
    }

    /// Registers a new `Agent` persona.
    pub fn register_agent(&mut self, agent: Agent) {
        self.inner.register_agent(agent);
    }

    /// Initiates a task execution cycle with ReAct loop.
    #[pyo3(signature = (task_id, buffer, agent_name = None))]
    pub fn run_task(
        &self,
        task_id: String,
        buffer: &HistoryBuffer,
        agent_name: Option<String>,
    ) -> PyResult<EbbiforgeContext> {
        self.inner.runtime.block_on(async {
            self.inner.run_task(&task_id, buffer, agent_name.as_deref()).await
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
    }

    /// Spawns a background task execution cycle with ReAct loop (Non-Blocking).
    /// Prevents Python threads from stalling during the LLM network requests.
    #[pyo3(signature = (task_id, buffer, agent_name = None))]
    pub fn spawn_task(
        &self,
        task_id: String,
        buffer: &HistoryBuffer,
        agent_name: Option<String>,
    ) -> PyResult<()> {
        let task_name = task_id.clone();
        let agent = agent_name.clone();
        let buf = buffer.clone();
        
        let config = self.inner.config.clone();
        let tasks_map = self.inner.active_tasks.clone();
        let handle_task_id = task_id.clone();
        
        // Spawn onto the existing tokio thread pool as a lightweight Future
        // preventing OS-level Thread Exhaustion (os error 11)
        let handle = self.inner.runtime.spawn(async move {
            let inner_graph = crate::core::runner::AgentGraph::with_config(config);
            let _ = inner_graph.run_task(&task_name, &buf, agent.as_deref()).await;
            
            // Cleanup on finish
            let mut map = tasks_map.lock().unwrap();
            map.remove(&task_name);
        });
        
        {
            let mut map = self.inner.active_tasks.lock().unwrap();
            // Prevent race condition: if future already finished instantly (e.g. auth error), don't resurrect its ID.
            if !handle.is_finished() {
                map.insert(handle_task_id, handle.abort_handle());
            }
        }
        
        Ok(())
    }

    /// Hard kills a task mid-flight, aborting the async tokio future instantly. 
    /// This causes the agent to "die" without warning, preventing further tool calls or LLM requests.
    pub fn kill_task(&self, task_id: String) -> PyResult<bool> {
        let mut map = self.inner.active_tasks.lock().unwrap();
        if let Some(abort_handle) = map.remove(&task_id) {
            abort_handle.abort();
            return Ok(true);
        }
        Ok(false)
    }
    
    /// Returns the number of currently active task futures.
    pub fn active_task_count(&self) -> usize {
        let map = self.inner.active_tasks.lock().unwrap();
        map.len()
    }
}
