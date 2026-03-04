use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::error::BenchError;
use crate::suite::{BenchScore, BenchSuite, BenchTask, ConversationTurn, TaskSubmission};

// ---------------------------------------------------------------------------
// Scenario format (matches ironclaw's BenchScenario JSON)
// ---------------------------------------------------------------------------

/// A multi-turn trajectory scenario loaded from a JSON file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryScenario {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub setup: ScenarioSetup,
    #[serde(default = "default_timeout_secs")]
    pub timeout_secs: u64,
    #[serde(default = "default_max_tool_iterations")]
    pub max_tool_iterations: usize,
    pub turns: Vec<ScenarioTurn>,
}

fn default_timeout_secs() -> u64 {
    120
}

fn default_max_tool_iterations() -> usize {
    10
}

/// Setup configuration for a scenario.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScenarioSetup {
    /// Tools that should be available (restrict registry to only these).
    #[serde(default)]
    pub tools: Vec<String>,
    /// Identity files to inject into workspace.
    #[serde(default)]
    pub identity: std::collections::HashMap<String, String>,
    /// Workspace documents to seed.
    #[serde(default)]
    pub workspace: WorkspaceSetup,
}

/// Workspace pre-seeding configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WorkspaceSetup {
    #[serde(default)]
    pub documents: std::collections::HashMap<String, String>,
}

/// A single turn in a trajectory scenario.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioTurn {
    pub user: String,
    #[serde(default)]
    pub assertions: TurnAssertions,
}

/// Multi-criterion assertions for a single turn.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TurnAssertions {
    #[serde(default)]
    pub response_contains: Vec<String>,
    #[serde(default)]
    pub response_not_contains: Vec<String>,
    #[serde(default)]
    pub tools_called: Vec<String>,
    #[serde(default)]
    pub tools_not_called: Vec<String>,
    #[serde(default)]
    pub response_matches: Option<String>,
    #[serde(default)]
    pub max_tool_calls: Option<usize>,
}

impl TurnAssertions {
    /// Evaluate assertions against a submission, returning (score, failures).
    pub fn evaluate(&self, submission: &TaskSubmission) -> (f64, Vec<String>) {
        let mut passed: usize = 0;
        let mut total: usize = 0;
        let mut failures: Vec<String> = Vec::new();

        let response_lower = submission.response.to_lowercase();

        // response_contains
        for needle in &self.response_contains {
            total += 1;
            if response_lower.contains(&needle.to_lowercase()) {
                passed += 1;
            } else {
                failures.push(format!("response_contains: missing \"{needle}\""));
            }
        }

        // response_not_contains
        for needle in &self.response_not_contains {
            total += 1;
            if response_lower.contains(&needle.to_lowercase()) {
                failures.push(format!("response_not_contains: found \"{needle}\""));
            } else {
                passed += 1;
            }
        }

        let tool_set: HashSet<&str> = submission.tool_calls.iter().map(|s| s.as_str()).collect();

        // tools_called
        for tool in &self.tools_called {
            total += 1;
            if tool_set.contains(tool.as_str()) {
                passed += 1;
            } else {
                failures.push(format!("tools_called: \"{tool}\" not called"));
            }
        }

        // tools_not_called
        for tool in &self.tools_not_called {
            total += 1;
            if tool_set.contains(tool.as_str()) {
                failures.push(format!("tools_not_called: \"{tool}\" was called"));
            } else {
                passed += 1;
            }
        }

        // response_matches
        if let Some(ref pattern) = self.response_matches {
            total += 1;
            match Regex::new(pattern) {
                Ok(re) => {
                    if re.is_match(&submission.response) {
                        passed += 1;
                    } else {
                        failures.push(format!("response_matches: /{pattern}/ did not match"));
                    }
                }
                Err(e) => {
                    failures.push(format!("response_matches: bad regex: {e}"));
                }
            }
        }

        // max_tool_calls
        if let Some(max) = self.max_tool_calls {
            total += 1;
            let call_count = submission.tool_calls.len();
            if call_count <= max {
                passed += 1;
            } else {
                failures.push(format!(
                    "max_tool_calls: expected <= {max}, got {call_count}"
                ));
            }
        }

        if total == 0 {
            return (1.0, failures);
        }

        let score = passed as f64 / total as f64;
        (score, failures)
    }
}

// ---------------------------------------------------------------------------
// TrajectorySuite
// ---------------------------------------------------------------------------

/// Trajectory benchmark suite: multi-turn scenario replay with per-turn assertions.
///
/// Loads scenarios from a directory of JSON files (recursive). Each file describes
/// a multi-turn conversation with setup requirements and per-turn assertions.
pub struct TrajectorySuite {
    dataset_path: PathBuf,
}

impl TrajectorySuite {
    pub fn new(dataset_path: impl Into<PathBuf>) -> Self {
        Self {
            dataset_path: dataset_path.into(),
        }
    }

    /// Recursively find all .json files under a directory.
    fn find_json_files(dir: &Path) -> Result<Vec<PathBuf>, BenchError> {
        let mut files = Vec::new();
        if dir.is_file() {
            files.push(dir.to_path_buf());
            return Ok(files);
        }
        if !dir.is_dir() {
            return Err(BenchError::Config(format!(
                "dataset path does not exist: {}",
                dir.display()
            )));
        }
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                files.extend(Self::find_json_files(&path)?);
            } else if path.extension().is_some_and(|ext| ext == "json") {
                files.push(path);
            }
        }
        files.sort();
        Ok(files)
    }

    /// Load a single scenario from a JSON file.
    fn load_scenario(path: &Path) -> Result<TrajectoryScenario, BenchError> {
        let content = std::fs::read_to_string(path)?;
        let scenario: TrajectoryScenario = serde_json::from_str(&content).map_err(|e| {
            BenchError::Config(format!("failed to parse {}: {}", path.display(), e))
        })?;
        Ok(scenario)
    }
}

#[async_trait]
impl BenchSuite for TrajectorySuite {
    fn name(&self) -> &str {
        "Trajectory Scenarios"
    }

    fn id(&self) -> &str {
        "trajectory"
    }

    async fn load_tasks(&self) -> Result<Vec<BenchTask>, BenchError> {
        let files = Self::find_json_files(&self.dataset_path)?;
        let mut tasks = Vec::new();

        for path in files {
            let scenario = Self::load_scenario(&path)?;

            // Store the full scenario in metadata for scoring and multi-turn support.
            let metadata = serde_json::to_value(&scenario).map_err(|e| {
                BenchError::Config(format!(
                    "failed to serialize scenario {}: {}",
                    scenario.name, e
                ))
            })?;

            let first_prompt = scenario
                .turns
                .first()
                .map(|t| t.user.clone())
                .unwrap_or_default();

            tasks.push(BenchTask {
                id: scenario.name.clone(),
                prompt: first_prompt,
                context: if scenario.description.is_empty() {
                    None
                } else {
                    Some(scenario.description.clone())
                },
                resources: vec![],
                tags: scenario.tags.clone(),
                expected_turns: Some(scenario.turns.len()),
                timeout: Some(Duration::from_secs(scenario.timeout_secs)),
                metadata,
            });
        }

        Ok(tasks)
    }

    async fn score(
        &self,
        task: &BenchTask,
        submission: &TaskSubmission,
    ) -> Result<BenchScore, BenchError> {
        let scenario: TrajectoryScenario =
            serde_json::from_value(task.metadata.clone()).map_err(|e| BenchError::Scoring {
                task_id: task.id.clone(),
                reason: format!("failed to deserialize scenario: {e}"),
            })?;

        // For single-turn evaluation, score the final submission against all turn
        // assertions. For multi-turn, the runner should call next_user_message()
        // and accumulate tool_calls across turns.
        //
        // Since the runner feeds one turn at a time via next_user_message(), the
        // final submission represents the last turn. We score based on what the
        // runner accumulated in submission.tool_calls and conversation.
        //
        // Aggregate approach: score each turn's assertions against the
        // accumulated conversation, then average.
        let num_turns = scenario.turns.len();

        if num_turns == 0 {
            return Ok(BenchScore::pass());
        }

        // For single-turn scenarios, just evaluate against the submission directly.
        if num_turns == 1 {
            let (score, failures) = scenario.turns[0].assertions.evaluate(submission);
            return if score >= 1.0 {
                Ok(BenchScore::pass())
            } else if score <= 0.0 {
                Ok(BenchScore::fail(failures.join("; ")))
            } else {
                Ok(BenchScore::partial(score, failures.join("; ")))
            };
        }

        // Multi-turn: evaluate the final turn's assertions against the submission.
        // Earlier turns should have been validated by the runner during execution.
        // The submission represents the full conversation outcome.
        let last_turn = &scenario.turns[num_turns - 1];
        let (score, failures) = last_turn.assertions.evaluate(submission);

        if score >= 1.0 {
            Ok(BenchScore::pass())
        } else if score <= 0.0 {
            Ok(BenchScore::fail(failures.join("; ")))
        } else {
            Ok(BenchScore::partial(score, failures.join("; ")))
        }
    }

    async fn next_user_message(
        &self,
        task: &BenchTask,
        conversation: &[ConversationTurn],
    ) -> Result<Option<String>, BenchError> {
        let scenario: TrajectoryScenario =
            serde_json::from_value(task.metadata.clone()).map_err(|e| BenchError::Scoring {
                task_id: task.id.clone(),
                reason: format!("failed to deserialize scenario: {e}"),
            })?;

        // Count how many user messages have been sent so far.
        let user_turns_sent = conversation
            .iter()
            .filter(|t| matches!(t.role, crate::suite::TurnRole::User))
            .count();

        // The first turn was already sent as the initial prompt.
        // Return the next turn if available.
        if user_turns_sent < scenario.turns.len() {
            Ok(Some(scenario.turns[user_turns_sent].user.clone()))
        } else {
            Ok(None)
        }
    }

    fn additional_tools(&self) -> Vec<Arc<dyn ironclaw::tools::Tool>> {
        // Trajectory scenarios specify tools in setup.tools per-scenario.
        // The runner should use retain_only() based on task metadata.
        // We provide the standard tool set here.
        vec![
            Arc::new(ironclaw::tools::builtin::ShellTool::new()),
            Arc::new(ironclaw::tools::builtin::ReadFileTool::new()),
            Arc::new(ironclaw::tools::builtin::WriteFileTool::new()),
            Arc::new(ironclaw::tools::builtin::ListDirTool::new()),
            Arc::new(ironclaw::tools::builtin::ApplyPatchTool::new()),
        ]
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn make_submission(
        response: &str,
        tool_calls: Vec<&str>,
        error: Option<&str>,
    ) -> TaskSubmission {
        TaskSubmission {
            response: response.to_string(),
            conversation: vec![],
            tool_calls: tool_calls.into_iter().map(|s| s.to_string()).collect(),
            error: error.map(|s| s.to_string()),
        }
    }

    // -- TurnAssertions unit tests --

    #[test]
    fn test_assertions_all_pass() {
        let assertions = TurnAssertions {
            response_contains: vec!["hello".to_string()],
            tools_called: vec!["echo".to_string()],
            max_tool_calls: Some(3),
            ..Default::default()
        };
        let sub = make_submission("Hello world!", vec!["echo"], None);
        let (score, failures) = assertions.evaluate(&sub);
        assert_eq!(score, 1.0);
        assert!(failures.is_empty());
    }

    #[test]
    fn test_assertions_partial_score() {
        let assertions = TurnAssertions {
            response_contains: vec!["alpha".to_string(), "beta".to_string()],
            ..Default::default()
        };
        let sub = make_submission("alpha is here", vec![], None);
        let (score, failures) = assertions.evaluate(&sub);
        assert_eq!(score, 0.5);
        assert_eq!(failures.len(), 1);
    }

    #[test]
    fn test_assertions_tools_not_called() {
        let assertions = TurnAssertions {
            tools_not_called: vec!["shell".to_string()],
            ..Default::default()
        };
        let sub = make_submission("ok", vec!["shell"], None);
        let (score, _) = assertions.evaluate(&sub);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_assertions_max_tool_calls_exceeded() {
        let assertions = TurnAssertions {
            max_tool_calls: Some(1),
            ..Default::default()
        };
        let sub = make_submission("ok", vec!["a", "b", "c"], None);
        let (score, failures) = assertions.evaluate(&sub);
        assert_eq!(score, 0.0);
        assert!(failures[0].contains("max_tool_calls"));
    }

    #[test]
    fn test_assertions_response_matches_regex() {
        let assertions = TurnAssertions {
            response_matches: Some(r"\d{4}".to_string()),
            ..Default::default()
        };
        let sub = make_submission("The year is 2026", vec![], None);
        let (score, _) = assertions.evaluate(&sub);
        assert_eq!(score, 1.0);
    }

    #[test]
    fn test_assertions_empty() {
        let assertions = TurnAssertions::default();
        let sub = make_submission("anything", vec!["whatever"], None);
        let (score, _) = assertions.evaluate(&sub);
        assert_eq!(score, 1.0);
    }

    // -- TrajectorySuite tests --

    #[tokio::test]
    async fn test_load_tasks_from_directory() {
        let dir = tempfile::tempdir().unwrap();
        let subdir = dir.path().join("tools");
        std::fs::create_dir_all(&subdir).unwrap();

        // Write a scenario file.
        let scenario_path = subdir.join("pick-echo.json");
        let mut file = std::fs::File::create(&scenario_path).unwrap();
        write!(
            file,
            r#"{{
                "name": "pick-echo-tool",
                "description": "Agent should use echo tool",
                "tags": ["tools"],
                "setup": {{ "tools": ["echo", "shell"] }},
                "timeout_secs": 30,
                "turns": [
                    {{
                        "user": "Use the echo tool to say hello",
                        "assertions": {{
                            "tools_called": ["echo"],
                            "response_contains": ["hello"]
                        }}
                    }}
                ]
            }}"#
        )
        .unwrap();

        let suite = TrajectorySuite::new(dir.path());
        let tasks = suite.load_tasks().await.unwrap();
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].id, "pick-echo-tool");
        assert_eq!(tasks[0].prompt, "Use the echo tool to say hello");
        assert!(tasks[0].tags.contains(&"tools".to_string()));
        assert_eq!(tasks[0].expected_turns, Some(1));
        assert_eq!(tasks[0].timeout, Some(Duration::from_secs(30)));
    }

    #[tokio::test]
    async fn test_load_tasks_multi_turn() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("multi.json");
        let mut file = std::fs::File::create(&path).unwrap();
        write!(
            file,
            r#"{{
                "name": "save-and-recall",
                "tags": ["memory"],
                "turns": [
                    {{
                        "user": "Save note: Project Alpha launches March 15",
                        "assertions": {{ "tools_called": ["memory_write"] }}
                    }},
                    {{
                        "user": "When does Project Alpha launch?",
                        "assertions": {{
                            "tools_called": ["memory_search"],
                            "response_contains": ["March 15"]
                        }}
                    }}
                ]
            }}"#
        )
        .unwrap();

        let suite = TrajectorySuite::new(&path);
        let tasks = suite.load_tasks().await.unwrap();
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].expected_turns, Some(2));
        assert_eq!(
            tasks[0].prompt,
            "Save note: Project Alpha launches March 15"
        );
    }

    #[tokio::test]
    async fn test_score_single_turn_pass() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.json");
        let mut file = std::fs::File::create(&path).unwrap();
        write!(
            file,
            r#"{{
                "name": "echo-test",
                "turns": [{{
                    "user": "Echo hello",
                    "assertions": {{
                        "tools_called": ["echo"],
                        "response_contains": ["hello"]
                    }}
                }}]
            }}"#
        )
        .unwrap();

        let suite = TrajectorySuite::new(&path);
        let tasks = suite.load_tasks().await.unwrap();

        let sub = make_submission("hello world", vec!["echo"], None);
        let score = suite.score(&tasks[0], &sub).await.unwrap();
        assert_eq!(score.value, 1.0);
        assert_eq!(score.label, "pass");
    }

    #[tokio::test]
    async fn test_score_single_turn_fail() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.json");
        let mut file = std::fs::File::create(&path).unwrap();
        write!(
            file,
            r#"{{
                "name": "echo-test",
                "turns": [{{
                    "user": "Echo hello",
                    "assertions": {{
                        "tools_called": ["echo"],
                        "response_contains": ["hello"]
                    }}
                }}]
            }}"#
        )
        .unwrap();

        let suite = TrajectorySuite::new(&path);
        let tasks = suite.load_tasks().await.unwrap();

        let sub = make_submission("goodbye", vec!["shell"], None);
        let score = suite.score(&tasks[0], &sub).await.unwrap();
        assert_eq!(score.value, 0.0);
        assert_eq!(score.label, "fail");
    }

    #[tokio::test]
    async fn test_next_user_message() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("multi.json");
        let mut file = std::fs::File::create(&path).unwrap();
        write!(
            file,
            r#"{{
                "name": "two-turn",
                "turns": [
                    {{ "user": "First message", "assertions": {{}} }},
                    {{ "user": "Second message", "assertions": {{}} }}
                ]
            }}"#
        )
        .unwrap();

        let suite = TrajectorySuite::new(&path);
        let tasks = suite.load_tasks().await.unwrap();

        // No conversation yet -> should return first turn.
        let msg = suite.next_user_message(&tasks[0], &[]).await.unwrap();
        assert_eq!(msg.as_deref(), Some("First message"));

        // After 1 user turn -> should return second turn.
        let conv = vec![ConversationTurn {
            role: crate::suite::TurnRole::User,
            content: "First message".to_string(),
        }];
        let msg = suite.next_user_message(&tasks[0], &conv).await.unwrap();
        assert_eq!(msg.as_deref(), Some("Second message"));

        // After 2 user turns -> should return None.
        let conv = vec![
            ConversationTurn {
                role: crate::suite::TurnRole::User,
                content: "First message".to_string(),
            },
            ConversationTurn {
                role: crate::suite::TurnRole::User,
                content: "Second message".to_string(),
            },
        ];
        let msg = suite.next_user_message(&tasks[0], &conv).await.unwrap();
        assert!(msg.is_none());
    }

    #[tokio::test]
    async fn test_nonexistent_path_errors() {
        let suite = TrajectorySuite::new("/nonexistent/path");
        let err = suite.load_tasks().await.unwrap_err();
        assert!(err.to_string().contains("does not exist"));
    }
}
