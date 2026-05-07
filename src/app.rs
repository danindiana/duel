use std::path::PathBuf;
use tokio::sync::mpsc;

use crate::gpu::GpuInfo;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Role {
    Actor,
    Critic,
}

impl Role {
    pub fn label(self) -> &'static str {
        match self { Role::Actor => "ACTOR", Role::Critic => "CRITIC" }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum LoopState {
    Idle,
    Prewarm,
    ActorThink,
    CriticThink,
    Paused,
    Error(String),
}

#[derive(Debug, Clone)]
pub struct Turn {
    pub role:        Role,
    pub content:     String,
    pub iteration:   usize,
    pub eval_count:  u32,
    pub duration_ms: u64,
    pub score:       Option<u8>,
}

#[allow(dead_code)]
pub enum AppCommand {
    Token       { role: Role, token: String },
    TurnDone    { role: Role, eval_count: u32, duration_ms: u64 },
    ModelReady,
    LoopError(String),
    GpuStatus(Vec<GpuInfo>),
    StatusMsg(String),
}

pub struct App {
    pub prompt:       String,
    pub history:      Vec<Turn>,
    pub actor_buf:    String,
    pub critic_buf:   String,
    pub state:        LoopState,
    pub iteration:    usize,
    pub input_buf:    String,
    pub scroll:       u16,
    pub theme_idx:    usize,
    pub gpu:          Vec<GpuInfo>,
    pub status:       String,
    pub tx:           mpsc::Sender<AppCommand>,
    pub show_help:    bool,
    pub session_path: Option<PathBuf>,
    pub project_dir:  PathBuf,
    pub pending_pause: bool,
}

impl App {
    pub fn new(tx: mpsc::Sender<AppCommand>, project_dir: PathBuf) -> Self {
        Self {
            prompt:       String::new(),
            history:      Vec::new(),
            actor_buf:    String::new(),
            critic_buf:   String::new(),
            state:        LoopState::Idle,
            iteration:    0,
            input_buf:    String::new(),
            scroll:       0,
            theme_idx:    0,
            gpu:          Vec::new(),
            status:       String::new(),
            tx,
            show_help:    false,
            session_path: None,
            project_dir,
            pending_pause: false,
        }
    }

    pub fn handle_command(&mut self, cmd: AppCommand) {
        match cmd {
            AppCommand::Token { role, token } => {
                match role {
                    Role::Actor  => self.actor_buf.push_str(&token),
                    Role::Critic => self.critic_buf.push_str(&token),
                }
            }
            AppCommand::TurnDone { role, eval_count, duration_ms } => {
                let content = match role {
                    Role::Actor  => std::mem::take(&mut self.actor_buf),
                    Role::Critic => std::mem::take(&mut self.critic_buf),
                };
                let score = if role == Role::Critic { extract_score(&content) } else { None };
                self.history.push(Turn {
                    role,
                    content,
                    iteration: self.iteration,
                    eval_count,
                    duration_ms,
                    score,
                });
                if self.pending_pause {
                    self.pending_pause = false;
                    self.state = LoopState::Paused;
                    self.status = "Paused — Space to resume, e to edit prompt".into();
                    return;
                }
                match role {
                    Role::Actor => {
                        self.state = LoopState::CriticThink;
                        self.spawn_critic();
                    }
                    Role::Critic => {
                        self.iteration += 1;
                        self.state = LoopState::ActorThink;
                        self.spawn_actor();
                    }
                }
            }
            AppCommand::ModelReady => {
                self.status = "Model ready — starting loop".into();
                self.iteration = 1;
                self.state = LoopState::ActorThink;
                self.spawn_actor();
            }
            AppCommand::LoopError(e) => {
                self.state = LoopState::Error(e.clone());
                self.status = format!("Error: {e}");
            }
            AppCommand::GpuStatus(gpus) => {
                self.gpu = gpus;
            }
            AppCommand::StatusMsg(msg) => {
                self.status = msg;
            }
        }
    }

    pub fn start_loop(&mut self) {
        let prompt = self.input_buf.trim().to_string();
        if prompt.is_empty() { return; }
        self.prompt = prompt;
        self.input_buf.clear();
        self.history.clear();
        self.actor_buf.clear();
        self.critic_buf.clear();
        self.iteration = 0;
        self.scroll = 0;
        self.pending_pause = false;
        self.session_path = None;
        self.state = LoopState::Prewarm;
        self.status = "Warming model…".into();
        let tx = self.tx.clone();
        tokio::spawn(crate::ollama::pre_warm(tx));
    }

    pub fn toggle_pause(&mut self) {
        match &self.state {
            LoopState::Paused => {
                self.state = LoopState::ActorThink;
                self.status = "Resuming…".into();
                self.spawn_actor();
            }
            LoopState::ActorThink | LoopState::CriticThink => {
                self.pending_pause = true;
                self.status = "Pausing after current turn…".into();
            }
            _ => {}
        }
    }

    pub fn edit_prompt(&mut self) {
        if matches!(self.state, LoopState::Paused | LoopState::Error(_) | LoopState::Idle) {
            self.input_buf = self.prompt.clone();
            self.state = LoopState::Idle;
            self.status = String::new();
        }
    }

    pub fn save_session(&mut self) {
        if self.history.is_empty() { return; }
        let ts = chrono::Local::now().format("%Y-%m-%dT%H%M%S").to_string();
        let fname = format!("duel-session-{ts}.json");
        let path = self.project_dir.join(&fname);
        let turns: Vec<serde_json::Value> = self.history.iter().map(|t| {
            serde_json::json!({
                "iteration":   t.iteration,
                "role":        format!("{:?}", t.role).to_lowercase(),
                "content":     t.content,
                "eval_count":  t.eval_count,
                "duration_ms": t.duration_ms,
                "score":       t.score,
            })
        }).collect();
        let doc = serde_json::json!({
            "prompt":   self.prompt,
            "model":    "qwen3.5:4b",
            "saved_at": ts,
            "turns":    turns,
        });
        match serde_json::to_string_pretty(&doc)
            .map_err(anyhow::Error::from)
            .and_then(|s| std::fs::write(&path, s).map_err(anyhow::Error::from))
        {
            Ok(_) => {
                self.session_path = Some(path.clone());
                self.status = format!("Saved: {fname}");
            }
            Err(e) => {
                self.status = format!("Save failed: {e}");
            }
        }
    }

    pub fn scroll_up(&mut self)   { self.scroll = self.scroll.saturating_sub(1); }
    pub fn scroll_down(&mut self) { self.scroll = self.scroll.saturating_add(1); }

    fn spawn_actor(&self) {
        let tx      = self.tx.clone();
        let prompt  = self.prompt.clone();
        let history = self.history.clone();
        let iter    = self.iteration;
        tokio::spawn(crate::ollama::run_actor(tx, prompt, history, iter));
    }

    fn spawn_critic(&self) {
        let tx      = self.tx.clone();
        let history = self.history.clone();
        let iter    = self.iteration;
        tokio::spawn(crate::ollama::run_critic(tx, history, iter));
    }
}

fn extract_score(text: &str) -> Option<u8> {
    // Match "Score: 8/10", "8/10", "score: 8", etc.
    let lower = text.to_lowercase();
    // Try "X/10" pattern
    for cap in lower.split_whitespace() {
        if let Some(slash) = cap.find("/10") {
            let num_str: String = cap[..slash].chars().filter(|c| c.is_ascii_digit()).collect();
            if let Ok(n) = num_str.parse::<u8>() {
                if n <= 10 { return Some(n); }
            }
        }
    }
    // Try "score: X"
    if let Some(pos) = lower.find("score:") {
        let after = lower[pos + 6..].trim_start();
        let num_str: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
        if let Ok(n) = num_str.parse::<u8>() {
            if n <= 10 { return Some(n); }
        }
    }
    None
}
