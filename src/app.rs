use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::mpsc;

use crate::config::Config;
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
    pub cfg:          Arc<Config>,
}

impl App {
    pub fn new(tx: mpsc::Sender<AppCommand>, project_dir: PathBuf, cfg: Arc<Config>) -> Self {
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
            cfg,
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
                // Auto-stop when Critic score reaches the configured threshold
                if role == Role::Critic {
                    if let (Some(s), Some(threshold)) = (
                        self.history.last().and_then(|t| t.score),
                        self.cfg.stop_at_score,
                    ) {
                        if s >= threshold {
                            self.state = LoopState::Paused;
                            self.status = format!(
                                "Auto-paused: score {s}/10 reached threshold ({threshold}) — Space to resume, e to edit"
                            );
                            return;
                        }
                    }
                }
                // Trim history when max_history cap is reached (keep newest entries)
                if let Some(cap) = self.cfg.max_history {
                    if self.history.len() > cap {
                        let excess = self.history.len() - cap;
                        self.history.drain(..excess);
                    }
                }
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
        self.status = if self.cfg.actor_model == self.cfg.critic_model
            && self.cfg.actor_url() == self.cfg.critic_url()
        {
            "Warming model…".into()
        } else {
            format!("Warming {} + {}…", self.cfg.actor_model, self.cfg.critic_model)
        };
        let tx = self.tx.clone();
        let cfg = Arc::clone(&self.cfg);
        tokio::spawn(crate::ollama::pre_warm(tx, cfg));
    }

    pub fn toggle_pause(&mut self) {
        match &self.state {
            LoopState::Paused => {
                self.status = "Resuming…".into();
                // Resume from the right side of the pair based on what the last turn was.
                // Also fix the iteration counter when resuming after a completed Critic turn
                // (pending_pause skips the normal iteration increment).
                match self.history.last().map(|t| t.role) {
                    Some(Role::Actor) => {
                        self.state = LoopState::CriticThink;
                        self.spawn_critic();
                    }
                    _ => {
                        self.iteration += 1;
                        self.state = LoopState::ActorThink;
                        self.spawn_actor();
                    }
                }
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
        let ts = chrono::Local::now().format("%Y-%m-%dT%H%M%S%.3f").to_string();
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
            "prompt":        self.prompt,
            "actor_model":   self.cfg.actor_model,
            "critic_model":  self.cfg.critic_model,
            "saved_at":      ts,
            "turns":         turns,
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

    pub fn export_markdown(&mut self) {
        if self.history.is_empty() { return; }
        let ts = chrono::Local::now().format("%Y-%m-%dT%H%M%S%.3f").to_string();
        let fname = format!("duel-session-{ts}.md");
        let path = self.project_dir.join(&fname);

        let mut out = String::new();
        out.push_str("---\n");
        out.push_str(&format!("prompt: {:?}\n", self.prompt));
        out.push_str(&format!("actor_model: {:?}\n", self.cfg.actor_model));
        out.push_str(&format!("critic_model: {:?}\n", self.cfg.critic_model));
        out.push_str(&format!("timestamp: {:?}\n", ts));
        out.push_str("---\n\n");

        for turn in &self.history {
            let heading = match turn.role {
                Role::Actor => format!("## Iteration {} — ACTOR", turn.iteration),
                Role::Critic => {
                    let score_str = turn.score
                        .map(|s| format!(" (Score: {s}/10)"))
                        .unwrap_or_default();
                    format!("## Iteration {} — CRITIC{score_str}", turn.iteration)
                }
            };
            out.push_str(&heading);
            out.push_str("\n\n");
            out.push_str(&turn.content);
            out.push_str("\n\n");
        }

        match std::fs::write(&path, out) {
            Ok(_) => {
                self.session_path = Some(path);
                self.status = format!("Exported: {fname}");
            }
            Err(e) => {
                self.status = format!("Export failed: {e}");
            }
        }
    }

    pub fn export_html(&mut self) {
        if self.history.is_empty() { return; }
        let ts = chrono::Local::now().format("%Y-%m-%dT%H%M%S%.3f").to_string();
        let fname = format!("duel-session-{ts}.html");
        let path = self.project_dir.join(&fname);

        fn esc(s: &str) -> String {
            s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
        }

        let mut turns_html = String::new();
        for turn in &self.history {
            let (cls, label) = match turn.role {
                Role::Actor  => ("actor",  format!("Iteration {} — ACTOR", turn.iteration)),
                Role::Critic => {
                    let sc = turn.score.map(|s| format!(" <span class=\"score\">(Score: {s}/10)</span>")).unwrap_or_default();
                    ("critic", format!("Iteration {} — CRITIC{}", turn.iteration, sc))
                }
            };
            turns_html.push_str(&format!(
                "<div class=\"turn {cls}\">\n\
                 <div class=\"turn-header\">{label}</div>\n\
                 <pre>{}</pre>\n\
                 </div>\n",
                esc(&turn.content)
            ));
        }

        let html = format!(r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>duel — {}</title>
<style>
  body{{background:#1e1e2e;color:#cdd6f4;font-family:monospace;max-width:920px;margin:0 auto;padding:2rem;line-height:1.5}}
  h1{{color:#cba6f7;border-bottom:1px solid #45475a;padding-bottom:.5rem;margin-bottom:.5rem}}
  .meta{{color:#6c7086;font-size:.85em;margin-bottom:2rem}}
  .turn{{margin:1.5rem 0}}
  .turn-header{{font-weight:bold;margin-bottom:.4rem}}
  .actor .turn-header{{color:#89b4fa}}
  .critic .turn-header{{color:#f38ba8}}
  .score{{color:#a6e3a1}}
  pre{{white-space:pre-wrap;word-break:break-word;background:#313244;padding:1rem;border-radius:.4rem;margin:0}}
</style>
</head>
<body>
<h1>duel session</h1>
<div class="meta">
  <div>Prompt: {}</div>
  <div>Actor: {} &nbsp;|&nbsp; Critic: {}</div>
  <div>Exported: {}</div>
</div>
{}
</body>
</html>
"#,
            esc(&self.prompt),
            esc(&self.prompt),
            esc(&self.cfg.actor_model),
            esc(&self.cfg.critic_model),
            ts,
            turns_html,
        );

        match std::fs::write(&path, html) {
            Ok(_)  => { self.session_path = Some(path); self.status = format!("Exported: {fname}"); }
            Err(e) => { self.status = format!("HTML export failed: {e}"); }
        }
    }

    /// Load a saved JSON session. Returns an error string on failure.
    pub fn resume_session(&mut self, path: &std::path::Path) -> anyhow::Result<()> {
        let data = std::fs::read_to_string(path)?;
        let doc: serde_json::Value = serde_json::from_str(&data)?;

        let prompt = doc["prompt"].as_str().unwrap_or("").to_string();
        if prompt.is_empty() {
            return Err(anyhow::anyhow!("session file has no prompt"));
        }

        let turns_val = doc["turns"].as_array()
            .ok_or_else(|| anyhow::anyhow!("session file has no turns array"))?;

        let mut history = Vec::with_capacity(turns_val.len());
        for t in turns_val {
            let role = match t["role"].as_str().unwrap_or("") {
                "actor"  => Role::Actor,
                "critic" => Role::Critic,
                other    => return Err(anyhow::anyhow!("unknown role: {other}")),
            };
            history.push(Turn {
                role,
                content:     t["content"].as_str().unwrap_or("").to_string(),
                iteration:   t["iteration"].as_u64().unwrap_or(1) as usize,
                eval_count:  t["eval_count"].as_u64().unwrap_or(0) as u32,
                duration_ms: t["duration_ms"].as_u64().unwrap_or(0),
                score:       t["score"].as_u64().map(|n| n as u8),
            });
        }

        let last_iter = history.last().map(|t| t.iteration).unwrap_or(1);
        self.history      = history;
        self.prompt       = prompt;
        self.iteration    = last_iter;
        self.scroll       = 0;
        self.session_path = Some(path.to_path_buf());
        self.state        = LoopState::Paused;
        let fname = path.file_name().unwrap_or_default().to_string_lossy();
        self.status = format!("Resumed from {fname} — Space to continue, e to edit");
        Ok(())
    }

    pub fn scroll_up(&mut self)   { self.scroll = self.scroll.saturating_sub(1); }
    pub fn scroll_down(&mut self) { self.scroll = self.scroll.saturating_add(1); }

    fn spawn_actor(&self) {
        self.warn_context_window();
        let tx      = self.tx.clone();
        let prompt  = self.prompt.clone();
        let history = self.history.clone();
        let iter    = self.iteration;
        let cfg     = Arc::clone(&self.cfg);
        tokio::spawn(crate::ollama::run_actor(tx, prompt, history, iter, cfg));
    }

    fn spawn_critic(&self) {
        self.warn_context_window();
        let tx      = self.tx.clone();
        let history = self.history.clone();
        let iter    = self.iteration;
        let cfg     = Arc::clone(&self.cfg);
        tokio::spawn(crate::ollama::run_critic(tx, history, iter, cfg));
    }

    fn warn_context_window(&self) {
        // Estimate context size from recent turns' content length (~4 chars/token).
        const WARN_TOKENS: usize = 28_000;
        let estimated: usize = self.history.iter().rev()
            .take(self.cfg.context_turns)
            .map(|t| t.content.len() / 4)
            .sum();
        if estimated > WARN_TOKENS {
            let tx = self.tx.clone();
            let msg = format!(
                "⚠ Context ~{estimated}tok (>{WARN_TOKENS}): consider reducing --context-turns"
            );
            tokio::spawn(async move { let _ = tx.send(crate::app::AppCommand::StatusMsg(msg)).await; });
        }
    }
}

fn extract_score(text: &str) -> Option<u8> {
    let lower = text.to_lowercase();
    // Strip markdown decoration before scanning
    let clean: String = lower.chars()
        .filter(|&c| c != '*' && c != '_' && c != '`')
        .collect();

    // Strategy 1: byte-level scan for "N/10" or "N / 10" (handles spaces around slash)
    let bytes = clean.as_bytes();
    let len = bytes.len();
    let mut i = 0;
    while i < len {
        if bytes[i].is_ascii_digit() {
            let ns = i;
            while i < len && bytes[i].is_ascii_digit() { i += 1; }
            let num_str = std::str::from_utf8(&bytes[ns..i]).unwrap_or("");
            let mut j = i;
            while j < len && bytes[j] == b' ' { j += 1; }
            if j < len && bytes[j] == b'/' {
                j += 1;
                while j < len && bytes[j] == b' ' { j += 1; }
                if j + 1 < len && bytes[j] == b'1' && bytes[j + 1] == b'0'
                    && (j + 2 >= len || !bytes[j + 2].is_ascii_digit())
                {
                    if let Ok(n) = num_str.parse::<u8>() {
                        if n <= 10 { return Some(n); }
                    }
                }
            }
        } else {
            i += 1;
        }
    }

    // Strategy 2: "score:" or "score " prefix followed by digits
    for prefix in &["score:", "score "] {
        if let Some(pos) = clean.find(prefix) {
            let after = clean[pos + prefix.len()..].trim_start();
            let num: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
            if let Ok(n) = num.parse::<u8>() {
                if n <= 10 { return Some(n); }
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::extract_score;

    #[test] fn score_colon_slash()      { assert_eq!(extract_score("Score: 8/10"), Some(8)); }
    #[test] fn score_bare_slash()       { assert_eq!(extract_score("8/10"), Some(8)); }
    #[test] fn score_spaces_slash()     { assert_eq!(extract_score("8 / 10"), Some(8)); }
    #[test] fn score_markdown_bold()    { assert_eq!(extract_score("**Score: 8/10**"), Some(8)); }
    #[test] fn score_bare_number()      { assert_eq!(extract_score("score 7"), Some(7)); }
    #[test] fn score_colon_spaces()     { assert_eq!(extract_score("score: 7 / 10"), Some(7)); }
    #[test] fn score_zero()             { assert_eq!(extract_score("Score: 0/10"), Some(0)); }
    #[test] fn score_ten()              { assert_eq!(extract_score("Score: 10/10"), Some(10)); }
    #[test] fn score_eleven_rejected()  { assert_eq!(extract_score("11/10"), None); }
    #[test] fn score_out_of_range()     { assert_eq!(extract_score("Score: 11"), None); }
    #[test] fn score_empty()            { assert_eq!(extract_score(""), None); }
    #[test] fn score_no_score()         { assert_eq!(extract_score("Great response!"), None); }
    #[test] fn score_not_nine_over_100(){ assert_eq!(extract_score("9/100"), None); }
    #[test] fn score_markdown_italic()  { assert_eq!(extract_score("_Score: 6/10_"), Some(6)); }
    #[test] fn score_mixed_case()       { assert_eq!(extract_score("SCORE: 5/10"), Some(5)); }
    #[test] fn score_embedded_in_text() {
        assert_eq!(extract_score("I give this a solid 7/10 overall."), Some(7));
    }
    #[test] fn score_newlines() {
        assert_eq!(extract_score("Good work.\nScore: 9/10\nKeep it up."), Some(9));
    }
}
