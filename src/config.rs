use serde::Deserialize;

// ── System prompts ────────────────────────────────────────────────────────────

pub const DEFAULT_ACTOR: &str = "\
You are the Actor. Your job is to generate the best possible response to the user's task. \
Be thorough, precise, and creative. In later iterations, incorporate the Critic's feedback \
to improve your output.";

const DEFAULT_CRITIC: &str = "\
You are the Critic. Evaluate the Actor's latest response. Identify specific weaknesses, \
errors, or missed opportunities. Provide 2-3 concrete, actionable improvement suggestions. \
End with a score in the format \"Score: X/10\" and one sentence explaining the score.";

const ADVERSARIAL_CRITIC: &str = "\
You are an Adversarial Critic. Challenge every assumption in the Actor's response. Find \
logical gaps, unsupported claims, and edge cases that fail. Be relentless — even strong \
responses have exploitable weak points. Provide 2-3 targeted attacks on the weakest aspects. \
End with a score in the format \"Score: X/10\".";

const SOCRATIC_CRITIC: &str = "\
You are a Socratic Critic. Rather than giving direct feedback, ask 2-3 probing questions \
that reveal unstated assumptions or gaps in the Actor's reasoning. Each question should be \
specific enough that answering it would meaningfully improve the response. End with a score \
in the format \"Score: X/10\" and one sentence on what the score reflects.";

const PEER_CRITIC: &str = "\
You are a Peer Reviewer. Give collegial, constructive feedback as if reviewing a colleague's \
work. Acknowledge what works well before identifying areas for improvement. Suggest 2-3 \
specific, actionable changes in a supportive tone. End with a score in the format \
\"Score: X/10\" and a brief rationale.";

const REDTEAM_CRITIC: &str = "\
You are a Red Team Critic. Look for security vulnerabilities, failure modes, ways the \
response could be misused, and assumptions that break under adversarial conditions. Think \
like an attacker or stress-tester. Provide 2-3 specific risks or failure cases. End with \
a score in the format \"Score: X/10\".";

// ── CriticMode ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Default)]
pub enum CriticMode {
    #[default]
    Default,
    Adversarial,
    Socratic,
    Peer,
    RedTeam,
}

impl CriticMode {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "adversarial" | "adv" => Self::Adversarial,
            "socratic"    | "soc" => Self::Socratic,
            "peer"                => Self::Peer,
            "redteam"    | "red"  => Self::RedTeam,
            _                     => Self::Default,
        }
    }

    pub fn system_prompt(&self) -> &'static str {
        match self {
            Self::Default     => DEFAULT_CRITIC,
            Self::Adversarial => ADVERSARIAL_CRITIC,
            Self::Socratic    => SOCRATIC_CRITIC,
            Self::Peer        => PEER_CRITIC,
            Self::RedTeam     => REDTEAM_CRITIC,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Default     => "default",
            Self::Adversarial => "adversarial",
            Self::Socratic    => "socratic",
            Self::Peer        => "peer",
            Self::RedTeam     => "redteam",
        }
    }
}

// ── Config ────────────────────────────────────────────────────────────────────

/// Runtime configuration merged from config file then CLI args (CLI wins).
#[derive(Debug, Clone)]
pub struct Config {
    pub actor_model:   String,
    pub critic_model:  String,
    pub ollama_url:    String,
    pub stop_at_score: Option<u8>,
    pub context_turns: usize,
    pub max_history:   Option<usize>,
    pub critic_mode:   CriticMode,
    /// Custom actor system prompt (overrides DEFAULT_ACTOR).
    pub actor_system:  Option<String>,
    /// Custom critic system prompt (overrides critic_mode selection).
    pub critic_system: Option<String>,
    /// Per-role Ollama URL overrides. Falls back to `ollama_url` when None.
    pub actor_ollama_url:  Option<String>,
    pub critic_ollama_url: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            actor_model:   "qwen3.5:4b".into(),
            critic_model:  "qwen3.5:4b".into(),
            ollama_url:    "http://127.0.0.1:11434".into(),
            stop_at_score: None,
            context_turns: 4,
            max_history:   None,
            critic_mode:   CriticMode::Default,
            actor_system:  None,
            critic_system: None,
            actor_ollama_url:  None,
            critic_ollama_url: None,
        }
    }
}

impl Config {
    pub fn actor_url(&self) -> &str {
        self.actor_ollama_url.as_deref().unwrap_or(&self.ollama_url)
    }

    pub fn critic_url(&self) -> &str {
        self.critic_ollama_url.as_deref().unwrap_or(&self.ollama_url)
    }

    pub fn actor_system_prompt(&self) -> &str {
        self.actor_system.as_deref().unwrap_or(DEFAULT_ACTOR)
    }

    /// Returns critic system prompt: custom string > critic_mode archetype.
    pub fn critic_system_prompt(&self) -> &str {
        self.critic_system.as_deref()
            .unwrap_or_else(|| self.critic_mode.system_prompt())
    }

    /// Load from `~/.config/duel/config.toml` (if present), then apply CLI args.
    /// Returns `None` if `--help` was requested.
    pub fn load() -> Option<Config> {
        let mut cfg = Config::default();
        if let Some(file_cfg) = load_toml_file() {
            apply_toml(&mut cfg, file_cfg);
        }
        apply_cli_args(&mut cfg)?;
        Some(cfg)
    }

    /// Alias kept for backward compatibility.
    pub fn from_args() -> Option<Config> {
        Self::load()
    }

    pub fn help_text() -> &'static str {
        "\
Usage: duel [OPTIONS]
       duel resume <session.json>

Options:
  --actor <model>         Ollama model for Actor role         [default: qwen3.5:4b]
  --critic <model>        Ollama model for Critic role        [default: qwen3.5:4b]
  --ollama-url <url>      Ollama base URL                     [default: http://127.0.0.1:11434]
  --stop-at-score <n>     Auto-pause when Critic score >= n (1-10)
  --context-turns <n>     Recent turns included in context    [default: 4]
  --max-history <n>       Cap history at N entries (oldest trimmed)
  --critic-mode <mode>    Critic archetype: default | adversarial | socratic | peer | redteam
  --actor-system <file>      Load Actor system prompt from a text file
  --critic-system <file>     Load Critic system prompt from a text file
  --actor-ollama-url <url>   Ollama URL for Actor  [default: --ollama-url]
  --critic-ollama-url <url>  Ollama URL for Critic [default: --ollama-url]
  --help, -h                 Show this help

Subcommands:
  resume <file>           Continue a saved JSON session from where it left off

Config file (CLI args override): ~/.config/duel/config.toml

Keys (inside TUI):
  Enter       Start loop          Space       Pause / resume
  e           Edit prompt         s           Save JSON
  m           Export Markdown     h           Export HTML
  ↑↓ / k j   Scroll history      Ctrl+T      Cycle theme
  ?           Help overlay        q / Ctrl+C  Quit
"
    }
}

// ── TOML file loading ─────────────────────────────────────────────────────────

#[derive(Debug, Deserialize, Default)]
struct TomlConfig {
    actor_model:   Option<String>,
    critic_model:  Option<String>,
    ollama_url:    Option<String>,
    stop_at_score: Option<u8>,
    context_turns: Option<usize>,
    max_history:   Option<usize>,
    critic_mode:   Option<String>,
    actor_system:      Option<String>,
    critic_system:     Option<String>,
    actor_ollama_url:  Option<String>,
    critic_ollama_url: Option<String>,
}

fn load_toml_file() -> Option<TomlConfig> {
    let home = std::env::var("HOME").ok()?;
    let path = std::path::Path::new(&home).join(".config/duel/config.toml");
    let text = std::fs::read_to_string(path).ok()?;
    toml::from_str(&text).ok()
}

fn apply_toml(cfg: &mut Config, t: TomlConfig) {
    if let Some(v) = t.actor_model   { cfg.actor_model   = v; }
    if let Some(v) = t.critic_model  { cfg.critic_model  = v; }
    if let Some(v) = t.ollama_url    { cfg.ollama_url    = v; }
    if let Some(v) = t.stop_at_score { cfg.stop_at_score = Some(v.min(10)); }
    if let Some(v) = t.context_turns { cfg.context_turns = v; }
    if let Some(v) = t.max_history   { cfg.max_history   = Some(v); }
    if let Some(v) = t.critic_mode   { cfg.critic_mode   = CriticMode::from_str(&v); }
    if let Some(v) = t.actor_system      { cfg.actor_system      = Some(v); }
    if let Some(v) = t.critic_system     { cfg.critic_system     = Some(v); }
    if let Some(v) = t.actor_ollama_url  { cfg.actor_ollama_url  = Some(v); }
    if let Some(v) = t.critic_ollama_url { cfg.critic_ollama_url = Some(v); }
}

// ── CLI argument parsing ──────────────────────────────────────────────────────

fn apply_cli_args(cfg: &mut Config) -> Option<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--help" | "-h" => return None,
            // subcommand — consume the path arg but leave detection to main
            "resume" => { it.next(); }
            "--actor" => {
                if let Some(v) = it.next() { cfg.actor_model = v.clone(); }
            }
            "--critic" => {
                if let Some(v) = it.next() { cfg.critic_model = v.clone(); }
            }
            "--ollama-url" => {
                if let Some(v) = it.next() { cfg.ollama_url = v.clone(); }
            }
            "--stop-at-score" => {
                if let Some(v) = it.next() {
                    if let Ok(n) = v.parse::<u8>() { cfg.stop_at_score = Some(n.min(10)); }
                }
            }
            "--context-turns" => {
                if let Some(v) = it.next() {
                    if let Ok(n) = v.parse::<usize>() { cfg.context_turns = n; }
                }
            }
            "--max-history" => {
                if let Some(v) = it.next() {
                    if let Ok(n) = v.parse::<usize>() { cfg.max_history = Some(n); }
                }
            }
            "--critic-mode" => {
                if let Some(v) = it.next() { cfg.critic_mode = CriticMode::from_str(v); }
            }
            "--actor-system" => {
                if let Some(path) = it.next() {
                    match std::fs::read_to_string(path) {
                        Ok(s) => cfg.actor_system = Some(s.trim().to_string()),
                        Err(e) => eprintln!("Warning: --actor-system: {e}"),
                    }
                }
            }
            "--critic-system" => {
                if let Some(path) = it.next() {
                    match std::fs::read_to_string(path) {
                        Ok(s) => cfg.critic_system = Some(s.trim().to_string()),
                        Err(e) => eprintln!("Warning: --critic-system: {e}"),
                    }
                }
            }
            "--actor-ollama-url" => {
                if let Some(v) = it.next() { cfg.actor_ollama_url = Some(v.clone()); }
            }
            "--critic-ollama-url" => {
                if let Some(v) = it.next() { cfg.critic_ollama_url = Some(v.clone()); }
            }
            _ => {}
        }
    }
    Some(())
}
