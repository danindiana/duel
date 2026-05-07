/// Runtime configuration parsed from CLI arguments.
#[derive(Debug, Clone)]
pub struct Config {
    pub actor_model:   String,
    pub critic_model:  String,
    pub ollama_url:    String,
    pub stop_at_score: Option<u8>,
    pub context_turns: usize,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            actor_model:   "qwen3.5:4b".into(),
            critic_model:  "qwen3.5:4b".into(),
            ollama_url:    "http://127.0.0.1:11434".into(),
            stop_at_score: None,
            context_turns: 4,
        }
    }
}

impl Config {
    /// Parse CLI arguments. Returns `None` if `--help` was requested.
    pub fn from_args() -> Option<Config> {
        let mut cfg = Config::default();
        let args: Vec<String> = std::env::args().skip(1).collect();
        let mut it = args.iter();
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--help" | "-h" => return None,
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
                        if let Ok(n) = v.parse::<u8>() {
                            cfg.stop_at_score = Some(n.min(10));
                        }
                    }
                }
                "--context-turns" => {
                    if let Some(v) = it.next() {
                        if let Ok(n) = v.parse::<usize>() { cfg.context_turns = n; }
                    }
                }
                _ => {}
            }
        }
        Some(cfg)
    }

    pub fn help_text() -> &'static str {
        "\
Usage: duel [OPTIONS]

Options:
  --actor <model>         Ollama model for Actor role  [default: qwen3.5:4b]
  --critic <model>        Ollama model for Critic role [default: qwen3.5:4b]
  --ollama-url <url>      Ollama base URL              [default: http://127.0.0.1:11434]
  --stop-at-score <n>     Auto-pause when Critic score >= n (1-10)
  --context-turns <n>     Recent turns included in context [default: 4]
  --help, -h              Show this help

Keys (inside TUI):
  Enter       Start duel loop
  Space       Pause / resume
  e           Edit prompt (when paused)
  s           Save session as JSON
  m           Export session as Markdown
  ↑↓ / k j   Scroll history
  Ctrl+T      Cycle theme
  ?           Toggle help overlay
  q / Ctrl+C  Quit
"
    }
}
