# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- **Config file** — `~/.config/duel/config.toml` for persistent settings; CLI args override file values
- **Critic archetypes** — `--critic-mode` flag selects from `default`, `adversarial`, `socratic`, `peer`, `redteam`
- **Custom system prompts** — `--actor-system <file>` and `--critic-system <file>` load prompts from text files
- **Session resume** — `duel resume <session.json>` reloads a saved session and continues the loop
- **HTML export** — `h` key writes a self-contained styled `.html` file
- **History cap** — `--max-history <n>` trims oldest turns when history exceeds N entries
- **Context-window warning** — emits a status message when estimated context exceeds ~28 000 tokens
- **Asymmetric models** — `--actor` and `--critic` flags select different models for each role
- **Auto-stop** — `--stop-at-score <n>` auto-pauses when the Critic score reaches the threshold
- **Markdown export** — `m` key writes a `.md` file with YAML frontmatter
- **GitHub Actions CI** — fmt, clippy, test, and release-build jobs on every push/PR

### Fixed
- **Score parser** — now handles `"8 / 10"` (spaces around slash), markdown bold/italic, and `"score N"` without a colon; 17 unit tests added
- **Save filename collision** — timestamp now includes milliseconds
- **nvidia-smi stderr** — suppressed on non-NVIDIA machines
- **Streaming hang** — per-chunk 60 s timeout prevents infinite block on Ollama stall
- **Connection pooling** — shared `reqwest::Client` via `OnceLock` eliminates per-request TCP setup
- **Toggle-pause iteration bug** — resuming after a Critic turn now correctly increments the iteration counter

## [0.1.0] — 2026-05-07

### Added
- Initial release: Actor/Critic infinite refinement loop TUI
- Streaming tokens from Ollama displayed live in split panels
- 5 colour themes with animated glow, GPU stats, session JSON save
