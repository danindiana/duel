# Contributing to duel

## Getting started

```bash
git clone https://github.com/danindiana/duel
cd duel
cargo build
```

You need [Ollama](https://ollama.com) running locally with at least one model pulled (`ollama pull qwen3.5:4b`).

## Code style

- `cargo fmt` before committing (enforced by CI)
- `cargo clippy -- -D warnings` must pass (enforced by CI)
- `cargo test` must pass

## Submitting changes

1. Fork the repo and create a branch from `main`
2. Make your changes, add tests where applicable
3. Open a pull request — describe what changes and why

## Areas to contribute

- **New Critic archetypes** — add a variant to `CriticMode` in `src/config.rs` with a new system prompt constant
- **Export formats** — add an `export_*` method to `App` in `src/app.rs` and bind a key in `src/main.rs`
- **TUI improvements** — rendering lives in `src/ui.rs`; themes are trivial to add
- **Tests** — `src/app.rs` and `src/util.rs` have unit test modules; more coverage is always welcome

## Commit messages

Short imperative subject line, present tense. Body optional for non-obvious changes.
```
Add redteam critic archetype
Fix score parser for "8 / 10" with spaces around slash
```
