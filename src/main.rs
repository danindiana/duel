mod app;
mod config;
mod gpu;
mod ollama;
mod ui;
mod util;

use anyhow::Result;
use config::Config;
use crossterm::{
    event::{self, Event, KeyCode, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};
use std::{io, path::PathBuf, sync::Arc, time::Duration};
use tokio::sync::mpsc;

use app::{App, AppCommand, LoopState};

const EVENT_POLL_MS:    u64   = 40;
const CHANNEL_CAPACITY: usize = 256;

#[tokio::main]
async fn main() -> Result<()> {
    let cfg = match Config::from_args() {
        Some(c) => Arc::new(c),
        None => { print!("{}", Config::help_text()); return Ok(()); }
    };

    let project_dir = std::env::current_dir()
        .unwrap_or_else(|_| PathBuf::from("."));

    // Panic hook: always restore terminal
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = disable_raw_mode();
        let _ = crossterm::execute!(
            std::io::stderr(),
            LeaveAlternateScreen,
        );
        default_hook(info);
    }));

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let (tx, rx) = mpsc::channel::<AppCommand>(CHANNEL_CAPACITY);

    tokio::spawn(gpu::poll_loop(tx.clone()));

    let mut app = App::new(tx, project_dir, cfg);

    let result = run_app(&mut terminal, &mut app, rx).await;

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    result
}

async fn run_app(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app:      &mut App,
    mut rx:   mpsc::Receiver<AppCommand>,
) -> Result<()> {
    loop {
        terminal.draw(|f| ui::draw(f, app))?;

        while let Ok(cmd) = rx.try_recv() {
            app.handle_command(cmd);
        }

        if event::poll(Duration::from_millis(EVENT_POLL_MS))? {
            if let Event::Key(key) = event::read()? {
                match (&app.state.clone(), key.modifiers, key.code) {

                    // ── Global ─────────────────────────────────────────────
                    (_, KeyModifiers::CONTROL, KeyCode::Char('c')) => return Ok(()),
                    (_, _, KeyCode::Char('q')) if !matches!(app.state, LoopState::Idle) => {
                        return Ok(())
                    }
                    (LoopState::Idle, _, KeyCode::Char('q')) => return Ok(()),

                    // ── Help overlay ───────────────────────────────────────
                    (_, _, KeyCode::Char('?')) => {
                        app.show_help = !app.show_help;
                    }
                    (_, _, KeyCode::Esc) if app.show_help => {
                        app.show_help = false;
                    }

                    // ── Prompt entry (Idle) ────────────────────────────────
                    (LoopState::Idle, _, KeyCode::Enter) => {
                        app.start_loop();
                    }
                    (LoopState::Idle, _, KeyCode::Backspace) => {
                        app.input_buf.pop();
                    }
                    (LoopState::Idle, _, KeyCode::Char(c)) if !app.show_help => {
                        app.input_buf.push(c);
                    }

                    // ── Loop controls ──────────────────────────────────────
                    (_, _, KeyCode::Char(' ')) if !matches!(app.state, LoopState::Idle) => {
                        app.toggle_pause();
                    }
                    (LoopState::Paused, _, KeyCode::Char('e')) => {
                        app.edit_prompt();
                    }
                    (LoopState::Error(_), _, KeyCode::Char('e')) => {
                        app.edit_prompt();
                    }

                    // ── Save session ───────────────────────────────────────
                    (_, _, KeyCode::Char('s')) if !matches!(app.state, LoopState::Idle) => {
                        app.save_session();
                    }

                    // ── Markdown export ────────────────────────────────────
                    (_, _, KeyCode::Char('m')) if !matches!(app.state, LoopState::Idle) => {
                        app.export_markdown();
                    }

                    // ── History scroll ─────────────────────────────────────
                    (_, _, KeyCode::Up)   | (_, _, KeyCode::Char('k')) => app.scroll_up(),
                    (_, _, KeyCode::Down) | (_, _, KeyCode::Char('j')) => app.scroll_down(),

                    // ── Theme cycle ────────────────────────────────────────
                    (_, KeyModifiers::CONTROL, KeyCode::Char('t')) => {
                        app.theme_idx = (app.theme_idx + 1) % ui::NUM_THEMES;
                    }

                    _ => {}
                }
            }
        }
    }
}
