use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, BorderType, Borders, Clear, Paragraph, Wrap},
    Frame,
};

use crate::app::{App, LoopState, Role, Turn};
use crate::gpu::GpuInfo;
use crate::util::truncate;

// ── Themes ────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
pub struct Rgb(pub u8, pub u8, pub u8);

impl From<Rgb> for Color {
    fn from(Rgb(r, g, b): Rgb) -> Self { Color::Rgb(r, g, b) }
}

#[derive(Clone, Copy)]
struct ThemeColors {
    title_fg:      Color,
    title_bg:      Color,
    select_base:   Rgb,
    select_bright: Rgb,
    actor_fg:      Color,
    critic_fg:     Color,
    accent_fg:     Color,
    dim_fg:        Color,
    text_fg:       Color,
}

pub const NUM_THEMES: usize = 5;
pub const THEME_NAMES: &[&str] = &["Classic", "Dracula", "Nord", "Solarized", "Matrix"];

fn theme(idx: usize) -> ThemeColors {
    match idx % NUM_THEMES {
        0 => ThemeColors {
            title_fg:      Color::White,
            title_bg:      Color::DarkGray,
            select_base:   Rgb(0, 30, 120),
            select_bright: Rgb(40, 160, 255),
            actor_fg:      Color::Cyan,
            critic_fg:     Color::Yellow,
            accent_fg:     Color::Green,
            dim_fg:        Color::DarkGray,
            text_fg:       Color::Gray,
        },
        1 => ThemeColors {  // Dracula
            title_fg:      Color::Rgb(248, 248, 242),
            title_bg:      Color::Rgb(68, 71, 90),
            select_base:   Rgb(98, 58, 150),
            select_bright: Rgb(189, 84, 255),
            actor_fg:      Color::Rgb(139, 233, 253),
            critic_fg:     Color::Rgb(255, 184, 108),
            accent_fg:     Color::Rgb(80, 250, 123),
            dim_fg:        Color::Rgb(68, 71, 90),
            text_fg:       Color::Rgb(248, 248, 242),
        },
        2 => ThemeColors {  // Nord
            title_fg:      Color::Rgb(216, 222, 233),
            title_bg:      Color::Rgb(46, 52, 64),
            select_base:   Rgb(46, 90, 130),
            select_bright: Rgb(136, 192, 208),
            actor_fg:      Color::Rgb(136, 192, 208),
            critic_fg:     Color::Rgb(235, 203, 139),
            accent_fg:     Color::Rgb(163, 190, 140),
            dim_fg:        Color::Rgb(67, 76, 94),
            text_fg:       Color::Rgb(216, 222, 233),
        },
        3 => ThemeColors {  // Solarized
            title_fg:      Color::Rgb(131, 148, 150),
            title_bg:      Color::Rgb(7, 54, 66),
            select_base:   Rgb(0, 74, 80),
            select_bright: Rgb(42, 161, 152),
            actor_fg:      Color::Rgb(42, 161, 152),
            critic_fg:     Color::Rgb(181, 137, 0),
            accent_fg:     Color::Rgb(133, 153, 0),
            dim_fg:        Color::Rgb(7, 54, 66),
            text_fg:       Color::Rgb(131, 148, 150),
        },
        _ => ThemeColors {  // Matrix
            title_fg:      Color::Rgb(0, 200, 0),
            title_bg:      Color::Rgb(0, 10, 0),
            select_base:   Rgb(0, 50, 0),
            select_bright: Rgb(0, 255, 50),
            actor_fg:      Color::Rgb(0, 255, 100),
            critic_fg:     Color::Rgb(180, 255, 0),
            accent_fg:     Color::Rgb(100, 255, 100),
            dim_fg:        Color::Rgb(0, 50, 0),
            text_fg:       Color::Rgb(0, 200, 0),
        },
    }
}

const GLOW_PERIOD_MS: f32 = 1400.0;

fn glow_style(base: Rgb, bright: Rgb) -> Style {
    let ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as f32;
    let t = ((ms % GLOW_PERIOD_MS) / GLOW_PERIOD_MS * 2.0 * std::f32::consts::PI).sin();
    let t = (t + 1.0) / 2.0;
    let r = (base.0 as f32 + (bright.0 as f32 - base.0 as f32) * t) as u8;
    let g = (base.1 as f32 + (bright.1 as f32 - base.1 as f32) * t) as u8;
    let b = (base.2 as f32 + (bright.2 as f32 - base.2 as f32) * t) as u8;
    Style::default().fg(Color::Rgb(r, g, b))
}

// ── Top-level draw ────────────────────────────────────────────────────────────

pub fn draw(f: &mut Frame, app: &App) {
    let area = f.area();
    let tc = theme(app.theme_idx);

    match &app.state {
        LoopState::Idle => draw_prompt(f, area, app, tc),
        _ => draw_duel(f, area, app, tc),
    }

    if app.show_help {
        draw_help(f, area, tc);
    }
}

// ── Prompt entry view ─────────────────────────────────────────────────────────

fn draw_prompt(f: &mut Frame, area: Rect, app: &App, tc: ThemeColors) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Min(0), Constraint::Length(1)])
        .split(area);

    // Title
    let title = format!(
        "  duel  ·  {}  ·  {}",
        app.cfg.actor_model,
        THEME_NAMES[app.theme_idx % NUM_THEMES]
    );
    f.render_widget(
        Paragraph::new(title).style(Style::default().fg(tc.title_fg).bg(tc.title_bg)),
        rows[0],
    );

    // Body
    let body_rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2),
            Constraint::Length(3),
            Constraint::Length(2),
            Constraint::Min(0),
        ])
        .split(rows[1]);

    // Input box
    let input_text = format!(" > {}▌", app.input_buf);
    let input_para = Paragraph::new(input_text)
        .block(
            Block::default()
                .title(" Prompt ")
                .borders(Borders::ALL)
                .border_type(BorderType::Double)
                .border_style(glow_style(tc.select_base, tc.select_bright)),
        )
        .style(Style::default().fg(tc.text_fg));
    f.render_widget(input_para, body_rows[1]);

    // Description
    let desc = vec![
        Line::from(""),
        Line::from(Span::styled(
            "  Actor generates  →  Critic scores  →  Actor refines  →  ∞",
            Style::default().fg(tc.accent_fg),
        )),
        Line::from(Span::styled(
            "  The loop runs until you pause (Space) or quit (q).",
            Style::default().fg(tc.dim_fg),
        )),
    ];
    f.render_widget(Paragraph::new(desc), body_rows[2]);

    // Status bar
    f.render_widget(
        Paragraph::new("  Enter:start   Backspace:delete   Ctrl+T:theme   ?:help   q:quit")
            .style(Style::default().fg(tc.title_fg).bg(tc.title_bg)),
        rows[2],
    );
}

// ── Duel loop view ────────────────────────────────────────────────────────────

fn draw_duel(f: &mut Frame, area: Rect, app: &App, tc: ThemeColors) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),  // title bar
            Constraint::Percentage(55),  // live panels
            Constraint::Min(4),     // history
            Constraint::Length(1),  // status bar
        ])
        .split(area);

    draw_title_bar(f, rows[0], app, tc);
    draw_live_panels(f, rows[1], app, tc);
    draw_history(f, rows[2], app, tc);
    draw_status_bar(f, rows[3], app, tc);
}

fn draw_title_bar(f: &mut Frame, area: Rect, app: &App, tc: ThemeColors) {
    let state_badge = match &app.state {
        LoopState::Prewarm    => "[⏳ WARMING]",
        LoopState::ActorThink => "[● RUNNING]",
        LoopState::CriticThink=> "[● RUNNING]",
        LoopState::Paused     => "[⏸ PAUSED ]",
        LoopState::Error(_)   => "[✗ ERROR  ]",
        LoopState::Idle       => "[  IDLE   ]",
    };
    let badge_color = match &app.state {
        LoopState::ActorThink | LoopState::CriticThink => Color::Green,
        LoopState::Paused => Color::Yellow,
        LoopState::Error(_) => Color::Red,
        _ => tc.dim_fg,
    };
    let gpu_str = format_gpu_brief(&app.gpu);
    let iter_str = if app.iteration == 0 {
        String::new()
    } else {
        format!("  iter {}", app.iteration)
    };
    let theme_name = THEME_NAMES[app.theme_idx % NUM_THEMES];
    let text = format!("  duel  ·  {}{iter_str}  ·  {gpu_str}  [{theme_name}]", app.cfg.actor_model);

    // Render base title then overlay badge
    f.render_widget(
        Paragraph::new(text).style(Style::default().fg(tc.title_fg).bg(tc.title_bg)),
        area,
    );
    // Overlay badge at right
    let badge_len = state_badge.chars().count() as u16 + 2;
    if area.width > badge_len + 4 {
        let badge_area = Rect {
            x: area.x + area.width - badge_len - 1,
            y: area.y,
            width: badge_len + 1,
            height: 1,
        };
        f.render_widget(
            Paragraph::new(format!("{state_badge} "))
                .style(Style::default().fg(badge_color).bg(tc.title_bg).add_modifier(Modifier::BOLD)),
            badge_area,
        );
    }
}

fn draw_live_panels(f: &mut Frame, area: Rect, app: &App, tc: ThemeColors) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(55), Constraint::Percentage(45)])
        .split(area);

    let actor_active  = matches!(&app.state, LoopState::ActorThink);
    let critic_active = matches!(&app.state, LoopState::CriticThink);

    draw_panel(f, cols[0], "ACTOR",  Role::Actor,  &app.actor_buf,  &app.history, actor_active,  tc);
    draw_panel(f, cols[1], "CRITIC", Role::Critic, &app.critic_buf, &app.history, critic_active, tc);
}

fn draw_panel(
    f:       &mut Frame,
    area:    Rect,
    label:   &str,
    role:    Role,
    buf:     &str,
    history: &[Turn],
    active:  bool,
    tc:      ThemeColors,
) {
    let role_color = match role {
        Role::Actor  => tc.actor_fg,
        Role::Critic => tc.critic_fg,
    };

    // Find last completed turn for this role
    let last_turn = history.iter().rev().find(|t| t.role == role);
    let score_str = if role == Role::Critic {
        last_turn
            .and_then(|t| t.score)
            .map(|s| {
                let color = if s >= 8 { "▲" } else if s >= 5 { "~" } else { "▼" };
                format!("  {color} {s}/10")
            })
            .unwrap_or_default()
    } else {
        String::new()
    };

    let title = format!(" {label}{score_str} ");

    let border_style = if active {
        glow_style(tc.select_base, tc.select_bright)
    } else {
        Style::default().fg(tc.dim_fg)
    };

    // Show streaming buffer if active, otherwise show last completed turn
    let display_text = if active && !buf.is_empty() {
        buf.to_string()
    } else if !active && buf.is_empty() {
        last_turn.map(|t| t.content.clone()).unwrap_or_default()
    } else {
        buf.to_string()
    };

    let cursor = if active { "▌" } else { "" };
    let full_text = format!("{display_text}{cursor}");

    // Word-wrap lines for the panel
    let lines: Vec<Line> = full_text
        .lines()
        .map(|l| Line::from(Span::styled(l.to_string(), Style::default().fg(tc.text_fg))))
        .collect();

    let para = Paragraph::new(Text::from(lines))
        .block(
            Block::default()
                .title(Span::styled(title, Style::default().fg(role_color).add_modifier(Modifier::BOLD)))
                .borders(Borders::ALL)
                .border_style(border_style),
        )
        .wrap(Wrap { trim: false });

    f.render_widget(para, area);
}

fn draw_history(f: &mut Frame, area: Rect, app: &App, tc: ThemeColors) {
    if area.height < 3 { return; }

    let inner_height = area.height.saturating_sub(2) as usize;
    let total_turns  = app.history.len();

    // Each turn renders as a header line + content lines (wrapped to area width - 4)
    let content_width = (area.width.saturating_sub(4)) as usize;

    // Build all display lines from history
    let mut all_lines: Vec<Line> = Vec::new();
    for turn in &app.history {
        let role_color = match turn.role {
            Role::Actor  => tc.actor_fg,
            Role::Critic => tc.critic_fg,
        };
        let tps = if turn.duration_ms > 0 {
            format!(" {:.0}t/s", turn.eval_count as f64 / (turn.duration_ms as f64 / 1000.0))
        } else { String::new() };
        let score_str = turn.score.map(|s| format!("  score:{s}/10")).unwrap_or_default();
        let header = format!(
            "[{}] iter {} — {}tok{}{tps}",
            turn.role.label(), turn.iteration, turn.eval_count, score_str
        );
        all_lines.push(Line::from(Span::styled(
            header,
            Style::default().fg(role_color).add_modifier(Modifier::BOLD),
        )));
        // Add abbreviated content (first line only, truncated)
        let first_line = turn.content.lines().next().unwrap_or("");
        let snippet = truncate(first_line, content_width.saturating_sub(4));
        all_lines.push(Line::from(Span::styled(
            format!("  {snippet}"),
            Style::default().fg(tc.dim_fg),
        )));
        all_lines.push(Line::from(""));
    }

    // Scroll: newest at bottom by default
    let total_lines = all_lines.len();
    let max_scroll  = total_lines.saturating_sub(inner_height) as u16;
    let effective_scroll = app.scroll.min(max_scroll);
    // Invert: scroll=0 shows the bottom (newest)
    let from_bottom_scroll = max_scroll.saturating_sub(effective_scroll);

    let title = format!(" History ({} turns) ", total_turns);
    let para = Paragraph::new(Text::from(all_lines))
        .block(
            Block::default()
                .title(Span::styled(title, Style::default().fg(tc.dim_fg)))
                .borders(Borders::TOP)
                .border_style(Style::default().fg(tc.dim_fg)),
        )
        .scroll((from_bottom_scroll, 0));

    f.render_widget(para, area);
}

fn draw_status_bar(f: &mut Frame, area: Rect, app: &App, tc: ThemeColors) {
    let hint = match &app.state {
        LoopState::Idle => "Enter:start  Backspace:del  Ctrl+T:theme  ?:help  q:quit",
        LoopState::Paused => "Space:resume  e:edit-prompt  s:save  m:md  ↑↓:history  Ctrl+T  ?:help  q",
        LoopState::Error(_) => "e:edit-prompt  q:quit",
        _ => "Space:pause  s:save  m:md  ↑↓:history  Ctrl+T:theme  ?:help  q:quit",
    };
    let status = if app.status.is_empty() {
        String::new()
    } else {
        format!("  │  {}", truncate(&app.status, 40))
    };
    let text = format!("  {hint}{status}");
    f.render_widget(
        Paragraph::new(text).style(Style::default().fg(tc.title_fg).bg(tc.title_bg)),
        area,
    );
}

// ── Help overlay ──────────────────────────────────────────────────────────────

fn draw_help(f: &mut Frame, area: Rect, tc: ThemeColors) {
    const BINDINGS: &[(&str, &str)] = &[
        ("Enter",        "Start duel loop (prompt view)"),
        ("Space",        "Pause / resume loop"),
        ("e",            "Edit prompt (when paused)"),
        ("s",            "Save session to JSON"),
        ("m",            "Export session to Markdown"),
        ("↑ / k",        "Scroll history up"),
        ("↓ / j",        "Scroll history down"),
        ("Ctrl+T",       "Cycle theme"),
        ("?",            "Toggle this help"),
        ("q / Ctrl+C",   "Quit"),
    ];

    let width  = 56u16.min(area.width.saturating_sub(4));
    let height = (BINDINGS.len() as u16 + 4).min(area.height.saturating_sub(4));
    let x = area.x + (area.width.saturating_sub(width)) / 2;
    let y = area.y + (area.height.saturating_sub(height)) / 2;
    let popup = Rect { x, y, width, height };

    let mut lines: Vec<Line> = vec![Line::from("")];
    for (key, desc) in BINDINGS {
        lines.push(Line::from(vec![
            Span::styled(
                format!("  {key:<16}"),
                Style::default().fg(tc.accent_fg).add_modifier(Modifier::BOLD),
            ),
            Span::styled(*desc, Style::default().fg(tc.text_fg)),
        ]));
    }

    let para = Paragraph::new(Text::from(lines))
        .block(
            Block::default()
                .title(" Help  [? or Esc to close] ")
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(tc.accent_fg).bg(tc.title_bg))
                .style(Style::default().bg(tc.title_bg)),
        );

    f.render_widget(Clear, popup);
    f.render_widget(para, popup);
}

// ── GPU brief ─────────────────────────────────────────────────────────────────

fn format_gpu_brief(gpus: &[GpuInfo]) -> String {
    if gpus.is_empty() { return "[GPU:--]".into(); }
    let parts: Vec<String> = gpus.iter().enumerate().map(|(i, g)| {
        format!("G{i}:{}% {}/{}M", g.util, g.mem_used_mb, g.mem_total_mb)
    }).collect();
    format!("[{}]", parts.join("  "))
}
