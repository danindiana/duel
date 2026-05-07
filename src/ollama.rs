use anyhow::Result;
use reqwest::Client;
use serde::Serialize;
use tokio::sync::mpsc;

use crate::app::{AppCommand, Role, Turn};

const OLLAMA_URL: &str = "http://127.0.0.1:11434";
const MODEL: &str = "qwen3.5:4b";
// Keep last N turns in context to avoid overflowing 32k context window
const CONTEXT_TURNS: usize = 4;

const ACTOR_SYSTEM: &str = "\
You are the Actor. Your job is to generate the best possible response to the user's task. \
Be thorough, precise, and creative. In later iterations, incorporate the Critic's feedback \
to improve your output.";

const CRITIC_SYSTEM: &str = "\
You are the Critic. Evaluate the Actor's latest response. Identify specific weaknesses, \
errors, or missed opportunities. Provide 2-3 concrete, actionable improvement suggestions. \
End with a score in the format \"Score: X/10\" and one sentence explaining the score.";

#[derive(Serialize)]
struct GenerateReq<'a> {
    model:  &'a str,
    prompt: String,
    system: &'a str,
    stream: bool,
    options: Opts,
}

#[derive(Serialize)]
struct Opts {
    temperature: f32,
}

pub async fn pre_warm(tx: mpsc::Sender<AppCommand>) {
    let client = Client::new();
    let body = GenerateReq {
        model:   MODEL,
        prompt:  "ready".into(),
        system:  "You are a helpful assistant.",
        stream:  false,
        options: Opts { temperature: 0.1 },
    };
    let result = client
        .post(format!("{OLLAMA_URL}/api/generate"))
        .json(&body)
        .timeout(std::time::Duration::from_secs(20))
        .send()
        .await;
    match result {
        Ok(_) => { let _ = tx.send(AppCommand::ModelReady).await; }
        Err(e) => { let _ = tx.send(AppCommand::LoopError(format!("pre-warm failed: {e}"))).await; }
    }
}

pub async fn run_actor(
    tx:      mpsc::Sender<AppCommand>,
    prompt:  String,
    history: Vec<Turn>,
    iter:    usize,
) {
    let recent: Vec<&Turn> = history.iter().rev().take(CONTEXT_TURNS).collect();
    let last_critic = recent.iter().find(|t| t.role == Role::Critic);

    let body = if iter == 1 || last_critic.is_none() {
        format!("{prompt}")
    } else {
        format!(
            "User task: {prompt}\n\nCritic feedback on your last response:\n{}\n\nGenerate your improved response now:",
            last_critic.unwrap().content
        )
    };

    stream_turn(tx, Role::Actor, ACTOR_SYSTEM, body, 0.7).await;
}

pub async fn run_critic(
    tx:      mpsc::Sender<AppCommand>,
    history: Vec<Turn>,
    iter:    usize,
) {
    let last_actor = history.iter().rev().find(|t| t.role == Role::Actor);
    let Some(actor_turn) = last_actor else {
        let _ = tx.send(AppCommand::LoopError("no actor turn to critique".into())).await;
        return;
    };

    let body = format!(
        "Actor's response (iteration {iter}):\n{}\n\nEvaluate it:",
        actor_turn.content
    );

    stream_turn(tx, Role::Critic, CRITIC_SYSTEM, body, 0.4).await;
}

async fn stream_turn(
    tx:          mpsc::Sender<AppCommand>,
    role:        Role,
    system:      &str,
    prompt:      String,
    temperature: f32,
) {
    if let Err(e) = do_stream(tx.clone(), role, system, prompt, temperature).await {
        let _ = tx.send(AppCommand::LoopError(e.to_string())).await;
    }
}

async fn do_stream(
    tx:          mpsc::Sender<AppCommand>,
    role:        Role,
    system:      &str,
    prompt:      String,
    temperature: f32,
) -> Result<()> {
    let client = Client::new();
    let body = GenerateReq {
        model: MODEL,
        prompt,
        system,
        stream: true,
        options: Opts { temperature },
    };

    let mut resp = client
        .post(format!("{OLLAMA_URL}/api/generate"))
        .json(&body)
        .timeout(std::time::Duration::from_secs(300))
        .send()
        .await?;

    let mut buf: Vec<u8> = Vec::new();
    let mut eval_count  = 0u32;
    let mut eval_dur_ns = 0u64;

    loop {
        let chunk = resp.chunk().await?;
        let Some(bytes) = chunk else { break; };
        buf.extend_from_slice(&bytes);

        // Process all complete newline-terminated lines in buf
        while let Some(pos) = buf.iter().position(|&b| b == b'\n') {
            let line_bytes: Vec<u8> = buf.drain(..=pos).collect();
            let trimmed = std::str::from_utf8(&line_bytes)
                .unwrap_or("")
                .trim();
            if trimmed.is_empty() { continue; }

            let Ok(v) = serde_json::from_str::<serde_json::Value>(trimmed) else { continue; };

            // Emit response tokens (skip thinking field — qwen3.5 chain-of-thought)
            if let Some(token) = v.get("response").and_then(|t| t.as_str()) {
                if !token.is_empty() {
                    let _ = tx.send(AppCommand::Token { role, token: token.to_string() }).await;
                }
            }

            if v.get("done").and_then(|d| d.as_bool()).unwrap_or(false) {
                eval_count  = v.get("eval_count").and_then(|x| x.as_u64()).unwrap_or(0) as u32;
                eval_dur_ns = v.get("eval_duration").and_then(|x| x.as_u64()).unwrap_or(0);
            }
        }
    }

    let duration_ms = eval_dur_ns / 1_000_000;
    let _ = tx.send(AppCommand::TurnDone { role, eval_count, duration_ms }).await;
    Ok(())
}
