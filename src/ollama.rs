use anyhow::Result;
use reqwest::Client;
use serde::Serialize;
use std::sync::{Arc, OnceLock};
use tokio::sync::mpsc;
use tokio::time::timeout;

use crate::app::{AppCommand, Role, Turn};
use crate::config::Config;

static CLIENT: OnceLock<Client> = OnceLock::new();

fn get_client() -> &'static Client {
    CLIENT.get_or_init(Client::new)
}


#[derive(Serialize)]
struct GenerateReq {
    model:      String,
    prompt:     String,
    system:     String,
    stream:     bool,
    keep_alive: i64,
    options:    Opts,
}

#[derive(Serialize)]
struct Opts {
    temperature: f32,
}

pub async fn pre_warm(tx: mpsc::Sender<AppCommand>, cfg: Arc<Config>) {
    let warm_actor = async {
        get_client()
            .post(format!("{}/api/generate", cfg.actor_url()))
            .json(&GenerateReq {
                model:      cfg.actor_model.clone(),
                prompt:     "ready".into(),
                system:     "You are a helpful assistant.".into(),
                stream:     false,
                keep_alive: -1,
                options:    Opts { temperature: 0.1 },
            })
            .timeout(std::time::Duration::from_secs(120))
            .send()
            .await
    };

    let actor_eq_critic = cfg.actor_model == cfg.critic_model
        && cfg.actor_url() == cfg.critic_url();

    if actor_eq_critic {
        match warm_actor.await {
            Ok(_)  => { let _ = tx.send(AppCommand::ModelReady).await; }
            Err(e) => { let _ = tx.send(AppCommand::LoopError(format!("pre-warm failed: {e}"))).await; }
        }
        return;
    }

    let warm_critic = async {
        get_client()
            .post(format!("{}/api/generate", cfg.critic_url()))
            .json(&GenerateReq {
                model:      cfg.critic_model.clone(),
                prompt:     "ready".into(),
                system:     "You are a helpful assistant.".into(),
                stream:     false,
                keep_alive: -1,
                options:    Opts { temperature: 0.1 },
            })
            .timeout(std::time::Duration::from_secs(120))
            .send()
            .await
    };

    let (ra, rc) = tokio::join!(warm_actor, warm_critic);
    match (ra, rc) {
        (Ok(_), Ok(_)) => { let _ = tx.send(AppCommand::ModelReady).await; }
        (Err(e), _) | (_, Err(e)) => {
            let _ = tx.send(AppCommand::LoopError(format!("pre-warm failed: {e}"))).await;
        }
    }
}

pub async fn run_actor(
    tx:      mpsc::Sender<AppCommand>,
    prompt:  String,
    history: Vec<Turn>,
    iter:    usize,
    cfg:     Arc<Config>,
) {
    let recent: Vec<&Turn> = history.iter().rev().take(cfg.context_turns).collect();
    let last_critic = recent.iter().find(|t| t.role == Role::Critic);

    let body = if iter == 1 || last_critic.is_none() {
        format!("{prompt}")
    } else {
        format!(
            "User task: {prompt}\n\nCritic feedback on your last response:\n{}\n\nGenerate your improved response now:",
            last_critic.unwrap().content
        )
    };

    let actor_sys = cfg.actor_system_prompt().to_string();
    stream_turn(tx, Role::Actor, &actor_sys, body, 0.7, cfg).await;
}

pub async fn run_critic(
    tx:      mpsc::Sender<AppCommand>,
    history: Vec<Turn>,
    iter:    usize,
    cfg:     Arc<Config>,
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

    let critic_sys = cfg.critic_system_prompt().to_string();
    stream_turn(tx, Role::Critic, &critic_sys, body, 0.4, cfg).await;
}

async fn stream_turn(
    tx:          mpsc::Sender<AppCommand>,
    role:        Role,
    system:      &str,
    prompt:      String,
    temperature: f32,
    cfg:         Arc<Config>,
) {
    if let Err(e) = do_stream(tx.clone(), role, system, prompt, temperature, cfg).await {
        let _ = tx.send(AppCommand::LoopError(e.to_string())).await;
    }
}

async fn do_stream(
    tx:          mpsc::Sender<AppCommand>,
    role:        Role,
    system:      &str,
    prompt:      String,
    temperature: f32,
    cfg:         Arc<Config>,
) -> Result<()> {
    let (model, url) = match role {
        Role::Actor  => (cfg.actor_model.clone(),  cfg.actor_url().to_string()),
        Role::Critic => (cfg.critic_model.clone(), cfg.critic_url().to_string()),
    };
    let body = GenerateReq {
        model,
        prompt,
        system:     system.into(),
        stream:     true,
        keep_alive: -1,
        options:    Opts { temperature },
    };

    let mut resp = get_client()
        .post(format!("{url}/api/generate"))
        .json(&body)
        .timeout(std::time::Duration::from_secs(300))
        .send()
        .await?;

    let mut buf: Vec<u8> = Vec::new();
    let mut eval_count  = 0u32;
    let mut eval_dur_ns = 0u64;

    loop {
        let chunk = timeout(
            std::time::Duration::from_secs(60),
            resp.chunk(),
        )
        .await
        .map_err(|_| anyhow::anyhow!("streaming timeout: no chunk for 60s"))??;
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
