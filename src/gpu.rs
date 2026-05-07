use tokio::process::Command;
use tokio::sync::mpsc;
use tokio::time::{sleep, Duration};

use crate::app::AppCommand;

const GPU_POLL_SECS: u64 = 5;

#[derive(Debug, Clone, Default)]
pub struct GpuInfo {
    pub name: String,
    pub util: u8,
    pub mem_used_mb: u32,
    pub mem_total_mb: u32,
}

pub async fn poll_loop(tx: mpsc::Sender<AppCommand>) {
    loop {
        let gpus = query().await;
        if !gpus.is_empty() {
            let _ = tx.send(AppCommand::GpuStatus(gpus)).await;
        }
        sleep(Duration::from_secs(GPU_POLL_SECS)).await;
    }
}

async fn query() -> Vec<GpuInfo> {
    let out = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .await;
    let out = match out {
        Ok(o) => o,
        Err(_) => return vec![],
    };
    let text = String::from_utf8_lossy(&out.stdout);
    text.lines().filter_map(parse_line).collect()
}

fn parse_line(line: &str) -> Option<GpuInfo> {
    let parts: Vec<&str> = line.splitn(4, ',').collect();
    if parts.len() < 4 { return None; }
    let name = shorten(parts[0].trim());
    let util: u8  = parts[1].trim().trim_end_matches('%').trim().parse().ok()?;
    let used: u32 = parts[2].trim().trim_end_matches("MiB").trim().parse().ok()?;
    let total: u32 = parts[3].trim().trim_end_matches("MiB").trim().parse().ok()?;
    Some(GpuInfo { name, util, mem_used_mb: used, mem_total_mb: total })
}

fn shorten(name: &str) -> String {
    // "NVIDIA GeForce RTX 5080" → "RTX 5080"
    if let Some(idx) = name.rfind("RTX") {
        return name[idx..].to_string();
    }
    if let Some(idx) = name.rfind("GTX") {
        return name[idx..].to_string();
    }
    name.to_string()
}
