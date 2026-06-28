//! The setup & serve phases for the `ptg` CLI.
//!
//! `ptg setup` prepares the local environment so that running a mesh is a
//! one-command affair: it detects `llama-server`, downloads the gated Gemma
//! QAT model into the cache, and writes a TOML config that `ptg` (the mesh
//! path) and `ptg serve` read for their defaults.
//!
//! `ptg serve` launches the local OpenAI-compatible inference server in the
//! foreground, reusing the config written by `ptg setup`.
//!
//! Boundary: setup automates everything *safe* (model fetch, config, server
//! detection/launch). It does **not** build or download `llama-server` itself
//! — that is GPU/platform/asset-naming brittle. If the server is missing,
//! setup prints exact install instructions and exits cleanly.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Args, Subcommand};
use serde::{Deserialize, Serialize};

// --- Defaults mirrored from scripts/start-gemma4-qat.sh ---------------------

/// HuggingFace repo for the default (verified) QAT model.
pub const DEFAULT_HF_REPO: &str = "unsloth/gemma-4-E2B-it-qat-GGUF";
/// GGUF file inside [`DEFAULT_HF_REPO`].
pub const DEFAULT_GGUF_FILE: &str = "gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf";
/// Model alias served by the server (`--alias`).
pub const DEFAULT_MODEL_ALIAS: &str = "gemma-4-e2b-qat";
/// Bind host.
pub const DEFAULT_HOST: &str = "127.0.0.1";
/// Bind port.
pub const DEFAULT_PORT: u16 = 18136;

/// A typed setup/serve error. Never panics; all fallible ops route through here.
#[derive(Debug, thiserror::Error)]
pub enum SetupError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("{0}")]
    Message(String),
}

/// The optional subcommand surface added to `ptg`.
///
/// `ptg` with no subcommand keeps running the mesh as before. `ptg setup` and
/// `ptg serve` route here.
#[derive(Subcommand, Debug)]
pub enum PhaseCommand {
    /// Prepare the local environment: detect llama-server, download the model,
    /// and write a config used by `ptg` and `ptg serve`.
    Setup(SetupArgs),
    /// Launch the local inference server in the foreground.
    Serve(ServeArgs),
}

#[derive(Args, Debug)]
pub struct SetupArgs {
    /// Assume "yes" to interactive prompts (e.g. a multi-GB model download).
    #[arg(long)]
    pub yes: bool,

    /// Re-download the model even if it is already present.
    #[arg(long)]
    pub force: bool,

    /// Print what would happen without downloading anything or writing a config.
    #[arg(long = "dry-run")]
    pub dry_run: bool,

    /// Directory for the GGUF model (default: OS cache dir under `ptg/gguf`).
    #[arg(long)]
    pub gguf_dir: Option<PathBuf>,

    /// Explicit path to `llama-server`. Overrides detection.
    #[arg(long)]
    pub llama_server: Option<PathBuf>,

    /// HuggingFace repo to download from (default: the verified QAT repo).
    #[arg(long, default_value = DEFAULT_HF_REPO)]
    pub hf_repo: String,

    /// GGUF file name inside the repo (default: the verified QAT file).
    #[arg(long, default_value = DEFAULT_GGUF_FILE)]
    pub gguf_file: String,

    /// Model alias served by the server.
    #[arg(long, default_value = DEFAULT_MODEL_ALIAS)]
    pub model_alias: String,

    /// Bind host.
    #[arg(long, default_value = DEFAULT_HOST)]
    pub host: String,

    /// Bind port.
    #[arg(long, default_value_t = DEFAULT_PORT)]
    pub port: u16,

    /// Environment variable holding a HuggingFace token for gated-model auth
    /// (default `HF_TOKEN`). The token is used only for the download and is
    /// never written to the config or disk by PTG.
    #[arg(long, default_value = "HF_TOKEN")]
    pub hf_token_env: String,
}

#[derive(Args, Debug)]
pub struct ServeArgs {
    /// Print the server command that would be launched, then exit.
    #[arg(long = "dry-run")]
    pub dry_run: bool,

    /// Override the config's bind host.
    #[arg(long)]
    pub host: Option<String>,

    /// Override the config's bind port.
    #[arg(long)]
    pub port: Option<u16>,

    /// Override the config's model alias.
    #[arg(long)]
    pub model_alias: Option<String>,
}

/// The persisted PTG environment config written by `setup`, read by `serve`
/// and (for defaults) by the mesh path.
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct PtgConfig {
    /// Absolute path to the `llama-server` binary (if detected).
    pub server_path: Option<String>,
    /// Absolute path to the GGUF model file.
    pub gguf_path: String,
    /// Bind host.
    pub host: String,
    /// Bind port.
    pub port: u16,
    /// Model alias served by the server.
    pub model_alias: String,
}

impl PtgConfig {
    /// `http://host:port` — the OpenAI-compatible base URL.
    pub fn server_url(&self) -> String {
        format!("http://{}:{}", self.host, self.port)
    }
}

// --- Path resolution --------------------------------------------------------

/// The PTG home dir: `$HOME` (unix) / `%USERPROFILE%` (windows), or empty.
pub fn home_dir() -> PathBuf {
    if let Ok(h) = std::env::var("HOME") {
        if !h.is_empty() {
            return PathBuf::from(h);
        }
    }
    if cfg!(windows) {
        if let Ok(h) = std::env::var("USERPROFILE") {
            if !h.is_empty() {
                return PathBuf::from(h);
            }
        }
    }
    PathBuf::new()
}

/// Config path: `$PTG_CONFIG`, else `$XDG_CONFIG_HOME/ptg/config.toml`
/// (unix) / `%APPDATA%\ptg\config.toml` (windows), else
/// `~/.config/ptg/config.toml`.
pub fn config_path() -> PathBuf {
    if let Ok(p) = std::env::var("PTG_CONFIG") {
        if !p.is_empty() {
            return PathBuf::from(p);
        }
    }
    if cfg!(windows) {
        if let Ok(appdata) = std::env::var("APPDATA") {
            if !appdata.is_empty() {
                return PathBuf::from(appdata).join("ptg").join("config.toml");
            }
        }
    }
    if let Ok(xdg) = std::env::var("XDG_CONFIG_HOME") {
        if !xdg.is_empty() {
            return PathBuf::from(xdg).join("ptg").join("config.toml");
        }
    }
    home_dir().join(".config").join("ptg").join("config.toml")
}

/// Default GGUF cache dir: `$XDG_CACHE_HOME/ptg/gguf` (unix) /
/// `%LOCALAPPDATA%\ptg\gguf` (windows), else `~/.cache/ptg/gguf`.
pub fn default_gguf_dir() -> PathBuf {
    if cfg!(windows) {
        if let Ok(local) = std::env::var("LOCALAPPDATA") {
            if !local.is_empty() {
                return PathBuf::from(local).join("ptg").join("gguf");
            }
        }
    }
    if let Ok(xdg) = std::env::var("XDG_CACHE_HOME") {
        if !xdg.is_empty() {
            return PathBuf::from(xdg).join("ptg").join("gguf");
        }
    }
    home_dir().join(".cache").join("ptg").join("gguf")
}

/// Load the config if it exists, else `None`. Never panics.
pub fn load_config() -> Option<PtgConfig> {
    let path = config_path();
    let contents = std::fs::read_to_string(&path).ok()?;
    toml::from_str(&contents).ok()
}

fn save_config(cfg: &PtgConfig) -> Result<(), SetupError> {
    let path = config_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let serialized = toml::to_string_pretty(cfg).map_err(|e| SetupError::Message(e.to_string()))?;
    let mut tmp = path.clone();
    tmp.set_extension("toml.tmp");
    std::fs::write(&tmp, serialized)?;
    std::fs::rename(&tmp, &path)?;
    Ok(())
}

// --- Server detection -------------------------------------------------------

/// All `llama-server` candidates, in resolution order. The `.exe` suffix is
/// appended automatically on Windows for non-PATH candidates.
pub fn server_candidates(explicit: Option<&Path>) -> Vec<PathBuf> {
    let exe = if cfg!(windows) {
        "llama-server.exe"
    } else {
        "llama-server"
    };
    let with_suffix = |p: &str| {
        let pb = PathBuf::from(p);
        if cfg!(windows) && pb.extension().is_none() {
            PathBuf::from(format!("{}.exe", p))
        } else {
            pb
        }
    };
    let mut out: Vec<PathBuf> = Vec::new();
    if let Some(e) = explicit {
        out.push(e.to_path_buf());
    }
    if let Ok(v) = std::env::var("PTG_LLAMA_SERVER") {
        if !v.is_empty() {
            out.push(PathBuf::from(v));
        }
    }
    if let Ok(v) = std::env::var("LLAMA_SERVER") {
        if !v.is_empty() {
            out.push(PathBuf::from(v));
        }
    }
    out.push(home_dir().join(".cache").join("ptg").join("bin").join(exe));
    out.push(with_suffix(
        home_dir()
            .join("llama-spike")
            .join("llama.cpp")
            .join("build")
            .join("bin")
            .join("llama-server")
            .to_str()
            .unwrap_or(""),
    ));
    out
}

/// True if `path` exists and (on unix) is executable by someone.
fn is_runnable(path: &Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        match std::fs::metadata(path) {
            Ok(meta) => meta.is_file() && (meta.permissions().mode() & 0o111 != 0),
            Err(_) => false,
        }
    }
    #[cfg(not(unix))]
    {
        path.is_file()
    }
}

/// Resolve the first runnable `llama-server` candidate, or `None`.
pub fn detect_server(explicit: Option<&Path>) -> Option<PathBuf> {
    server_candidates(explicit)
        .into_iter()
        .find(|p| !p.as_os_str().is_empty() && is_runnable(p))
}

// --- Model download ---------------------------------------------------------

/// Download `repo/file` into `dest_dir`. Prefers the `hf`/`huggingface-cli`
/// tool if present (handles auth + gating cleanly); falls back to a native
/// resumable reqwest download using a bearer token from `hf_token_env`.
pub async fn download_model(
    repo: &str,
    file: &str,
    dest_dir: &Path,
    hf_token_env: &str,
    dry_run: bool,
) -> Result<PathBuf, SetupError> {
    let dest = dest_dir.join(file);
    if dry_run {
        return Ok(dest);
    }
    std::fs::create_dir_all(dest_dir)?;
    // Prefer the official HF CLIs: they handle Gemma gating/auth best. If a CLI
    // is present but broken (e.g. a stale shim) or errors, fall through to the
    // native download rather than hard-failing.
    for tool in ["hf", "huggingface-cli"] {
        if let Ok(which) = which_async(tool).await {
            match run_hf_cli(&which, repo, file, dest_dir).await {
                Ok(()) if dest.exists() => return Ok(dest),
                Ok(()) => { /* CLI reported success but file not at dest */ }
                Err(e) => {
                    eprintln!("  ({tool} CLI unavailable: {e}); trying native download");
                }
            }
        }
    }
    // Fallback: native resumable download with an optional bearer token.
    let token = std::env::var(hf_token_env).ok().filter(|t| !t.is_empty());
    native_download(repo, file, &dest, token.as_deref()).await?;
    Ok(dest)
}

async fn which_async(tool: &str) -> Result<PathBuf, ()> {
    // `which` crate-free lookup: probe PATH directly.
    let path_var = std::env::var_os("PATH").ok_or(())?;
    let exe = if cfg!(windows) {
        format!("{}.exe", tool)
    } else {
        tool.to_string()
    };
    for dir in std::env::split_paths(&path_var) {
        let candidate = dir.join(&exe);
        if is_runnable(&candidate) {
            return Ok(candidate);
        }
    }
    Err(())
}

async fn run_hf_cli(
    tool: &Path,
    repo: &str,
    file: &str,
    dest_dir: &Path,
) -> Result<(), SetupError> {
    let mut cmd = std::process::Command::new(tool);
    cmd.arg("download")
        .arg(repo)
        .arg(file)
        .arg("--local-dir")
        .arg(dest_dir);
    let status = cmd
        .status()
        .map_err(|e| SetupError::Message(format!("failed to run {tool:?}: {e}")))?;
    if !status.success() {
        return Err(SetupError::Message(format!(
            "`{tool:?} download` exited with {status}. If the model is gated, \
             accept the license at https://huggingface.co/{repo} and run \
             `{tool:?} login` (or set HF_TOKEN).",
        )));
    }
    Ok(())
}

/// Native resumable download of a HuggingFace file via `resolve/main`.
async fn native_download(
    repo: &str,
    file: &str,
    dest: &Path,
    token: Option<&str>,
) -> Result<(), SetupError> {
    let url = format!("https://huggingface.co/{repo}/resolve/main/{file}");
    let client = reqwest::Client::builder().build()?;
    let part = dest.with_extension("gguf.part");
    let mut have_bytes = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);

    let mut attempt = 0u32;
    loop {
        attempt += 1;
        let mut req = client
            .get(&url)
            .header("Range", format!("bytes={have_bytes}-"));
        if let Some(t) = token {
            req = req.bearer_auth(t);
        }
        let resp = req.send().await?;
        let status = resp.status();

        if status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN {
            return Err(SetupError::Message(format!(
                "HuggingFace returned {status} for {url}.\n\
                 Gemma is gated: accept the license at \
                 https://huggingface.co/{repo}, then authenticate — either run \
                 `hf login` or set HF_TOKEN (env var name configurable via \
                 --hf-token-env).",
            )));
        }
        // If the server ignored the Range (200 instead of 206), restart.
        let resumed = status == reqwest::StatusCode::PARTIAL_CONTENT;
        if status == reqwest::StatusCode::OK && have_bytes > 0 {
            have_bytes = 0;
            let _ = std::fs::remove_file(&part);
        } else if !status.is_success() {
            return Err(SetupError::Message(format!(
                "download failed: HTTP {status} for {url}",
            )));
        }

        use tokio::io::AsyncWriteExt;
        let mut f = if resumed {
            tokio::fs::OpenOptions::new()
                .append(true)
                .create(true)
                .open(&part)
                .await?
        } else {
            tokio::fs::File::create(&part).await?
        };
        let total = resp.content_length();
        let mut written = 0u64;
        let mut last_report = std::time::Instant::now();
        // Stream the body in chunks; never hold the whole file in memory.
        let mut resp = resp;
        while let Some(chunk) = resp.chunk().await? {
            f.write_all(chunk.as_ref()).await?;
            written += chunk.len() as u64;
            if last_report.elapsed() >= std::time::Duration::from_secs(3) {
                report_progress(have_bytes + written, total);
                last_report = std::time::Instant::now();
            }
        }
        f.flush().await?;
        drop(f);
        // Verify the file landed whole: if we got a Content-Length, compare.
        let final_len = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
        if let Some(expected) = total {
            let expected_total = if resumed {
                have_bytes + expected
            } else {
                expected
            };
            if final_len < expected_total {
                // Incomplete; resume on the next loop iteration.
                have_bytes = final_len;
                if attempt >= 5 {
                    return Err(SetupError::Message(format!(
                        "download stalled at {final_len}/{expected_total} bytes after \
                         {attempt} attempts",
                    )));
                }
                eprintln!("download incomplete ({final_len}/{expected_total} bytes); resuming...");
                continue;
            }
        }
        std::fs::rename(&part, dest)?;
        return Ok(());
    }
}

fn report_progress(have: u64, total: Option<u64>) {
    let mib = |b: u64| b as f64 / (1024.0 * 1024.0);
    match total {
        Some(t) => eprintln!(
            "  downloaded {:.1} / {:.1} MiB ({:.0}%)",
            mib(have),
            mib(t),
            (have as f64 / t.max(1) as f64) * 100.0
        ),
        None => eprintln!("  downloaded {:.1} MiB", mib(have)),
    }
}

// --- Phase entry points -----------------------------------------------------

/// Run the `ptg setup` phase.
pub async fn run_setup(args: SetupArgs) -> Result<ExitCode, SetupError> {
    let gguf_dir = args.gguf_dir.clone().unwrap_or_else(default_gguf_dir);
    let gguf_path = gguf_dir.join(&args.gguf_file);

    println!("PTG setup");
    println!("  config : {}", config_path().display());
    println!("  gguf   : {}", gguf_path.display());
    println!(
        "  server : {}:{} (alias `{}`)",
        args.host, args.port, args.model_alias
    );

    // 1. Server detection.
    let server = detect_server(args.llama_server.as_deref());
    match &server {
        Some(p) => println!("  llama-server: FOUND at {}", p.display()),
        None => {
            println!("  llama-server: NOT FOUND");
            if !args.dry_run {
                print_server_install_hint();
            }
        }
    }

    // 2. Model presence.
    let need_download = args.force || !gguf_path.exists();
    if !need_download {
        println!("  model: present");
    } else if args.dry_run {
        println!("  model: would download from {}", args.hf_repo);
    } else if !args.yes {
        // Interactive gate before a multi-GB download.
        let size_hint = "~2.7 GiB (QAT)";
        eprintln!(
            "\nModel `{}/{}` ({size_hint}) is missing. Download it now? [y/N] ",
            args.hf_repo, args.gguf_file
        );
        std::io::stderr().flush().ok();
        let mut line = String::new();
        if std::io::stdin().read_line(&mut line).is_err() {
            return Err(SetupError::Message(
                "could not read confirmation from stdin (pass --yes to skip)".into(),
            ));
        }
        if !line.trim().eq_ignore_ascii_case("y") {
            eprintln!("declined; no config written. Re-run `ptg setup` when ready.");
            return Ok(ExitCode::SUCCESS);
        }
    }

    // 3. Download.
    if need_download && !args.dry_run {
        println!("downloading model...");
        download_model(
            &args.hf_repo,
            &args.gguf_file,
            &gguf_dir,
            &args.hf_token_env,
            false,
        )
        .await?;
        println!("  model: downloaded to {}", gguf_path.display());
    }

    // 4. Write config.
    if args.dry_run {
        println!("\n(dry-run; no config written)");
        return Ok(ExitCode::SUCCESS);
    }
    let cfg = PtgConfig {
        server_path: server.as_ref().map(|p| p.to_string_lossy().into_owned()),
        gguf_path: gguf_path.to_string_lossy().into_owned(),
        host: args.host.clone(),
        port: args.port,
        model_alias: args.model_alias.clone(),
    };
    save_config(&cfg)?;
    println!("\nsetup complete. Next:");
    println!("  ptg serve                                              # start the server");
    println!("  ptg --probe                                            # verify it");
    if server.is_none() {
        println!("\n(note: `ptg serve` will not work until llama-server is installed; see above.)");
    }
    Ok(ExitCode::SUCCESS)
}

/// Run the `ptg serve` phase.
pub async fn run_serve(args: ServeArgs) -> Result<ExitCode, SetupError> {
    let cfg = load_config()
        .ok_or_else(|| SetupError::Message("no PTG config found. Run `ptg setup` first.".into()))?;
    let host = args.host.unwrap_or_else(|| cfg.host.clone());
    let port = args.port.unwrap_or(cfg.port);
    let alias = args.model_alias.unwrap_or_else(|| cfg.model_alias.clone());

    let server = cfg.server_path.as_deref().map(PathBuf::from);
    let server = detect_server(server.as_deref()).ok_or_else(|| {
        SetupError::Message(
            "llama-server not found. Install it (see `ptg setup` output) or set \
             PTG_LLAMA_SERVER, then re-run `ptg setup`."
                .into(),
        )
    })?;

    // Already running?
    if let Ok(resp) = reqwest::get(format!("http://{host}:{port}/v1/models")).await {
        if resp.status().is_success() {
            println!("server already running at http://{host}:{port} (alias `{alias}`)");
            println!("next: ptg --probe --vllm-url http://{host}:{port} --model {alias}");
            return Ok(ExitCode::SUCCESS);
        }
    }

    let launch = build_server_command(&server, &cfg.gguf_path, &host, port, &alias);
    if args.dry_run {
        println!(
            "{} {}",
            server.display(),
            launch
                .iter()
                .map(|s| shell_quote(s))
                .collect::<Vec<_>>()
                .join(" ")
        );
        return Ok(ExitCode::SUCCESS);
    }

    println!("launching {} ...", server.display());
    println!("  model : {}", cfg.gguf_path);
    println!("  bind  : {host}:{port} (alias `{alias}`)");
    println!("  (Ctrl-C to stop)\n");

    let mut cmd = std::process::Command::new(&server);
    cmd.args(&launch);
    let status = cmd
        .status()
        .map_err(|e| SetupError::Message(format!("failed to launch server: {e}")))?;
    if status.success() {
        Ok(ExitCode::SUCCESS)
    } else {
        Ok(ExitCode::FAILURE)
    }
}

/// The exact server flags, mirrored from `scripts/start-gemma4-qat.sh`.
pub fn build_server_command(
    _server: &Path,
    gguf: &str,
    host: &str,
    port: u16,
    alias: &str,
) -> Vec<String> {
    vec![
        "-m".into(),
        gguf.into(),
        "--host".into(),
        host.into(),
        "--port".into(),
        port.to_string(),
        "-c".into(),
        "4096".into(),
        "-ngl".into(),
        "99".into(),
        "-fa".into(),
        "on".into(),
        "--jinja".into(),
        "--alias".into(),
        alias.into(),
        "--reasoning".into(),
        "off".into(),
        "--reasoning-format".into(),
        "none".into(),
    ]
}

fn shell_quote(s: &str) -> String {
    if s.chars()
        .all(|c| c.is_ascii_alphanumeric() || "-_=./:".contains(c))
    {
        s.into()
    } else {
        format!("'{}'", s.replace('\'', "'\\''"))
    }
}

fn print_server_install_hint() {
    eprintln!("\nllama-server is required to run meshes but was not found.");
    eprintln!("Install it (one of):");
    eprintln!("  • Build from source:  https://github.com/ggml-org/llama.cpp");
    eprintln!("      cmake -B build && cmake --build build --config Release");
    eprintln!("  • Then point PTG at it:  export PTG_LLAMA_SERVER=/path/to/llama-server");
    eprintln!("  • (or place it at ~/.cache/ptg/bin/llama-server)");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn server_candidates_includes_explicit_first() {
        let explicit = PathBuf::from("/custom/llama-server");
        let cands = server_candidates(Some(&explicit));
        assert_eq!(cands.first(), Some(&explicit));
    }

    #[test]
    fn server_candidates_includes_env_vars_when_set() {
        std::env::set_var("PTG_LLAMA_SERVER", "/env/ptg/llama-server");
        let cands = server_candidates(None);
        assert!(cands
            .iter()
            .any(|p| p == &PathBuf::from("/env/ptg/llama-server")));
        std::env::remove_var("PTG_LLAMA_SERVER");
    }

    #[test]
    fn config_round_trips() -> Result<(), Box<dyn std::error::Error>> {
        let cfg = PtgConfig {
            server_path: Some("/x/llama-server".into()),
            gguf_path: "/y/model.gguf".into(),
            host: "127.0.0.1".into(),
            port: 18136,
            model_alias: "gemma-4-e2b-qat".into(),
        };
        let s = toml::to_string(&cfg)?;
        let back: PtgConfig = toml::from_str(&s)?;
        assert_eq!(back.server_url(), "http://127.0.0.1:18136");
        assert_eq!(back.model_alias, "gemma-4-e2b-qat");
        Ok(())
    }

    #[test]
    fn server_command_has_required_flags() {
        let args = build_server_command(
            Path::new("/x/llama-server"),
            "/m/gguf",
            "127.0.0.1",
            18136,
            "alias",
        );
        let joined = args.join(" ");
        for required in [
            "-m /m/gguf",
            "--host 127.0.0.1",
            "--port 18136",
            "-c 4096",
            "-ngl 99",
            "-fa on",
            "--jinja",
            "--alias alias",
            "--reasoning off",
            "--reasoning-format none",
        ] {
            assert!(
                joined.contains(required),
                "server command missing `{required}`: got {joined}"
            );
        }
    }

    #[test]
    fn shell_quote_passes_simple_args() {
        assert_eq!(shell_quote("4096"), "4096");
        assert_eq!(shell_quote("/path/to gguf"), "'/path/to gguf'");
    }

    #[test]
    fn config_path_respects_ptg_config_env() {
        std::env::set_var("PTG_CONFIG", "/tmp/ptg-test-config.toml");
        assert_eq!(config_path(), PathBuf::from("/tmp/ptg-test-config.toml"));
        std::env::remove_var("PTG_CONFIG");
    }

    #[test]
    fn default_gguf_dir_is_under_ptg() {
        let d = default_gguf_dir();
        let s = d.to_string_lossy();
        assert!(s.contains("ptg"), "gguf dir should be under ptg: {s}");
        assert!(s.contains("gguf"), "gguf dir should end in gguf: {s}");
    }
}
