//! Shared local inference engine — the PTG "thalamus" (§3.1.1, §7.1, §8.3).
//!
//! One engine instance is shared (behind an `Arc`) by every virtual column,
//! reflecting the single-engine topology of the specification: a local vLLM
//! server multiplexes the model across columns via dynamic system prompts and
//! prefix caching instead of loading one model per column.
//!
//! The [`ColumnEngine`] trait decouples the mesh runtime from the HTTP backend,
//! so the orchestration loop is unit-testable with a mock and a `vLLM` backend,
//! `mistral.rs`, or any other server can be swapped in without touching the
//! runtime.

use std::time::Duration;

use async_trait::async_trait;
use ptg_core::{ColumnOutputSchema, CorticalColumn, SchemaError, Stimulus, StimulusPart};
use serde::{Deserialize, Serialize};

/// Errors surfaced by the inference engine layer.
#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    /// The underlying HTTP request to the engine failed.
    #[error("HTTP request to inference engine failed: {0}")]
    Http(#[from] reqwest::Error),
    /// The engine response body was not valid JSON.
    #[error("engine response was not valid JSON: {0}")]
    Json(#[from] serde_json::Error),
    /// The engine response omitted the expected `choices[0].message.content`.
    #[error("engine response missing message content")]
    MissingContent,
    /// The structured output failed schema validation.
    #[error("structured output failed validation: {0}")]
    Validation(#[from] SchemaError),
}

/// Contract for a column's inference backend.
///
/// Implementations must be cheaply cloneable behind an `Arc` and safe to share
/// across many concurrent column ticks.
#[async_trait]
pub trait ColumnEngine: Send + Sync {
    /// Run one column tick: combine the column's system prompt with the
    /// stimulus and injected lateral context, and return the structured output.
    ///
    /// # Errors
    /// - [`EngineError::Http`] if the request to the engine fails.
    /// - [`EngineError::Json`] if the response or content is malformed.
    /// - [`EngineError::MissingContent`] if no message content is returned.
    /// - [`EngineError::Validation`] if the structured output fails common or
    ///   per-sphere schema validation.
    async fn execute_column_tick(
        &self,
        column: &CorticalColumn,
        stimulus: &Stimulus,
        lateral_context: &str,
    ) -> Result<ColumnOutputSchema, EngineError>;
}

/// Probe a server's `/v1/models` endpoint.
///
/// # Errors
/// [`EngineError::Http`] if the request fails or the server is unreachable.
/// [`EngineError::Json`] if the response body is not a valid models listing.
pub async fn list_models(
    client: &reqwest::Client,
    base_url: &str,
) -> Result<Vec<String>, EngineError> {
    let endpoint = format!("{base_url}/v1/models");
    let response = client.get(endpoint).send().await?.error_for_status()?;
    let body: serde_json::Value = response.json().await?;
    parse_models_body(&body)
}

/// Parse an OpenAI-compatible `{ "data": [{ "id": "..." }, ...] }` models body.
/// Separated from [`list_models`] so it can be unit-tested without a server.
///
/// # Errors
/// [`EngineError::Json`] if the body does not deserialize into [`ModelsResponse`].
fn parse_models_body(body: &serde_json::Value) -> Result<Vec<String>, EngineError> {
    let parsed: ModelsResponse = serde_json::from_value(body.clone())?;
    Ok(parsed.data.into_iter().map(|m| m.id).collect())
}

#[derive(Debug, Deserialize)]
struct ModelsResponse {
    data: Vec<ModelEntry>,
}

#[derive(Debug, Deserialize)]
struct ModelEntry {
    id: String,
}

// ---------------------------------------------------------------------------
// OpenAI-style chat completion wire types (§8.3, multimodal §2.3)
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<ChatRequestMessage>,
    temperature: f32,
    max_tokens: u32,
    response_format: ResponseFormat,
}

/// A message's `content`: a plain string for text, or a typed-part array for
/// multimodal inputs (OpenAI chat-completions content schema).
#[derive(Debug, Serialize)]
#[serde(untagged)]
enum ChatContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

/// One element of a multimodal content array. The text part is always first; the
/// image/audio parts are forwarded from the [`Stimulus`] verbatim (they already
/// serialize to the OpenAI `type`/`image_url`/`input_audio` shapes via
/// `ptg_core::StimulusPart`).
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ContentPart {
    Text {
        text: String,
    },
    ImageUrl {
        image_url: ptg_core::ImageUrlRef,
    },
    InputAudio {
        input_audio: ptg_core::InputAudioRef,
    },
}

impl ContentPart {
    /// Forward a `ptg_core::StimulusPart` into the wire `ContentPart`, preserving
    /// the OpenAI shapes.
    #[must_use]
    fn from_stimulus(part: &StimulusPart) -> Self {
        match part {
            StimulusPart::ImageUrl { image_url } => Self::ImageUrl {
                image_url: image_url.clone(),
            },
            StimulusPart::InputAudio { input_audio } => Self::InputAudio {
                input_audio: input_audio.clone(),
            },
        }
    }
}

#[derive(Debug, Serialize)]
struct ChatRequestMessage {
    role: &'static str,
    content: ChatContent,
}

#[derive(Debug, Serialize)]
struct ResponseFormat {
    #[serde(rename = "type")]
    kind: &'static str,
}

impl ResponseFormat {
    const JSON_OBJECT: Self = Self {
        kind: "json_object",
    };
}

#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatMessage,
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    content: Option<String>,
}

// ---------------------------------------------------------------------------
// Concrete vLLM client
// ---------------------------------------------------------------------------

/// Concrete client for a local OpenAI-compatible (vLLM) inference server.
///
/// Construction is infallible from the caller's perspective only in the sense
/// that it returns a `Result`; the underlying HTTP client build (connection
/// pool, timeouts) is fallible and never panics.
pub struct InferenceEngine {
    client: reqwest::Client,
    vllm_url: String,
    model: String,
    temperature: f32,
    max_tokens: u32,
}

impl InferenceEngine {
    /// Create a new engine client targeting the given vLLM base URL and model.
    ///
    /// Uses a generously pooled HTTP client (§7.3: connection-pooled reqwest
    /// multiplexing many parallel column updates).
    ///
    /// # Errors
    /// [`EngineError::Http`] if the HTTP client cannot be constructed.
    pub fn new(url: &str, model: &str) -> Result<Self, EngineError> {
        Self::builder(url, model).build()
    }

    /// Start configuring a new engine client.
    #[must_use]
    pub fn builder(url: &str, model: &str) -> EngineBuilder {
        EngineBuilder {
            url: url.to_string(),
            model: model.to_string(),
            temperature: 0.2,
            max_tokens: 512,
            pool_max_idle_per_host: 500,
            timeout: Duration::from_secs(120),
        }
    }

    /// Borrow the shared HTTP client (e.g. for `--probe` / `list_models`).
    #[must_use]
    pub fn http_client(&self) -> &reqwest::Client {
        &self.client
    }

    /// The configured engine base URL.
    #[must_use]
    pub fn vllm_url(&self) -> &str {
        &self.vllm_url
    }

    /// Build the composite prompt exactly as specified in §8.3.
    fn compose_prompt(
        column: &CorticalColumn,
        stimulus: &Stimulus,
        lateral_context: &str,
    ) -> String {
        format!(
            "{system}\n\nINPUT DATA TO PARSE:\n{input}\n\nLATERAL CONNECTIONS (NEIGHBOR SUGGESTIONS):\n{lateral}",
            system = column.system_prompt,
            input = stimulus.text_str(),
            lateral = lateral_context,
        )
    }

    /// Parse the engine's returned content string into a schema that passes the
    /// common (cross-sphere) validation. Sphere-specific validation is performed
    /// by the caller via [`ColumnOutputSchema::validate_for_sphere`].
    ///
    /// Exposed for unit testing without a live server. Tolerates minor
    /// conversational filler by extracting the outermost JSON object.
    fn parse_output(content: &str) -> Result<ColumnOutputSchema, EngineError> {
        let payload = extract_json_block(content);
        let schema: ColumnOutputSchema = serde_json::from_str(payload)?;
        schema.validate()?;
        Ok(schema)
    }

    /// Build the chat message content for a stimulus: a plain string for text,
    /// or a typed-part array for multimodal inputs (text part first).
    fn build_content(full_prompt: String, stimulus: &Stimulus) -> ChatContent {
        match stimulus {
            Stimulus::Text(_) => ChatContent::Text(full_prompt),
            Stimulus::Multimodal { parts, .. } => {
                let mut content_parts = Vec::with_capacity(parts.len() + 1);
                content_parts.push(ContentPart::Text { text: full_prompt });
                for p in parts {
                    content_parts.push(ContentPart::from_stimulus(p));
                }
                ChatContent::Parts(content_parts)
            }
        }
    }
}

/// Builder for [`InferenceEngine`].
pub struct EngineBuilder {
    url: String,
    model: String,
    temperature: f32,
    max_tokens: u32,
    pool_max_idle_per_host: usize,
    timeout: Duration,
}

impl EngineBuilder {
    /// Set the sampling temperature (default `0.2`).
    #[must_use]
    pub const fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set the max output tokens (default `400`).
    #[must_use]
    pub const fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Set the per-host idle connection pool size (default `500`).
    #[must_use]
    pub const fn pool_max_idle_per_host(mut self, size: usize) -> Self {
        self.pool_max_idle_per_host = size;
        self
    }

    /// Set the request timeout (default 120s).
    #[must_use]
    pub const fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Finalize the engine.
    ///
    /// # Errors
    /// [`EngineError::Http`] if the HTTP client cannot be constructed.
    pub fn build(self) -> Result<InferenceEngine, EngineError> {
        let client = reqwest::Client::builder()
            .pool_max_idle_per_host(self.pool_max_idle_per_host)
            .timeout(self.timeout)
            .build()?;
        Ok(InferenceEngine {
            client,
            vllm_url: self.url,
            model: self.model,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
        })
    }
}

#[async_trait]
impl ColumnEngine for InferenceEngine {
    async fn execute_column_tick(
        &self,
        column: &CorticalColumn,
        stimulus: &Stimulus,
        lateral_context: &str,
    ) -> Result<ColumnOutputSchema, EngineError> {
        let prompt = Self::compose_prompt(column, stimulus, lateral_context);
        let content = Self::build_content(prompt, stimulus);
        let request = ChatRequest {
            model: &self.model,
            messages: vec![ChatRequestMessage {
                role: "user",
                content,
            }],
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            response_format: ResponseFormat::JSON_OBJECT,
        };

        let endpoint = format!("{}/v1/chat/completions", self.vllm_url);
        let response = self
            .client
            .post(endpoint)
            .json(&request)
            .send()
            .await?
            .error_for_status()?;
        let parsed: ChatCompletionResponse = response.json().await?;

        let raw = parsed
            .choices
            .into_iter()
            .next()
            .and_then(|c| c.message.content)
            .ok_or(EngineError::MissingContent)?;

        let output = Self::parse_output(&raw)?;
        // Enforce the column's sphere-specific schema (Item B). A small model
        // that omits a required field surfaces as EngineError::Validation.
        output.validate_for_sphere(column.sphere)?;
        Ok(output)
    }
}

/// Return the outermost `{ ... }` block of `content`, falling back to the whole
/// string when no balanced block is found. This lets the parser tolerate minor
/// conversational filler despite the prompts forbidding it.
fn extract_json_block(content: &str) -> &str {
    let start = content.find('{');
    let end = content.rfind('}');
    match (start, end) {
        (Some(s), Some(e)) if s <= e => &content[s..=e],
        _ => content,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ptg_core::DomainSphere;

    #[test]
    fn builder_constructs_client() -> Result<(), Box<dyn std::error::Error>> {
        let engine = InferenceEngine::new("http://localhost:8000", "test/model")?;
        assert_eq!(engine.model, "test/model");
        assert_eq!(engine.vllm_url, "http://localhost:8000");
        Ok(())
    }

    #[test]
    fn compose_prompt_contains_sections() {
        let col = CorticalColumn::with_defaults("CC_X", DomainSphere::Physics);
        let prompt = InferenceEngine::compose_prompt(&col, &Stimulus::text("INPUT"), "LATERAL");
        assert!(prompt.contains("INPUT DATA TO PARSE:"));
        assert!(prompt.contains("LATERAL CONNECTIONS"));
        assert!(prompt.contains("INPUT"));
        assert!(prompt.contains("LATERAL"));
    }

    #[test]
    fn parse_output_handles_filler() -> Result<(), Box<dyn std::error::Error>> {
        let raw = "Here is the result:\n{\"prediction\":\"p\",\"reference_frame_coordinates\":\"c\",\"confidence\":0.5}\nDone.";
        let out = InferenceEngine::parse_output(raw)?;
        assert_eq!(out.prediction, "p");
        assert!((out.confidence - 0.5).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn parse_output_rejects_bad_confidence() {
        let raw = "{\"prediction\":\"p\",\"reference_frame_coordinates\":\"c\",\"confidence\":2.5}";
        assert!(InferenceEngine::parse_output(raw).is_err());
    }

    #[test]
    fn build_content_text_uses_plain_string() -> Result<(), Box<dyn std::error::Error>> {
        let content = InferenceEngine::build_content("hello".into(), &Stimulus::text("hi"));
        let json = serde_json::to_string(&content)?;
        assert_eq!(json, "\"hello\"");
        Ok(())
    }

    #[test]
    fn build_content_multimodal_uses_part_array() -> Result<(), Box<dyn std::error::Error>> {
        use ptg_core::{AudioFormat, ImageDetail, ImageUrlRef, InputAudioRef};
        let stimulus = Stimulus::Multimodal {
            text: "caption".into(),
            parts: vec![
                StimulusPart::ImageUrl {
                    image_url: ImageUrlRef {
                        url: "data:image/png;base64,AAAA".into(),
                        detail: ImageDetail::High,
                    },
                },
                StimulusPart::InputAudio {
                    input_audio: InputAudioRef {
                        // OpenAI `input_audio.data` is raw base64; the field is
                        // forwarded verbatim, so callers must NOT use a data URI.
                        data: "Bg==".into(),
                        format: AudioFormat::Wav,
                    },
                },
            ],
        };
        let content = InferenceEngine::build_content("full prompt".into(), &stimulus);
        let json = serde_json::to_string(&content)?;
        assert!(
            json.starts_with('['),
            "multimodal content must be an array: {json}"
        );
        assert!(
            json.contains(r#""type":"text","text":"full prompt""#),
            "text part missing: {json}"
        );
        assert!(
            json.contains(r#""type":"image_url""#),
            "image part missing: {json}"
        );
        assert!(
            json.contains(r#""type":"input_audio""#),
            "audio part missing: {json}"
        );
        Ok(())
    }

    #[test]
    fn extract_json_block_handles_nested() {
        let content = r#"noise {"a": {"b": 1}, "c": 2} tail"#;
        assert_eq!(extract_json_block(content), r#"{"a": {"b": 1}, "c": 2}"#);
    }

    #[test]
    fn parse_models_body_extracts_ids() -> Result<(), Box<dyn std::error::Error>> {
        let body: serde_json::Value =
            serde_json::from_str(r#"{ "data": [{ "id": "gemma-4-e4b" }, { "id": "other" }] }"#)?;
        let ids = parse_models_body(&body)?;
        assert_eq!(ids, vec!["gemma-4-e4b".to_string(), "other".to_string()]);
        Ok(())
    }

    #[test]
    fn parse_models_body_rejects_bad_shape() {
        // missing 'data' field
        assert!(parse_models_body(&serde_json::json!({ "oops": [] })).is_err());
        // 'data' present but an entry lacks required 'id'
        assert!(parse_models_body(&serde_json::json!({ "data": [{ "not_id": "x" }] })).is_err());
        // well-formed empty listing is fine
        assert!(parse_models_body(&serde_json::json!({ "data": [] })).is_ok_and(|v| v.is_empty()));
    }
}
