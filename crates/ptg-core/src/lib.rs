//! Core domain types for Project Thousand-Gemma (PTG).
//!
//! Models the virtual cortical column (§3.1.2), the structured output schema
//! every column emits (§5), the domain spheres and their immutable system
//! prompts (§5), and the fixed-capacity history buffer each column maintains
//! to respect its strict per-column memory budget (§3.1.2 / §7.2).

use std::collections::{BTreeMap, VecDeque};

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// System prompts (§5 "Column Prompt Blueprint")
// ---------------------------------------------------------------------------

/// Physics reference-frame prompt (§5).
pub const PROMPT_PHYSICS: &str = r#"
ROLE: Cortical Column - Primary Physics Sensor.
CONTEXT COMPARTMENTALIZATION: You parse all incoming inputs strictly through the laws of classical mechanics, thermodynamics, kinetics, electromagnetism, and quantum principles. Ignore emotional intent, language syntax, or historical origin.
REFERENCE FRAME: Map the input data to a spatial reference frame consisting of forces (vectors), energy fields (Joules), masses (kg), and thermodynamic gradients.
OUTPUT FORMAT: You must output a structured JSON schema conforming exactly to:
{
  "reference_frame_coordinates": "x,y,z spatial/conceptual bounds",
  "isolated_variables": ["var1", "var2"],
  "empirical_observation": "Brief summary of input through physical laws",
  "prediction": "What the system will do next based on physical mechanics",
  "confidence": 0.00
}
Do not include any conversational filler outside the JSON block.
"#;

/// Mathematics reference-frame prompt (§5).
pub const PROMPT_MATHEMATICS: &str = r#"
ROLE: Cortical Column - Quantitative Reasoning Engine.
CONTEXT COMPARTMENTALIZATION: You analyze inputs strictly for mathematical constants, geometric structures, algorithmic complexity, numerical relationships, and formal logic. Ignore material composition, time period, and human bias.
REFERENCE FRAME: Establish an algebraic, geometric, or statistical coordinate structure.
OUTPUT FORMAT: Output a structured JSON schema:
{
  "reference_frame_coordinates": "matrix or tensor spatial bounds",
  "axiomatic_assertions": ["assertion1", "assertion2"],
  "deductive_synthesis": "Brief formal proof or numerical analysis",
  "prediction": "Extrapolated quantitative trend line",
  "confidence": 0.00
}
"#;

/// Coding reference-frame prompt (§5).
pub const PROMPT_CODING: &str = r#"
ROLE: Cortical Column - Algorithmic Synthesis Unit.
CONTEXT COMPARTMENTALIZATION: Interpret incoming information as software systems, computational logic, control flows, state machines, data structures, and algorithmic transformations.
REFERENCE FRAME: Map data to a computational graph or state transition matrix.
OUTPUT FORMAT: Output a structured JSON schema:
{
  "reference_frame_coordinates": "state_machine_id / memory_offset",
  "state_variables": ["param1", "param2"],
  "algorithmic_analysis": "Logic evaluation, time complexity big-O, structure verification",
  "prediction": "Deterministic outcome of execution flow",
  "confidence": 0.00
}
"#;

/// Psychology reference-frame prompt (§5).
pub const PROMPT_PSYCHOLOGY: &str = r#"
ROLE: Cortical Column - Behavioral / Intention Analyzer.
CONTEXT COMPARTMENTALIZATION: Evaluate inputs purely for human psychological states, evolutionary drivers, cognitive biases, communicative intent, emotional dynamics, or behavioral patterns.
REFERENCE FRAME: Map data to a psychological profile or sociometric matrix.
OUTPUT FORMAT: Output a structured JSON schema:
{
  "reference_frame_coordinates": "emotional_valence / behavioral_vector",
  "cognitive_biases": ["bias1", "bias2"],
  "behavioral_synthesis": "Assessment of underlying motivation or intent",
  "prediction": "Expected behavioral choice or adaptation profile",
  "confidence": 0.00
}
"#;

// ---------------------------------------------------------------------------
// Domain spheres (§5, §8.1)
// ---------------------------------------------------------------------------

/// Domain specialization of a cortical column.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DomainSphere {
    Physics,
    Mathematics,
    Coding,
    Psychology,
}

impl DomainSphere {
    /// Human-readable sphere name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Physics => "Physics",
            Self::Mathematics => "Mathematics",
            Self::Coding => "Coding",
            Self::Psychology => "Psychology",
        }
    }

    /// The immutable base system prompt that restricts this column to its
    /// reference frame (§5).
    #[must_use]
    pub const fn default_prompt(self) -> &'static str {
        match self {
            Self::Physics => PROMPT_PHYSICS,
            Self::Mathematics => PROMPT_MATHEMATICS,
            Self::Coding => PROMPT_CODING,
            Self::Psychology => PROMPT_PSYCHOLOGY,
        }
    }

    /// The domain-specific keys this sphere's output must contain (§5 prompts).
    /// Used by [`ColumnOutputSchema::validate_for_sphere`].
    #[must_use]
    pub const fn required_fields(self) -> &'static [&'static str] {
        match self {
            Self::Physics => &["isolated_variables", "empirical_observation"],
            Self::Mathematics => &["axiomatic_assertions", "deductive_synthesis"],
            Self::Coding => &["state_variables", "algorithmic_analysis"],
            Self::Psychology => &["cognitive_biases", "behavioral_synthesis"],
        }
    }
}

// ---------------------------------------------------------------------------
// History buffer (§3.1.2 history_buffer, §7.2 memory budget)
// ---------------------------------------------------------------------------

/// One tick of a column's recorded history.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HistoryEntry {
    pub tick: u32,
    pub prediction: String,
    pub confidence: f32,
}

/// Fixed-capacity ring buffer retaining the last `capacity` ticks of column
/// state, enforcing the strict per-column memory budget described in §7.2.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryBuffer {
    capacity: usize,
    entries: VecDeque<HistoryEntry>,
}

impl HistoryBuffer {
    /// Create a new ring buffer that retains at most `capacity` entries.
    ///
    /// A `capacity` of `0` yields a buffer that discards everything pushed.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            entries: VecDeque::with_capacity(capacity),
        }
    }

    /// Record a new tick, dropping the oldest entry when at capacity.
    pub fn push(&mut self, entry: HistoryEntry) {
        if self.capacity == 0 {
            return;
        }
        if self.entries.len() == self.capacity {
            self.entries.pop_front();
        }
        self.entries.push_back(entry);
    }

    /// Number of retained entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no entries are retained.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate over retained entries, oldest first.
    pub fn iter(&self) -> impl Iterator<Item = &HistoryEntry> {
        self.entries.iter()
    }
}

// ---------------------------------------------------------------------------
// Cortical column (§3.1.2, §8.1)
// ---------------------------------------------------------------------------

/// A single virtual cortical column: a thread-safe unit of cognition bound to
/// one domain sphere and reference frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorticalColumn {
    /// Unique identifier, e.g. `"CC_PHYSICS_01"`.
    pub id: String,
    /// Domain specialization.
    pub sphere: DomainSphere,
    /// Immutable base instruction set mapping inputs to a reference frame.
    pub system_prompt: String,
    /// Current conceptual/spatial coordinate in the problem space.
    pub current_coordinate: String,
    /// Internal prediction certainty in `[0.0, 1.0]`.
    pub last_confidence: f32,
    /// Most recent prediction string.
    pub last_prediction: String,
    /// Ring buffer of recent ticks (§3.1.2 / §7.2).
    pub history_buffer: HistoryBuffer,
}

impl CorticalColumn {
    /// Construct a new column with an explicit system prompt.
    #[must_use]
    pub fn new(id: &str, sphere: DomainSphere, prompt: &str) -> Self {
        Self {
            id: id.to_string(),
            sphere,
            system_prompt: prompt.to_string(),
            current_coordinate: String::from("0.0,0.0,0.0"),
            last_confidence: 0.0,
            last_prediction: String::new(),
            history_buffer: HistoryBuffer::new(64),
        }
    }

    /// Construct a new column using the default system prompt for its sphere.
    #[must_use]
    pub fn with_defaults(id: &str, sphere: DomainSphere) -> Self {
        Self::new(id, sphere, sphere.default_prompt())
    }

    /// Apply one tick's structured output to the column's running state.
    pub fn record_tick(&mut self, tick: u32, output: &ColumnOutputSchema) {
        self.last_prediction = output.prediction.clone();
        self.last_confidence = output.confidence;
        self.current_coordinate = output.reference_frame_coordinates.clone();
        self.history_buffer.push(HistoryEntry {
            tick,
            prediction: output.prediction.clone(),
            confidence: output.confidence,
        });
    }
}

// ---------------------------------------------------------------------------
// Structured output schema (§5)
// ---------------------------------------------------------------------------

/// Errors raised when a column's structured output fails validation.
#[derive(Debug, thiserror::Error)]
pub enum SchemaError {
    /// Confidence was NaN or infinite.
    #[error("confidence is not a finite number")]
    NonFiniteConfidence,
    /// Confidence fell outside `[0.0, 1.0]`.
    #[error("confidence {0} out of range [0.0, 1.0]")]
    ConfidenceOutOfRange(f32),
    /// A domain-specific field required by the column's sphere was absent.
    #[error("output missing required field for its sphere: {0}")]
    MissingRequiredField(String),
}

/// The structured JSON every column emits.
///
/// Only the three fields common to all domain prompts (`reference_frame_coordinates`,
/// `prediction`, `confidence`) are typed; the domain-specific fields (e.g.
/// `isolated_variables`, `axiomatic_assertions`, `algorithmic_analysis`,
/// `behavioral_synthesis`) are preserved verbatim in `domain_fields`. This makes
/// the schema parse all four spheres (§5), unlike a struct that hard-codes a
/// single sphere's field names.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnOutputSchema {
    pub reference_frame_coordinates: String,
    pub prediction: String,
    pub confidence: f32,
    /// Domain-specific fields emitted by the column's prompt, keyed by name.
    #[serde(default, flatten)]
    pub domain_fields: BTreeMap<String, serde_json::Value>,
}

impl ColumnOutputSchema {
    /// Validate the structured output: confidence must be finite and in range.
    ///
    /// # Errors
    /// - [`SchemaError::NonFiniteConfidence`] if `confidence` is NaN/infinite.
    /// - [`SchemaError::ConfidenceOutOfRange`] if `confidence` is outside
    ///   `[0.0, 1.0]`.
    pub fn validate(&self) -> Result<(), SchemaError> {
        if !self.confidence.is_finite() {
            return Err(SchemaError::NonFiniteConfidence);
        }
        if !(0.0..=1.0).contains(&self.confidence) {
            return Err(SchemaError::ConfidenceOutOfRange(self.confidence));
        }
        Ok(())
    }

    /// Validate the output against a sphere's required domain-specific fields.
    ///
    /// Runs [`validate`](Self::validate) first, then asserts the sphere's
    /// required keys are *present* in `domain_fields` (presence only; value
    /// types are not enforced, to avoid over-validating model output).
    ///
    /// # Errors
    /// - Any [`SchemaError`] from [`validate`](Self::validate).
    /// - [`SchemaError::MissingRequiredField`] for the first missing key.
    pub fn validate_for_sphere(&self, sphere: DomainSphere) -> Result<(), SchemaError> {
        self.validate()?;
        for required in sphere.required_fields() {
            if !self.domain_fields.contains_key(*required) {
                return Err(SchemaError::MissingRequiredField((*required).to_string()));
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Stimulus (§2.3 afferent pathway, multimodal extension)
// ---------------------------------------------------------------------------

/// Resolution hint for an image stimulus part (OpenAI `detail`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ImageDetail {
    Low,
    High,
    #[default]
    Auto,
}

/// Audio container format for an audio stimulus part (OpenAI `format`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AudioFormat {
    Wav,
    Mp3,
}

/// One non-text component of a multimodal stimulus.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StimulusPart {
    /// An image referenced by URL (including inline `data:image/...;base64,...`).
    ImageUrl { image_url: ImageUrlRef },
    /// Inline audio as base64-encoded data.
    InputAudio { input_audio: InputAudioRef },
}

/// The `image_url` object nested under an [`StimulusPart::ImageUrl`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageUrlRef {
    pub url: String,
    #[serde(default)]
    pub detail: ImageDetail,
}

/// The `input_audio` object nested under an [`StimulusPart::InputAudio`].
///
/// `data` is forwarded to the server verbatim; per the OpenAI schema it must be
/// raw base64 (NOT a `data:audio/...;base64,...` URI).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputAudioRef {
    pub data: String,
    pub format: AudioFormat,
}

/// The afferent stimulus broadcast to all columns during an epoch (§2.3, §6).
///
/// `Text` is the common case; `Multimodal` carries image/audio parts alongside
/// a text anchor. Both variants expose their text portion via [`Stimulus::text_str`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Stimulus {
    /// Text-only stimulus.
    Text(String),
    /// Text plus one or more non-text parts.
    Multimodal {
        text: String,
        parts: Vec<StimulusPart>,
    },
}

impl Stimulus {
    /// Construct a text-only stimulus.
    #[must_use]
    pub fn text(s: impl Into<String>) -> Self {
        Self::Text(s.into())
    }

    /// The text portion of the stimulus (the whole string for `Text`, the
    /// `text` field for `Multimodal`).
    #[must_use]
    pub fn text_str(&self) -> &str {
        match self {
            Self::Text(s) => s,
            Self::Multimodal { text, .. } => text,
        }
    }
}

// ---------------------------------------------------------------------------
// Default mesh wiring (§8.4)
// ---------------------------------------------------------------------------

/// The four default cortical columns from the reference wiring (§8.4).
#[must_use]
pub fn default_columns() -> Vec<CorticalColumn> {
    vec![
        CorticalColumn::with_defaults("CC_PHYSICS_01", DomainSphere::Physics),
        CorticalColumn::with_defaults("CC_MATH_01", DomainSphere::Mathematics),
        CorticalColumn::with_defaults("CC_CODE_01", DomainSphere::Coding),
        CorticalColumn::with_defaults("CC_PSYCH_01", DomainSphere::Psychology),
    ]
}

/// The default bidirectional lateral connections from the reference wiring (§8.4),
/// as `(from_id, to_id)` pairs.
#[must_use]
pub fn default_connections() -> Vec<(&'static str, &'static str)> {
    vec![
        ("CC_PHYSICS_01", "CC_MATH_01"),
        ("CC_MATH_01", "CC_PHYSICS_01"),
        ("CC_MATH_01", "CC_CODE_01"),
        ("CC_CODE_01", "CC_PSYCH_01"),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_prompt_matches_sphere() {
        assert_eq!(DomainSphere::Physics.default_prompt(), PROMPT_PHYSICS);
        assert_eq!(
            DomainSphere::Mathematics.default_prompt(),
            PROMPT_MATHEMATICS
        );
        assert_eq!(DomainSphere::Coding.default_prompt(), PROMPT_CODING);
        assert_eq!(DomainSphere::Psychology.default_prompt(), PROMPT_PSYCHOLOGY);
    }

    #[test]
    fn history_buffer_evicts_oldest_at_capacity() {
        let mut buf = HistoryBuffer::new(2);
        assert!(buf.is_empty());
        buf.push(HistoryEntry {
            tick: 1,
            prediction: "a".into(),
            confidence: 0.1,
        });
        buf.push(HistoryEntry {
            tick: 2,
            prediction: "b".into(),
            confidence: 0.2,
        });
        buf.push(HistoryEntry {
            tick: 3,
            prediction: "c".into(),
            confidence: 0.3,
        });
        assert_eq!(buf.len(), 2);
        assert_eq!(
            buf.iter().map(|e| e.tick).collect::<Vec<_>>(),
            vec![2, 3],
            "oldest entry should have been evicted"
        );
    }

    #[test]
    fn history_buffer_zero_capacity_discards() {
        let mut buf = HistoryBuffer::new(0);
        buf.push(HistoryEntry {
            tick: 1,
            prediction: "a".into(),
            confidence: 0.1,
        });
        assert!(buf.is_empty());
    }

    #[test]
    fn parses_physics_output_shape() -> Result<(), Box<dyn std::error::Error>> {
        let json = r#"{
            "reference_frame_coordinates": "4.0,12.0,-2.0",
            "isolated_variables": ["v_kinetic"],
            "empirical_observation": "Kinetic energy burst along vector",
            "prediction": "Energy dissipates thermally",
            "confidence": 0.82
        }"#;
        let out: ColumnOutputSchema = serde_json::from_str(json)?;
        out.validate()?;
        assert_eq!(out.prediction, "Energy dissipates thermally");
        assert!(out.domain_fields.contains_key("isolated_variables"));
        assert!(out.domain_fields.contains_key("empirical_observation"));
        Ok(())
    }

    #[test]
    fn parses_math_output_shape() -> Result<(), Box<dyn std::error::Error>> {
        // Mathematics emits `deductive_synthesis`, not `empirical_observation`.
        let json = r#"{
            "reference_frame_coordinates": "tensor[R3]",
            "axiomatic_assertions": ["|v|=sqrt(164)"],
            "deductive_synthesis": "Vector magnitude exceeds threshold",
            "prediction": "Magnitude stabilizes",
            "confidence": 0.77
        }"#;
        let out: ColumnOutputSchema = serde_json::from_str(json)?;
        out.validate()?;
        assert!(out.domain_fields.contains_key("deductive_synthesis"));
        assert!(out.domain_fields.contains_key("axiomatic_assertions"));
        Ok(())
    }

    #[test]
    fn parses_coding_and_psychology_shapes() -> Result<(), Box<dyn std::error::Error>> {
        let code = r#"{
            "reference_frame_coordinates": "state[INIT]",
            "state_variables": ["init_step"],
            "algorithmic_analysis": "Automation init failed at step 0",
            "prediction": "Retry succeeds",
            "confidence": 0.69
        }"#;
        let psych = r#"{
            "reference_frame_coordinates": "valence[-]",
            "cognitive_biases": ["automation_bias"],
            "behavioral_synthesis": "Operator stress from failure",
            "prediction": "Manual intervention",
            "confidence": 0.61
        }"#;
        let c: ColumnOutputSchema = serde_json::from_str(code)?;
        let p: ColumnOutputSchema = serde_json::from_str(psych)?;
        c.validate()?;
        p.validate()?;
        assert!(c.domain_fields.contains_key("algorithmic_analysis"));
        assert!(p.domain_fields.contains_key("behavioral_synthesis"));
        Ok(())
    }

    #[test]
    fn validate_rejects_out_of_range_confidence() {
        let out = ColumnOutputSchema {
            reference_frame_coordinates: "x".into(),
            prediction: "p".into(),
            confidence: 1.5,
            domain_fields: BTreeMap::new(),
        };
        assert!(out.validate().is_err());
    }

    #[test]
    fn record_tick_updates_state_and_history() {
        let mut col = CorticalColumn::with_defaults("CC_X", DomainSphere::Physics);
        let out = ColumnOutputSchema {
            reference_frame_coordinates: "1,2,3".into(),
            prediction: "boom".into(),
            confidence: 0.42,
            domain_fields: BTreeMap::new(),
        };
        col.record_tick(1, &out);
        assert_eq!(col.last_prediction, "boom");
        assert!((col.last_confidence - 0.42).abs() < 1e-6);
        assert_eq!(col.current_coordinate, "1,2,3");
        assert_eq!(col.history_buffer.len(), 1);
    }

    #[test]
    fn validate_for_sphere_accepts_complete_output() -> Result<(), Box<dyn std::error::Error>> {
        let json = r#"{
            "reference_frame_coordinates": "x",
            "isolated_variables": ["v"],
            "empirical_observation": "obs",
            "prediction": "p",
            "confidence": 0.5
        }"#;
        let out: ColumnOutputSchema = serde_json::from_str(json)?;
        assert!(out.validate_for_sphere(DomainSphere::Physics).is_ok());
        Ok(())
    }

    #[test]
    fn validate_for_sphere_rejects_missing_field() {
        let out = ColumnOutputSchema {
            reference_frame_coordinates: "x".into(),
            prediction: "p".into(),
            confidence: 0.5,
            domain_fields: BTreeMap::new(), // missing isolated_variables + empirical_observation
        };
        let result = out.validate_for_sphere(DomainSphere::Physics);
        assert!(
            matches!(result, Err(SchemaError::MissingRequiredField(_))),
            "expected MissingRequiredField, got {result:?}"
        );
    }

    #[test]
    fn stimulus_text_helpers() {
        let s = Stimulus::text("hello");
        assert_eq!(s.text_str(), "hello");
        let m = Stimulus::Multimodal {
            text: "caption".into(),
            parts: vec![],
        };
        assert_eq!(m.text_str(), "caption");
    }

    #[test]
    fn multimodal_stimulus_serializes_openai_shapes() -> Result<(), Box<dyn std::error::Error>> {
        let parts = vec![
            StimulusPart::ImageUrl {
                image_url: ImageUrlRef {
                    url: "data:image/png;base64,AAAA".into(),
                    detail: ImageDetail::High,
                },
            },
            StimulusPart::InputAudio {
                input_audio: InputAudioRef {
                    // OpenAI `input_audio.data` is raw base64; forwarded verbatim.
                    data: "Bg==".into(),
                    format: AudioFormat::Wav,
                },
            },
        ];
        let json = serde_json::to_string(&parts)?;
        assert!(
            json.contains(r#""type":"image_url""#),
            "image part must use type=image_url: {json}"
        );
        assert!(
            json.contains(r#""type":"input_audio""#),
            "audio part must use type=input_audio: {json}"
        );
        assert!(
            json.contains(r#""image_url":{"url":"data:image/png;base64,AAAA","detail":"high"}"#),
            "nested image_url object wrong: {json}"
        );
        assert!(
            json.contains(r#""input_audio":{"data":"Bg==","format":"wav"}"#),
            "nested input_audio object wrong: {json}"
        );
        Ok(())
    }
}
