//! TOML column-pack parser (§5 column prompts, Phase 3 abstraction experiment).
//!
//! A column pack defines an explicit set of columns — each with an id, a sphere
//! (the validation *envelope*, one of the four existing `DomainSphere` variants),
//! a system prompt (which is where a column's **abstraction level** lives), and
//! an optional `level` tag (0 = lowest / most concrete, higher = more abstract)
//! so experiment results are attributable to abstraction level rather than only
//! to column id.
//!
//! No new `DomainSphere` variant is introduced. The sphere selects which keys
//! `validate_for_sphere` will presence-check; the prompt must instruct the model
//! to emit a single *flat* JSON object containing `reference_frame_coordinates`,
//! `prediction`, `confidence`, and the two sphere-specific keys.

use std::path::Path;

use ptg_core::{CorticalColumn, DomainSphere};
use serde::Deserialize;

/// Parse a TOML column pack from disk into validated `(column, level)` pairs
/// in document order. The order is significant: generated topologies apply
/// positionally over this list.
///
/// # Errors
/// Friendly `String` messages for: unreadable file, malformed TOML, unknown
/// TOML fields, empty pack, duplicate ids, empty prompts, or an unknown sphere.
pub fn load_column_pack_with_levels(
    path: &Path,
) -> Result<Vec<(CorticalColumn, Option<u8>)>, String> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| format!("could not read column pack {}: {e}", path.display()))?;
    let pack: ColumnPackToml = toml::from_str(&raw)
        .map_err(|e| format!("malformed column pack {}: {e}", path.display()))?;
    materialize_pack(pack)
}

/// A whole column pack.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ColumnPackToml {
    /// Optional human description (ignored by the runtime; for readers).
    #[serde(default, rename = "description")]
    _description: Option<String>,
    #[serde(default)]
    columns: Vec<ColumnPackColumnToml>,
}

/// One column in a pack.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ColumnPackColumnToml {
    id: String,
    /// Case-insensitive: Physics | Mathematics | Coding | Psychology. Selects
    /// the validation envelope; does NOT change the model — abstraction level
    /// lives in `system_prompt`.
    sphere: String,
    system_prompt: String,
    /// Optional abstraction-level tag (0 = lowest / most concrete). Surfaced in
    /// `--dry-run` output so experiment results are attributable to level.
    /// Not consulted by the runtime otherwise.
    #[serde(default)]
    level: Option<u8>,
}

/// Map a raw sphere string to a [`DomainSphere`]. Case-insensitive.
///
/// # Errors
/// `Err(String)` naming the accepted values if `raw` is none of them.
fn parse_sphere(raw: &str) -> Result<DomainSphere, String> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "physics" => Ok(DomainSphere::Physics),
        "mathematics" | "math" | "maths" => Ok(DomainSphere::Mathematics),
        "coding" | "code" => Ok(DomainSphere::Coding),
        "psychology" | "psych" => Ok(DomainSphere::Psychology),
        other => Err(format!(
            "unknown sphere `{other}` (expected Physics | Mathematics | Coding | Psychology)"
        )),
    }
}

/// Validate and materialize a parsed pack into columns.
///
/// # Errors
/// Empty pack, duplicate ids, blank id, blank prompt, or unknown sphere.
fn materialize_pack(pack: ColumnPackToml) -> Result<Vec<(CorticalColumn, Option<u8>)>, String> {
    if pack.columns.is_empty() {
        return Err("column pack has no [[columns]] entries".to_string());
    }
    let mut seen_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut out = Vec::with_capacity(pack.columns.len());
    for c in pack.columns {
        let id = c.id.trim();
        if id.is_empty() {
            return Err("a column has a blank id".to_string());
        }
        if !seen_ids.insert(id.to_string()) {
            return Err(format!("duplicate column id: {id}"));
        }
        if c.system_prompt.trim().is_empty() {
            return Err(format!("column {id} has a blank system_prompt"));
        }
        // `level` is intentionally not range-checked; it is an opaque analysis tag.
        let sphere = parse_sphere(&c.sphere)?;
        let col = CorticalColumn::new(id, sphere, &c.system_prompt);
        out.push((col, c.level));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    //! Pack parser tests. Use `?` throughout (no unwrap/expect/panic).

    use std::io::Write;

    use super::*;

    use std::sync::atomic::{AtomicU64, Ordering};

    static PACK_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn write_pack(toml_body: &str) -> Result<std::path::PathBuf, String> {
        let dir = std::env::temp_dir();
        let id = PACK_COUNTER.fetch_add(1, Ordering::SeqCst);
        // Include the PID: nextest runs each test in its own process, so a
        // per-process counter alone collides on a shared temp path. PID + counter
        // is unique across the whole run.
        let path = dir.join(format!("ptg-pack-test-{}-{id}.toml", std::process::id()));
        let mut f = std::fs::File::create(&path).map_err(|e| format!("create: {e}"))?;
        f.write_all(toml_body.as_bytes())
            .map_err(|e| format!("write: {e}"))?;
        Ok(path)
    }

    /// Assert a Result is an Err whose message contains `needle`, returning the message.
    fn require_err<T>(res: Result<T, String>, needle: &str) -> Result<String, String> {
        match res {
            Err(msg) => {
                if !msg.contains(needle) {
                    return Err(format!("error missing `{needle}`: {msg}"));
                }
                Ok(msg)
            }
            Ok(_) => Err(format!("expected error containing `{needle}`, got Ok")),
        }
    }

    const GOOD: &str = r#"
description = "test pack"
[[columns]]
id = "CC_PHYSICS_01"
sphere = "Physics"
level = 1
system_prompt = "be physics"
[[columns]]
id = "CC_MATH_01"
sphere = "mathematics"
system_prompt = "be math"
"#;

    #[test]
    fn valid_pack_parses_in_order() -> Result<(), String> {
        let p = write_pack(GOOD)?;
        let cols = load_column_pack_with_levels(&p)?;
        let cols: Vec<_> = cols.into_iter().map(|(c, _)| c).collect();
        assert_eq!(cols.len(), 2);
        assert_eq!(cols[0].id, "CC_PHYSICS_01");
        assert_eq!(cols[0].sphere, DomainSphere::Physics);
        assert_eq!(cols[1].id, "CC_MATH_01");
        assert_eq!(cols[1].sphere, DomainSphere::Mathematics);
        Ok(())
    }

    #[test]
    fn duplicate_ids_rejected() -> Result<(), String> {
        let body = r#"
[[columns]]
id = "CC_SAME"
sphere = "Physics"
system_prompt = "one"
[[columns]]
id = "CC_SAME"
sphere = "Coding"
system_prompt = "two"
"#;
        let p = write_pack(body)?;
        let res = load_column_pack_with_levels(&p);
        std::fs::remove_file(&p).ok();
        require_err(res.map(|v| v.len()), "duplicate column id")?;
        Ok(())
    }

    #[test]
    fn blank_prompt_rejected() -> Result<(), String> {
        let body = r#"
[[columns]]
id = "CC_A"
sphere = "Physics"
system_prompt = "real prompt"
[[columns]]
id = "CC_B"
sphere = "Coding"
system_prompt = "   "
"#;
        let p = write_pack(body)?;
        let res = load_column_pack_with_levels(&p);
        std::fs::remove_file(&p).ok();
        require_err(res.map(|v| v.len()), "blank system_prompt")?;
        Ok(())
    }

    #[test]
    fn unknown_sphere_rejected() -> Result<(), String> {
        let body = r#"
[[columns]]
id = "CC_A"
sphere = "Biology"
system_prompt = "prompt"
"#;
        let p = write_pack(body)?;
        let res = load_column_pack_with_levels(&p);
        std::fs::remove_file(&p).ok();
        require_err(res.map(|v| v.len()), "unknown sphere")?;
        Ok(())
    }

    #[test]
    fn empty_pack_rejected() -> Result<(), String> {
        let p = write_pack("description = \"empty\"\n")?;
        let res = load_column_pack_with_levels(&p);
        std::fs::remove_file(&p).ok();
        require_err(res.map(|v| v.len()), "no [[columns]]")?;
        Ok(())
    }

    #[test]
    fn level_is_optional() -> Result<(), String> {
        let body = r#"
[[columns]]
id = "CC_X"
sphere = "Coding"
system_prompt = "be code"
"#;
        let p = write_pack(body)?;
        let cols = load_column_pack_with_levels(&p)?;
        std::fs::remove_file(&p).ok();
        assert_eq!(cols.len(), 1);
        Ok(())
    }
}
