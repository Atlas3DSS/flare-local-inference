use super::jinja::{CompiledTemplate, Error};

const QWEN: &str = include_str!("../../../ignition/recipes/qwen38/assets/chat_template.jinja");
const STEP: &str = include_str!("../../../ignition/recipes/step5/assets/chat_template.jinja");
const KIMI3: &str = include_str!("../../../ignition/recipes/kimi3/assets/chat_template.jinja");

pub(super) fn compile(source: &str) -> Result<CompiledTemplate, Error> {
    for annotated in [QWEN, STEP, KIMI3] {
        if source == annotated.replace("{%- mark -%}", "").replace("{%- mark ns.segments | length -%}", "") {
            return CompiledTemplate::compile(annotated);
        }
    }
    let compiled = CompiledTemplate::compile(source)?;
    if compiled.annotation_spans().is_empty() {
        return Err(Error { message: "chat template requires message-boundary mark annotations; only the deployed Qwen3.8, Step, and Kimi K3 templates can be upgraded automatically".into(), offset: 0 });
    }
    Ok(compiled)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compatibility_requires_exact_legacy_bytes() {
        for annotated in [QWEN, STEP, KIMI3] {
            let mut raw = annotated.to_owned();
            for span in CompiledTemplate::compile(annotated).unwrap().annotation_spans().iter().rev() {
                raw.replace_range(span.clone(), "");
            }
            assert!(!compile(&raw).unwrap().annotation_spans().is_empty());
            assert!(compile(&(raw + " ")).is_err());
            assert!(compile(annotated).is_ok());
        }
        assert!(compile("{{ messages }}").is_err());
    }
}
