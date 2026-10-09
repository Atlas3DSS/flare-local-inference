//! Native JSON-schema and llguidance Lark constraints over exact tokenizer bytes.

use std::sync::Arc;

use llguidance::api::{ParserLimits, TopLevelGrammar};
use llguidance::toktrie::{TokEnv, TokRxInfo, TokTrie, TokenId, TokenizerEnv};
use llguidance::{Matcher, ParserFactory, token_bytes_from_tokenizer_json};
use serde_json::Value;
use tokenizers::Tokenizer;

use crate::Result;

const MAX_GRAMMAR_BYTES: usize = 262_144;
const MAX_SCHEMA_NODES: usize = 8_192;
const MAX_SCHEMA_DEPTH: usize = 64;

struct CheckpointTokenizer {
    tokenizer: Tokenizer,
    trie: TokTrie,
}

impl TokenizerEnv for CheckpointTokenizer {
    fn tok_trie(&self) -> &TokTrie {
        &self.trie
    }

    fn tokenize_bytes(&self, bytes: &[u8]) -> Vec<TokenId> {
        self.trie.tokenize_with_greedy_fallback(bytes, |text| {
            self.tokenizer
                .encode(text, false)
                .expect("validated byte-level checkpoint tokenizer failed to encode UTF-8")
                .get_ids()
                .to_vec()
        })
    }

    fn tokenize_is_canonical(&self) -> bool {
        // NFC normalization need not preserve grammar bytes. Masking uses the exact
        // byte trie; disabling token forcing prevents retokenization of those bytes.
        false
    }
}

/// Shared tokenizer and bounded grammar compiler; each request owns its parser.
pub struct GrammarCompiler {
    factory: ParserFactory,
    model_vocab: usize,
    tokenizer_vocab: usize,
    eos_ids: Arc<[u32]>,
}

impl GrammarCompiler {
    pub fn new(tokenizer: &Tokenizer, model_vocab: usize, eos_ids: &[u32]) -> Result<Self> {
        if model_vocab == 0 || model_vocab > u32::MAX as usize {
            return Err("grammar model vocabulary must fit a nonzero u32".into());
        }
        let json = tokenizer
            .to_string(false)
            .map_err(|error| error.to_string())?;
        let json: Value = serde_json::from_str(&json).map_err(|error| error.to_string())?;
        if json["decoder"]["type"] != "ByteLevel" {
            return Err("grammar requires the checkpoint ByteLevel decoder".into());
        }
        let bytes = token_bytes_from_tokenizer_json(&json).map_err(|error| error.to_string())?;
        if bytes.len() > model_vocab || bytes.is_empty() || bytes.iter().any(Vec::is_empty) {
            return Err("grammar tokenizer has missing IDs or exceeds the model vocabulary".into());
        }
        if eos_ids.is_empty()
            || eos_ids.iter().enumerate().any(|(index, &id)| {
                id as usize >= bytes.len()
                    || eos_ids[..index].contains(&id)
                    || bytes[id as usize][0] != TokTrie::SPECIAL_TOKEN_MARKER
            })
        {
            return Err("grammar EOS IDs must be distinct checkpoint special tokens".into());
        }
        let tokenizer_vocab = bytes.len();
        let trie = TokTrie::from(&TokRxInfo::new(tokenizer_vocab as u32, eos_ids[0]), &bytes)
            .with_eos_tokens(eos_ids);
        let mut tokenizer = tokenizer.clone();
        tokenizer
            .with_truncation(None)
            .map_err(|error| error.to_string())?;
        tokenizer.with_padding(None);
        let environment: TokEnv = Arc::new(CheckpointTokenizer { tokenizer, trie });
        let mut factory =
            ParserFactory::new_simple(&environment).map_err(|error| error.to_string())?;
        factory.quiet();
        *factory.limits_mut() = ParserLimits {
            max_grammar_size: 131_072,
            max_lexer_states: 32_768,
            verbose_errors: false,
            ..ParserLimits::default()
        };
        Ok(Self {
            factory,
            model_vocab,
            tokenizer_vocab,
            eos_ids: eos_ids.into(),
        })
    }

    /// Compile supported JSON-schema assertions strictly; external references and
    /// compiler option extensions are rejected before compilation.
    pub fn compile_json_schema(&self, schema: &Value) -> Result<RequestGrammar> {
        if schema.to_string().len() > MAX_GRAMMAR_BYTES {
            return Err(format!("JSON schema exceeds {MAX_GRAMMAR_BYTES} bytes"));
        }
        validate_schema_nodes(schema, "$", 0, &mut 0)?;
        self.compile(TopLevelGrammar::from_json_schema(schema.clone()))
    }

    /// Compile the llguidance Lark dialect. Embedded JSON and compiler directives
    /// use separate APIs so they cannot override strict JSON-schema validation.
    pub fn compile_lark(&self, grammar: &str) -> Result<RequestGrammar> {
        if grammar.is_empty() || grammar.len() > MAX_GRAMMAR_BYTES {
            return Err(format!(
                "Lark grammar must contain 1..={MAX_GRAMMAR_BYTES} bytes"
            ));
        }
        if grammar.contains("%json") || grammar.contains("%llguidance") {
            return Err("Lark %json and %llguidance directives are unsupported; use the JSON-schema API for schemas".into());
        }
        self.compile(TopLevelGrammar::from_lark(grammar.to_owned()))
    }

    fn compile(&self, grammar: TopLevelGrammar) -> Result<RequestGrammar> {
        let parser = self
            .factory
            .create_parser(grammar)
            .map_err(|error| format!("invalid grammar: {error:#}"))?;
        let mut matcher = Matcher::new(Ok(parser));
        let warnings = matcher.grammar_warnings();
        if !warnings.is_empty() {
            return Err(format!(
                "grammar compilation warnings are rejected: {}",
                warnings.join("; ")
            ));
        }
        let mut request = RequestGrammar {
            matcher,
            mask: vec![0; self.model_vocab.div_ceil(32)],
            tokenizer_vocab: self.tokenizer_vocab,
            eos_ids: self.eos_ids.clone(),
            accepting: false,
            stopped: false,
            error: None,
        };
        request.refresh_mask()?;
        Ok(request)
    }
}

/// Per-generation parser. Advance once per sampled completion before uploading
/// the following mask; prompt tokens and consumed pending tokens do not advance it.
pub struct RequestGrammar {
    matcher: Matcher,
    mask: Vec<i32>,
    tokenizer_vocab: usize,
    eos_ids: Arc<[u32]>,
    accepting: bool,
    stopped: bool,
    error: Option<String>,
}

impl RequestGrammar {
    /// Returns masks before each legal non-EOS draft and after the final draft.
    /// The committed parser, including its lexer caches, remains independent.
    pub fn copy_masks(&self, tokens: &[u32]) -> Result<(usize, Vec<i32>)> {
        if let Some(error) = &self.error {
            return Err(error.clone());
        }
        if self.stopped {
            return Err("cannot propose tokens for a stopped grammar".into());
        }
        if tokens.len() > 7 {
            return Err("copy grammar supports at most seven draft tokens".into());
        }
        let mut tentative = Self {
            matcher: self.matcher.deep_clone(),
            mask: self.mask.clone(),
            tokenizer_vocab: self.tokenizer_vocab,
            eos_ids: self.eos_ids.clone(),
            accepting: self.accepting,
            stopped: false,
            error: None,
        };
        let mut masks = Vec::with_capacity((tokens.len() + 1) * self.mask.len());
        masks.extend_from_slice(&self.mask);
        let mut count = 0;
        for &token in tokens {
            let id = token as usize;
            // EOS remains a target outcome; a stopped parser has no bonus mask.
            if self.eos_ids.contains(&token)
                || id >= self.tokenizer_vocab
                || (tentative.mask[id / 32] as u32 & (1 << (id % 32))) == 0
            {
                break;
            }
            if let Err(error) = tentative.matcher.consume_token(token) {
                return Err(format!("grammar advancement failed: {error:#}"));
            }
            if !tentative.refresh_mask_available()? {
                break;
            }
            masks.extend_from_slice(tentative.mask());
            count += 1;
        }
        Ok((count, masks))
    }

    /// Int32 words with token t stored in bit (t % 32) of word (t / 32).
    /// Padded model IDs and the mask of a stopped or failed parser are zero.
    pub fn mask(&self) -> &[i32] {
        &self.mask
    }

    pub fn is_accepting(&self) -> bool {
        self.accepting && self.error.is_none()
    }

    /// True after a permitted EOS token was consumed, not merely at a valid prefix.
    pub fn is_stopped(&self) -> bool {
        self.stopped
    }

    pub fn advance(&mut self, token: u32) -> Result<()> {
        if let Some(error) = &self.error {
            return Err(error.clone());
        }
        if self.stopped {
            return Err("grammar already consumed EOS".into());
        }
        let id = token as usize;
        if id >= self.tokenizer_vocab || (self.mask[id / 32] as u32 & (1 << (id % 32))) == 0 {
            return self.fail(format!("token {token} is forbidden by the grammar mask"));
        }
        if self.eos_ids.contains(&token) {
            if !self.accepting {
                return self.fail("grammar EOS reached a non-accepting parser state".into());
            }
            self.stopped = true;
            self.mask.fill(0);
            return Ok(());
        }
        if let Err(error) = self.matcher.consume_token(token) {
            return self.fail(format!("grammar advancement failed: {error:#}"));
        }
        self.refresh_mask()
    }

    fn refresh_mask(&mut self) -> Result<()> {
        if !self.refresh_mask_available()? {
            return self.fail("grammar has no allowed checkpoint tokens".into());
        }
        Ok(())
    }

    fn refresh_mask_available(&mut self) -> Result<bool> {
        self.mask.fill(0);
        if !self.matcher.stop_reason().is_ok() {
            return self.fail(format!(
                "grammar stopped with {:?}",
                self.matcher.stop_reason()
            ));
        }
        self.accepting = match self.matcher.is_accepting() {
            Ok(value) => value,
            Err(error) => return self.fail(format!("grammar acceptance check failed: {error:#}")),
        };
        let allowed = match self.matcher.compute_mask_or_eos() {
            Ok(mask) => mask,
            Err(error) => return self.fail(format!("grammar mask computation failed: {error:#}")),
        };
        for (destination, &word) in self.mask.iter_mut().zip(allowed.as_slice()) {
            *destination = word as i32;
        }
        let tail = self.tokenizer_vocab % 32;
        if tail != 0 {
            self.mask[self.tokenizer_vocab / 32] &= ((1_u32 << tail) - 1) as i32;
        }
        Ok(self.mask.iter().any(|&word| word != 0))
    }

    fn fail<T>(&mut self, error: String) -> Result<T> {
        self.mask.fill(0);
        self.accepting = false;
        self.error = Some(error.clone());
        Err(error)
    }
}

fn validate_schema_nodes(
    schema: &Value,
    path: &str,
    depth: usize,
    nodes: &mut usize,
) -> Result<()> {
    *nodes += 1;
    if depth > MAX_SCHEMA_DEPTH || *nodes > MAX_SCHEMA_NODES {
        return Err("JSON schema exceeds the depth or node limit".into());
    }
    if schema.is_boolean() {
        return Ok(());
    }
    let object = schema
        .as_object()
        .ok_or_else(|| format!("schema {path} must be an object or boolean"))?;
    for (key, value) in object {
        let child = format!("{path}/{key}");
        match key.as_str() {
            "$ref" => {
                if !value
                    .as_str()
                    .is_some_and(|reference| reference.starts_with('#'))
                {
                    return Err(format!("schema {child} must reference the same document"));
                }
            }
            "$defs" | "definitions" | "properties" | "patternProperties" => {
                let entries = value
                    .as_object()
                    .ok_or_else(|| format!("schema {child} must be an object"))?;
                for (name, nested) in entries {
                    validate_schema_nodes(nested, &format!("{child}/{name}"), depth + 1, nodes)?;
                }
            }
            "anyOf" | "oneOf" | "allOf" | "prefixItems" => {
                let entries = value
                    .as_array()
                    .ok_or_else(|| format!("schema {child} must be an array"))?;
                for (index, nested) in entries.iter().enumerate() {
                    validate_schema_nodes(nested, &format!("{child}/{index}"), depth + 1, nodes)?;
                }
            }
            "items" if value.is_array() => {
                for (index, nested) in value.as_array().unwrap().iter().enumerate() {
                    validate_schema_nodes(nested, &format!("{child}/{index}"), depth + 1, nodes)?;
                }
            }
            "items" | "additionalItems" | "additionalProperties" => {
                validate_schema_nodes(value, &child, depth + 1, nodes)?;
            }
            "$anchor" | "$schema" | "$id" | "id" | "$comment" | "title" | "description"
            | "default" | "readOnly" | "writeOnly" | "examples" | "contentMediaType"
            | "contentEncoding" | "const" | "enum" | "type" | "minItems" | "maxItems"
            | "required" | "minProperties" | "maxProperties" | "minLength" | "maxLength"
            | "pattern" | "format" | "minimum" | "maximum" | "exclusiveMinimum"
            | "exclusiveMaximum" | "multipleOf" => {}
            _ => return Err(format!("unsupported JSON-schema keyword {child}")),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn byte_tokenizer() -> Tokenizer {
        let mut vocab = serde_json::Map::new();
        let mut extra = 256;
        for byte in 0..=255_u32 {
            let mapped = if matches!(byte, 33..=126 | 161..=172 | 174..=255) {
                byte
            } else {
                extra += 1;
                extra - 1
            };
            vocab.insert(char::from_u32(mapped).unwrap().to_string(), json!(byte));
        }
        let tokenizer = json!({
            "version": "1.0", "truncation": null, "padding": null,
            "added_tokens": [
                {"id":256,"content":"<eos>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},
                {"id":257,"content":"<end>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}
            ],
            "normalizer": null,
            "pre_tokenizer": {"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},
            "post_processor": null,
            "decoder": {"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},
            "model": {"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":"","end_of_word_suffix":"","fuse_unk":false,"vocab":vocab,"merges":[]}
        });
        Tokenizer::from_bytes(tokenizer.to_string()).unwrap()
    }

    fn allowed(request: &RequestGrammar, token: u32) -> bool {
        request.mask()[token as usize / 32] as u32 & (1 << (token % 32)) != 0
    }

    fn accepts(compiler: &GrammarCompiler, schema: &Value, text: &str, eos: u32) -> bool {
        let mut request = compiler.compile_json_schema(schema).unwrap();
        for &token in text.as_bytes() {
            if request.advance(u32::from(token)).is_err() {
                return false;
            }
        }
        request.advance(eos).is_ok()
    }

    #[test]
    fn exact_mask_words_eos_and_padded_ids() {
        let compiler = GrammarCompiler::new(&byte_tokenizer(), 320, &[256, 257]).unwrap();
        let mut request = compiler.compile_lark(r"start: /[\x1f]/").unwrap();
        assert_eq!(request.mask().len(), 10);
        assert_eq!(request.mask()[0], i32::MIN);
        assert_eq!(request.mask().iter().filter(|&&word| word != 0).count(), 1);
        request.advance(31).unwrap();
        assert!(request.is_accepting());
        assert!(!request.is_stopped());
        assert!(allowed(&request, 256) && allowed(&request, 257));
        assert!((258..320).all(|id| !allowed(&request, id)));
        request.advance(257).unwrap();
        assert!(request.is_stopped());
        assert!(request.mask().iter().all(|&word| word == 0));
        assert!(request.advance(257).is_err());
    }

    #[test]
    fn required_fields_enums_ranges_and_additional_properties() {
        let compiler = GrammarCompiler::new(&byte_tokenizer(), 320, &[256, 257]).unwrap();
        let schema = json!({"type":"object","properties":{
            "action":{"enum":["read","write"]},
            "count":{"type":"integer","minimum":1,"maximum":3}
        },"required":["action","count"],"additionalProperties":false});
        for text in [
            r#"{"action":"read","count":1}"#,
            r#"{"action":"write", "count":3}"#,
        ] {
            assert!(accepts(&compiler, &schema, text, 256), "{text}");
        }
        for text in [
            r#"{"action":"read"}"#,
            r#"{"action":"erase","count":1}"#,
            r#"{"action":"read","count":0}"#,
            r#"{"action":"read","count":4}"#,
            r#"{"action":"read","count":2,"extra":true}"#,
            r#"{"action":"read","count":"2"}"#,
        ] {
            assert!(!accepts(&compiler, &schema, text, 257), "{text}");
        }
    }

    #[test]
    fn local_references_arrays_unicode_and_nullable_values() {
        let compiler = GrammarCompiler::new(&byte_tokenizer(), 320, &[256, 257]).unwrap();
        let schema = json!({"$defs":{"item":{"anyOf":[{"type":"null"},{"enum":["é","猫","😀"]}]}},
            "type":"array","items":{"$ref":"#/$defs/item"},"minItems":1,"maxItems":2});
        for text in [r#"[null]"#, r#"["é","猫"]"#, r#"["😀"]"#] {
            assert!(accepts(&compiler, &schema, text, 257), "{text}");
        }
        for text in ["[]", "[null,null,null]", "[false]", r#"["wrong"]"#] {
            assert!(!accepts(&compiler, &schema, text, 256), "{text}");
        }
    }

    #[test]
    fn masked_tokens_poison_parser_without_permissive_fallback() {
        let compiler = GrammarCompiler::new(&byte_tokenizer(), 320, &[256, 257]).unwrap();
        let mut request = compiler
            .compile_json_schema(&json!({"const":true}))
            .unwrap();
        assert!(!allowed(&request, 256));
        let error = request.advance(b'f' as u32).unwrap_err();
        assert!(error.contains("forbidden"));
        assert_eq!(request.advance(b't' as u32).unwrap_err(), error);
        assert!(request.mask().iter().all(|&word| word == 0));
        assert!(!request.is_accepting());
    }

    #[test]
    fn copy_masks_include_bonus_and_rejection_does_not_advance_committed_parser() {
        let compiler = GrammarCompiler::new(&byte_tokenizer(), 320, &[256, 257]).unwrap();
        let mut committed = compiler.compile_lark("start: \"abc\" | \"axy\"").unwrap();
        committed.advance(b'a' as u32).unwrap();
        let before = committed.mask().to_vec();
        let (count, masks) = committed
            .copy_masks(&[b'b' as u32, b'c' as u32, 256])
            .unwrap();
        assert_eq!(count, 2);
        assert_eq!(masks.len(), 3 * before.len());
        assert_eq!(committed.mask(), before);
        let mut serial = compiler.compile_lark("start: \"abc\" | \"axy\"").unwrap();
        serial.advance(b'a' as u32).unwrap();
        for (index, mask) in masks.chunks(before.len()).enumerate() {
            assert_eq!(mask, serial.mask());
            if index < 2 {
                serial.advance(b"bc"[index] as u32).unwrap();
            }
        }
        assert!(serial.is_accepting());
        assert!(masks[2 * before.len() + 256 / 32] as u32 & 1 != 0);
        // A correction at the first candidate follows the unmodified parser.
        committed.advance(b'x' as u32).unwrap();
        committed.advance(b'y' as u32).unwrap();
        committed.advance(257).unwrap();
        assert!(committed.is_stopped());
        assert!(committed.copy_masks(&[]).is_err());
    }

    #[test]
    fn copy_masks_trim_forbidden_ids_without_poisoning_and_preserve_utf8_prefixes() {
        let compiler = GrammarCompiler::new(&byte_tokenizer(), 320, &[256, 257]).unwrap();
        let mut committed = compiler
            .compile_json_schema(&json!({"const":"猫"}))
            .unwrap();
        let original = committed.mask().to_vec();
        let tokens = "\"猫\""
            .as_bytes()
            .iter()
            .map(|b| *b as u32)
            .collect::<Vec<_>>();
        let (count, masks) = committed.copy_masks(&tokens).unwrap();
        assert_eq!(count, tokens.len());
        assert_eq!(committed.mask(), original);
        for (index, &token) in tokens.iter().enumerate() {
            assert_eq!(&masks[index * 10..(index + 1) * 10], committed.mask());
            committed.advance(token).unwrap();
        }
        assert_eq!(&masks[count * 10..], committed.mask());
        assert_eq!(
            committed.copy_masks(&[u32::MAX]).unwrap(),
            (0, committed.mask().to_vec())
        );
        assert!(committed.advance(256).is_ok());
        let parser = compiler.compile_lark("start: \"ab\"").unwrap();
        let (count, masks) = parser.copy_masks(&[b'a' as u32, b'x' as u32]).unwrap();
        assert_eq!((count, masks.len()), (1, 20));
        assert_eq!(
            parser.copy_masks(&[b'x' as u32]).unwrap(),
            (0, parser.mask().to_vec())
        );
    }

    #[test]
    fn unsupported_invalid_and_external_schemas_fail_before_admission() {
        let compiler = GrammarCompiler::new(&byte_tokenizer(), 320, &[256, 257]).unwrap();
        for schema in [
            json!({"type":"array","uniqueItems":true}),
            json!({"type":"integer","minimim":1}),
            json!({"properties":{"x":{"nullable":true}}}),
            json!({"type":"string","minLength":-1}),
            json!({"type":"string","format":"not-a-format"}),
            json!({"oneOf":[{"type":"integer"},{"type":"number"}]}),
            json!({"x-guidance":{"lenient":true},"uniqueItems":true}),
            json!({"$ref":"https://example.invalid/schema"}),
            json!({"$ref":"file:///etc/passwd"}),
            json!({"$ref":"#/$defs/missing"}),
            json!({"type":"integer","minimum":5,"maximum":1}),
            json!(false),
        ] {
            assert!(
                compiler.compile_json_schema(&schema).is_err(),
                "accepted {schema}"
            );
        }
        assert!(
            compiler
                .compile_lark("start: %json {\"x-guidance\":{\"lenient\":true}}")
                .is_err()
        );
        assert!(compiler.compile_lark("start: missing").is_err());
    }

    #[test]
    fn bounded_compilation_and_thread_ownership() {
        fn send_sync<T: Send + Sync>() {}
        fn send<T: Send>() {}
        send_sync::<GrammarCompiler>();
        send::<RequestGrammar>();
        let compiler = Arc::new(GrammarCompiler::new(&byte_tokenizer(), 320, &[256, 257]).unwrap());
        let mut schema = json!(true);
        for _ in 0..=MAX_SCHEMA_DEPTH {
            schema = json!({"type":"array","items":schema});
        }
        assert!(compiler.compile_json_schema(&schema).is_err());
        assert!(
            compiler
                .compile_lark(&" ".repeat(MAX_GRAMMAR_BYTES + 1))
                .is_err()
        );
        let compiler2 = compiler.clone();
        let mut request =
            std::thread::spawn(move || compiler2.compile_json_schema(&json!({"const":7})).unwrap())
                .join()
                .unwrap();
        request.advance(b'7' as u32).unwrap();
        request.advance(256).unwrap();
        assert!(request.is_stopped());
    }

    #[test]
    fn reject_tokenizer_or_eos_geometry_mismatch() {
        let tokenizer = byte_tokenizer();
        assert!(GrammarCompiler::new(&tokenizer, 257, &[256]).is_err());
        assert!(GrammarCompiler::new(&tokenizer, 320, &[]).is_err());
        assert!(GrammarCompiler::new(&tokenizer, 320, &[256, 256]).is_err());
        assert!(GrammarCompiler::new(&tokenizer, 320, &[256, 319]).is_err());
        assert!(GrammarCompiler::new(&tokenizer, 320, &[b'a' as u32]).is_err());
    }

    #[test]
    #[ignore = "requires SPARKGAP_CHECKPOINT pointing to the exact Qwen3.8 checkpoint"]
    fn checkpoint_mask_trace() {
        let snapshot = std::path::PathBuf::from(
            std::env::var_os("SPARKGAP_CHECKPOINT").expect("SPARKGAP_CHECKPOINT is required"),
        );
        let reference = check_checkpoint_masks(&snapshot, None);
        let trace_path = std::env::temp_dir().join(format!(
            "sparkgap-grammar-{}-{}.json",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let traced = check_checkpoint_masks(&snapshot, Some(&trace_path));
        let saved: Value = serde_json::from_slice(&std::fs::read(&trace_path).unwrap()).unwrap();
        std::fs::remove_file(trace_path).unwrap();
        assert_eq!(reference, traced);
        assert_eq!(reference, saved);
    }

    fn check_checkpoint_masks(
        snapshot: &std::path::Path,
        trace_output: Option<&std::path::Path>,
    ) -> Value {
        use sha2::{Digest, Sha256};
        let bytes = std::fs::read(snapshot.join("tokenizer.json")).unwrap();
        assert_eq!(
            format!("{:x}", Sha256::digest(&bytes)),
            "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"
        );
        let tokenizer = Tokenizer::from_bytes(&bytes).unwrap();
        let compiler = GrammarCompiler::new(&tokenizer, 248_320, &[248_046, 248_044]).unwrap();
        assert_eq!(compiler.tokenizer_vocab, 248_077);
        let cases = [
            (
                "tool_arguments",
                json!({"type":"object","properties":{"path":{"const":"/tmp/猫.txt"},"lines":{"type":"integer","minimum":1,"maximum":3}},"required":["path","lines"],"additionalProperties":false}),
                r#"{"path":"/tmp/猫.txt","lines":2}"#,
            ),
            (
                "unicode_choices",
                json!({"type":"array","items":{"enum":["é","猫","😀"]},"minItems":1,"maxItems":3}),
                r#"["é","猫","😀"]"#,
            ),
        ];
        let mut traces = Vec::new();
        for (name, schema, text) in cases {
            for eos in [248_046, 248_044] {
                let mut request = compiler.compile_json_schema(&schema).unwrap();
                let tokens = tokenizer.encode(text, false).unwrap();
                let mut steps = Vec::new();
                for &token in tokens.get_ids().iter().chain([eos].iter()) {
                    assert!(allowed(&request, token), "{name}: forbidden token {token}");
                    assert!((248_077..248_320).all(|id| !allowed(&request, id)));
                    let words: Vec<u8> = request
                        .mask()
                        .iter()
                        .flat_map(|word| word.to_le_bytes())
                        .collect();
                    steps.push(json!({"token":token,"allowed_count":request.mask().iter().map(|word| word.count_ones()).sum::<u32>(),"mask_sha256":format!("{:x}",Sha256::digest(&words)),"accepting":request.is_accepting()}));
                    request.advance(token).unwrap();
                }
                assert!(request.is_stopped());
                assert_eq!(tokenizer.decode(tokens.get_ids(), false).unwrap(), text);
                traces
                    .push(json!({"name":name,"schema":schema,"text":text,"eos":eos,"steps":steps}));
            }
        }
        let tokenizer_json: Value = serde_json::from_slice(&bytes).unwrap();
        let token_bytes = token_bytes_from_tokenizer_json(&tokenizer_json).unwrap();
        for text in ["é猫😀", "e\u{301}猫😀"] {
            let grammar = format!("start: {}", serde_json::to_string(text).unwrap());
            let tokens: Vec<u32> = text
                .as_bytes()
                .iter()
                .map(|&byte| {
                    token_bytes
                        .iter()
                        .position(|value| value.as_slice() == [byte])
                        .unwrap() as u32
                })
                .chain([248_046])
                .collect();
            let mut request = compiler.compile_lark(&grammar).unwrap();
            let mut consumed = 0;
            let mut steps = Vec::new();
            for &token in &tokens {
                let remaining = &text.as_bytes()[consumed..];
                for id in 0..248_320 {
                    let expected = if id == 248_046 || id == 248_044 {
                        remaining.is_empty()
                    } else {
                        token_bytes
                            .get(id)
                            .is_some_and(|bytes| !bytes.is_empty() && remaining.starts_with(bytes))
                    };
                    assert_eq!(
                        allowed(&request, id as u32),
                        expected,
                        "literal {text:?}, consumed {consumed}, token {id}"
                    );
                }
                let words: Vec<u8> = request
                    .mask()
                    .iter()
                    .flat_map(|word| word.to_le_bytes())
                    .collect();
                steps.push(json!({"token":token,"allowed_count":request.mask().iter().map(|word| word.count_ones()).sum::<u32>(),"mask_sha256":format!("{:x}",Sha256::digest(&words)),"accepting":request.is_accepting()}));
                request.advance(token).unwrap();
                if token != 248_046 {
                    consumed += token_bytes[token as usize].len();
                }
            }
            assert!(request.is_stopped());
            assert_eq!(
                tokenizer
                    .decode(&tokens[..tokens.len() - 1], false)
                    .unwrap(),
                text
            );
            traces.push(json!({"name":"literal_byte_boundaries","lark":grammar,"text":text,"eos":248046,"full_vocab_mask_equality":true,"steps":steps}));
        }
        let mut incomplete = compiler
            .compile_json_schema(&json!({"const":{"ok":true}}))
            .unwrap();
        assert!(incomplete.advance(248_046).is_err());
        let trace = json!({"model":"local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD","revision":"dc5b8ab2fa14bc4fecd6886273b7ebc6095662e2","llguidance":"1.9.1","tokenizers":"0.23.2","mask_words":7760,"tokenizer_vocab":248077,"model_vocab":248320,"traces":traces});
        if let Some(path) = trace_output {
            std::fs::write(path, serde_json::to_string_pretty(&trace).unwrap() + "\n").unwrap();
        }
        trace
    }
}
