use crate::Result;
use serde_json::{Map, Value};

pub(crate) fn parse(body: &str, tools: &[Value], separator: &str) -> Result<Vec<(String, String)>> {
    let mut rest = body.trim();
    let mut calls = Vec::new();
    while !rest.is_empty() {
        let invoke = rest
            .strip_prefix(&format!("<｜DSML｜{separator}invoke name=\""))
            .ok_or("DSML call requires an invoke tag")?;
        let (name, tail) = invoke
            .split_once("\">")
            .ok_or("unterminated DSML function name")?;
        if name.is_empty()
            || !tools
                .iter()
                .any(|tool| tool["function"]["name"].as_str() == Some(name))
        {
            return Err(format!("model called unregistered tool {name:?}").into());
        }
        let (mut parameters, tail) = tail
            .split_once(&format!("</｜DSML｜{separator}invoke>"))
            .ok_or("unterminated DSML invocation")?;
        let mut arguments = Map::new();
        while !parameters.trim().is_empty() {
            let parameter = parameters
                .trim_start()
                .strip_prefix(&format!("<｜DSML｜{separator}parameter name=\""))
                .ok_or("DSML invocation requires parameter tags")?;
            let (key, parameter) = parameter
                .split_once("\" string=\"")
                .ok_or("DSML parameter requires a string attribute")?;
            let (string, parameter) = parameter
                .split_once("\">")
                .ok_or("unterminated DSML parameter attributes")?;
            let (raw, next) = parameter
                .split_once(&format!("</｜DSML｜{separator}parameter>"))
                .ok_or("unterminated DSML parameter")?;
            if key.is_empty() || arguments.contains_key(key) {
                return Err("DSML parameter names must be nonempty and unique".into());
            }
            let value = match string {
                "true" => Value::String(raw.into()),
                "false" => serde_json::from_str(raw)
                    .map_err(|e| format!("invalid DSML JSON parameter: {e}"))?,
                _ => return Err("DSML string attribute must be true or false".into()),
            };
            arguments.insert(key.into(), value);
            parameters = next;
        }
        calls.push((name.into(), Value::Object(arguments).to_string()));
        rest = tail.trim_start();
    }
    if calls.is_empty() {
        return Err("DSML calls block is empty".into());
    }
    Ok(calls)
}
