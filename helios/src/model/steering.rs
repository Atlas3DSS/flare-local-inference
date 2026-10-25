//! Immutable request-local additive activation vectors and their cache identity.

use super::contract::SteeringContract;
use crate::Result;

#[derive(Clone, Debug, Default)]
pub struct SteeringData {
    pub vectors: Vec<(usize, Vec<f32>)>,
    pub phase: i32,
    pub prompt_tokens: usize,
    pub identity: Option<[u8; 32]>,
}

impl SteeringData {
    pub fn validate(&self, catalog: Option<&SteeringContract>) -> Result<()> {
        if self.vectors.is_empty() {
            if self.identity.is_some() || self.phase != 0 || self.prompt_tokens != 0 {
                return Err("empty steering must have no phase or identity".into());
            }
            return Ok(());
        }
        let catalog = catalog.ok_or("steering is not available in this model package")?;
        if !(1..=3).contains(&self.phase)
            || self.identity.is_none()
            || self.prompt_tokens == 0
            || self.vectors.len() > catalog.max_sites
            || self.vectors.windows(2).any(|pair| pair[0].0 >= pair[1].0)
            || self.vectors.iter().any(|(site, vector)| {
                catalog
                    .sites
                    .get(*site)
                    .is_none_or(|s| s.width != vector.len())
                    || vector.iter().any(|v| !v.is_finite())
            })
        {
            return Err("steering vectors violate the prepared site catalog".into());
        }
        Ok(())
    }
}

#[cfg(feature = "server")]
#[derive(Clone, Debug, PartialEq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SteeringRequest {
    #[serde(default)]
    pub phase: SteeringPhase,
    pub sites: Vec<SteeringVector>,
}

#[cfg(feature = "server")]
#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SteeringPhase {
    Prefill,
    Decode,
    #[default]
    All,
}

#[cfg(feature = "server")]
#[derive(Clone, Debug, PartialEq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SteeringVector {
    pub site: String,
    pub vector: Vec<f32>,
}

#[cfg(feature = "server")]
impl SteeringRequest {
    pub fn prepare(
        &self,
        catalog: Option<&SteeringContract>,
        prompt_tokens: usize,
    ) -> Result<SteeringData> {
        use blake2::{Blake2b, Digest, digest::consts::U32};
        if self.sites.is_empty() {
            return Ok(SteeringData::default());
        }
        let catalog = catalog.ok_or("steering is not available in this model package")?;
        if self.sites.len() > catalog.max_sites {
            return Err(format!(
                "steering exceeds this package's maximum of {} sites",
                catalog.max_sites
            )
            .into());
        }
        let phase: i32 = match self.phase {
            SteeringPhase::Prefill => 1,
            SteeringPhase::Decode => 2,
            SteeringPhase::All => 3,
        };
        let mut vectors = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for entry in &self.sites {
            let (index, site) = catalog
                .sites
                .iter()
                .enumerate()
                .find(|(_, s)| s.name == entry.site)
                .ok_or_else(|| format!("unknown steering site {:?}", entry.site))?;
            if !seen.insert(index)
                || entry.vector.len() != site.width
                || entry.vector.iter().any(|v| !v.is_finite())
            {
                return Err(
                    "steering sites must be unique with finite vectors of their declared width"
                        .into(),
                );
            }
            if entry.vector.iter().any(|v| *v != 0.0) {
                // Signed zero is the same disabled coordinate, and must hash identically.
                vectors.push((
                    index,
                    entry
                        .vector
                        .iter()
                        .map(|v| if *v == 0.0 { 0.0 } else { *v })
                        .collect::<Vec<_>>(),
                ));
            }
        }
        if vectors.is_empty() {
            return Ok(SteeringData::default());
        }
        vectors.sort_by_key(|(site, _)| *site);
        let mut hash = Blake2b::<U32>::new();
        hash.update(b"flare-additive-bf16-sites-v1\0");
        hash.update(phase.to_le_bytes());
        // The boundary is semantic, independent of graph chunking and prefix-cache hits.
        if phase != 3 {
            hash.update((prompt_tokens as u64).to_le_bytes());
        }
        for (index, vector) in &vectors {
            let site = &catalog.sites[*index];
            hash.update((site.name.len() as u64).to_le_bytes());
            hash.update(site.name.as_bytes());
            hash.update((site.width as u64).to_le_bytes());
            for value in vector {
                hash.update(value.to_bits().to_le_bytes());
            }
        }
        let result = SteeringData {
            vectors,
            phase,
            prompt_tokens,
            identity: Some(hash.finalize().into()),
        };
        result.validate(Some(catalog))?;
        Ok(result)
    }
}

#[cfg(all(test, feature = "server"))]
mod tests {
    use super::*;
    use crate::cache::{ConditioningEvent, PrefixHistory};
    use crate::model::contract::SteeringSite;
    use serde_json::json;

    fn catalog() -> SteeringContract {
        SteeringContract {
            max_sites: 2,
            sites: vec![
                SteeringSite {
                    name: "layers.0.mlp_out".into(),
                    width: 2,
                },
                SteeringSite {
                    name: "layers.1.residual".into(),
                    width: 2,
                },
                SteeringSite {
                    name: "layers.0.mlp_input".into(),
                    width: 3,
                },
            ],
        }
    }

    fn request(phase: &str, value: f32) -> SteeringRequest {
        serde_json::from_value(json!({"phase":phase,"sites":[
            {"site":"layers.0.mlp_out","vector":[value,0.0]}]}))
        .unwrap()
    }

    #[test]
    fn unsupported_packages_reject_nonempty_programs_and_zero_is_neutral() {
        assert!(request("all", 0.0).prepare(None, 4).is_err());
        let empty: SteeringRequest = serde_json::from_value(json!({"sites":[]})).unwrap();
        assert!(empty.prepare(None, 4).unwrap().identity.is_none());
        let zero = request("all", -0.0).prepare(Some(&catalog()), 4).unwrap();
        assert!(zero.vectors.is_empty());
        assert!(zero.identity.is_none());
        assert_eq!(zero.phase, 0);
    }

    #[test]
    fn vectors_validate_names_width_capacity_duplicates_and_finiteness() {
        for sites in [
            json!([{"site":"unknown","vector":[1,2]}]),
            json!([{"site":"layers.0.mlp_out","vector":[1]}]),
            json!([{"site":"layers.0.mlp_out","vector":[1,2]},
                   {"site":"layers.0.mlp_out","vector":[1,2]}]),
        ] {
            let r: SteeringRequest = serde_json::from_value(json!({"sites":sites})).unwrap();
            assert!(r.prepare(Some(&catalog()), 4).is_err());
        }
        let mut r = request("all", 1.0);
        r.sites[0].vector[0] = f32::INFINITY;
        assert!(r.prepare(Some(&catalog()), 4).is_err());
        assert!(
            serde_json::from_value::<SteeringRequest>(json!({"phase":"later","sites":[]})).is_err()
        );
        assert!(
            serde_json::from_value::<SteeringRequest>(json!({"sites":[],"callback":"anything"}))
                .is_err()
        );
    }

    #[test]
    fn neuron_site_uses_its_intermediate_width_and_distinct_cache_identity() {
        let c = catalog();
        let neuron: SteeringRequest = serde_json::from_value(json!({"sites":[
            {"site":"layers.0.mlp_input","vector":[1.0,0.0,-0.5]}]})).unwrap();
        let prepared = neuron.prepare(Some(&c), 4).unwrap();
        assert_eq!(prepared.vectors, vec![(2, vec![1.0, 0.0, -0.5])]);
        assert_ne!(prepared.identity, request("all", 1.0).prepare(Some(&c), 4).unwrap().identity);
        let mut wrong_width = neuron.clone();
        wrong_width.sites[0].vector.pop();
        assert!(wrong_width.prepare(Some(&c), 4).is_err());
        let old = SteeringContract { sites: c.sites[..2].to_vec(), max_sites: 2 };
        assert!(neuron.prepare(Some(&old), 4).is_err());
    }

    #[test]
    fn cache_identity_is_canonical_and_separates_dose_phase_and_boundary() {
        let c = catalog();
        let base = request("all", 1.0).prepare(Some(&c), 4).unwrap();
        assert_eq!(
            base.identity,
            request("all", 1.0).prepare(Some(&c), 8).unwrap().identity
        );
        for other in [
            request("all", 2.0),
            request("prefill", 1.0),
            request("decode", 1.0),
        ] {
            assert_ne!(base.identity, other.prepare(Some(&c), 4).unwrap().identity);
        }
        assert_ne!(
            request("decode", 1.0)
                .prepare(Some(&c), 4)
                .unwrap()
                .identity,
            request("decode", 1.0)
                .prepare(Some(&c), 8)
                .unwrap()
                .identity
        );
        let mut r = request("all", 1.0);
        r.sites.push(SteeringVector {
            site: "layers.1.residual".into(),
            vector: vec![0.0, 3.0],
        });
        let ordered = r.prepare(Some(&c), 4).unwrap();
        r.sites.reverse();
        assert_eq!(ordered.identity, r.prepare(Some(&c), 4).unwrap().identity);
        let tokens = [7; 65];
        let mut ordinary = PrefixHistory::new([1; 32]);
        ordinary.append(&tokens, &[]).unwrap();
        let mut steered = PrefixHistory::new([1; 32]);
        steered
            .append(
                &tokens,
                &[ConditioningEvent {
                    position: 0,
                    identity: base.identity.unwrap(),
                }],
            )
            .unwrap();
        for boundary in [1, 64, 65] {
            assert_ne!(ordinary.key_at(boundary), steered.key_at(boundary));
        }
    }
}
