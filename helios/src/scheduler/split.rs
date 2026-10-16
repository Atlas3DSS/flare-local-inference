//! Causal prefix selection between retained proposal and verification operations.

use super::*;

pub(super) struct SplitSpeculation {
    pub width: usize,
    pub threshold: f64,
    pub retained: Option<RetainedProposal>,
}

#[derive(Clone, Copy)]
pub(super) struct RetainedProposal {
    pub request: RequestHandle,
    pub fence: Fence,
    pub consumed: u64,
    pub width: usize,
}

/// Confidence for position i may depend only on history and proposals preceding i.
/// Looking ahead past the first failed prefix would condition sampling on future draws.
pub fn causal_prefix(confidence: &[f32], threshold: f64) -> Result<usize> {
    if !threshold.is_finite() || !(0.0 < threshold && threshold <= 1.0) {
        return Err("proposal confidence threshold must be finite and in (0, 1]".into());
    }
    if confidence
        .iter()
        .any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
    {
        return Err("proposal confidence must be finite and in [0, 1]".into());
    }
    let mut survival = 1.0;
    for (index, &p) in confidence.iter().enumerate() {
        survival *= f64::from(p);
        if survival < threshold {
            return Ok(index);
        }
    }
    Ok(confidence.len())
}

impl RequestScheduler {
    pub fn split_speculation(&mut self, width: usize, threshold: f64) -> Result<()> {
        causal_prefix(&[], threshold)?;
        if self.submitted != Fence(0)
            || self.slots.iter().any(|s| s.request.is_some())
            || self.limits.max_active != 1
            || self.limits.request_capacity() != 1
            || !(2..=8).contains(&width)
            || self.speculative_widths != (1..=width).collect::<Vec<_>>()
            || !self.mixed_catalog.is_empty()
            || self.proposal_source != speculation::ProposalSource::Mtp
        {
            return Err("split speculation requires a single retained lane and every prefix width before admission".into());
        }
        self.split = Some(SplitSpeculation {
            width,
            threshold,
            retained: None,
        });
        Ok(())
    }

    pub(super) fn retained_proposal(&self, request: &LiveRequest) -> Option<RetainedProposal> {
        self.split.as_ref()?.retained.filter(|p| {
            p.request == request.handle
                && p.fence == self.completed
                && p.fence == self.submitted
                && p.consumed == request.consumed_tokens
        })
    }

    pub fn complete_proposal(
        &mut self,
        fence: Fence,
        records: &[CompletionRecord],
        timing: crate::execution::ExecutionTiming,
        confidence: &[f32],
    ) -> Result<Vec<GeneratedToken>> {
        let split = self
            .split
            .as_ref()
            .ok_or("proposal completion without a split catalog")?;
        let step = self.submitted_step(fence)?;
        if step.kind != GraphKind::Proposal
            || step.lanes.len() != 1
            || confidence.len() + 1 != split.width
        {
            return Err(
                "proposal completion does not match its retained lane and confidence extent".into(),
            );
        }
        let lane = step.lanes[0].clone();
        let width = self
            .speculative_fixed_width
            .unwrap_or(causal_prefix(confidence, split.threshold)? + 1);
        let generated = self.complete(fence, records, timing)?;
        let retained = self
            .request(lane.request)
            .ok()
            .filter(|request| request.phase == RequestPhase::Decode && request.error_code.is_none())
            .map(|request| RetainedProposal {
                request: lane.request,
                fence,
                consumed: request.consumed_tokens,
                width,
            });
        self.split.as_mut().unwrap().retained = retained;
        Ok(generated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::{GraphDescriptor, SamplingMode, TokenVector};

    fn engine() -> RequestScheduler {
        let mut s = super::super::tests::scheduler(1, 128, 8);
        s.enable_speculation(&(1..=8).collect::<Vec<_>>()).unwrap();
        let mut graphs = super::super::tests::duration_graphs(1);
        graphs.retain(|g| g.kind != GraphKind::Speculative);
        for (kind, widths) in [
            (GraphKind::Proposal, vec![8]),
            (GraphKind::Verification, (1..=8).collect()),
        ] {
            graphs.extend(widths.into_iter().map(|width| GraphDescriptor {
                name: format!("{kind:?}-{width}"),
                kind,
                sampling_mode: SamplingMode::General,
                handle: 1,
                batch_capacity: 1,
                token_capacity: width,
                prefill_capacity: 0,
                estimated_duration_ns: 1000,
                input_bindings: vec![],
            }));
        }
        s.measured_graph_durations(&graphs).unwrap();
        s.split_speculation(8, 0.15).unwrap();
        s
    }

    fn ready(s: &mut RequestScheduler) -> RequestHandle {
        let handle = s.admit(super::super::tests::spec(65, 32)).unwrap();
        while s.request(handle).unwrap().phase != RequestPhase::Decode {
            let step = s.plan().unwrap();
            let fence = Fence(s.submitted.0 + 1);
            s.commit(&step, fence).unwrap();
            let live = s.request(handle).unwrap();
            let sampled = step.kind.samples();
            let record = CompletionRecord {
                generation: handle.generation as i64,
                submission_sequence: fence.0 as i64,
                consumed_tokens: if sampled {
                    65
                } else {
                    live.consumed_tokens as i64
                },
                generated_tokens: live.generated_tokens as i64 + i64::from(sampled),
                token_id: if sampled { 3 } else { -1 },
                status: 1,
                ..Default::default()
            };
            s.complete(fence, &[record], Default::default()).unwrap();
        }
        handle
    }

    fn propose(s: &mut RequestScheduler, handle: RequestHandle) -> (Fence, CompletionRecord) {
        let step = s.plan().unwrap();
        assert_eq!((step.kind, step.token_capacity), (GraphKind::Proposal, 8));
        let fence = Fence(s.submitted.0 + 1);
        s.commit(&step, fence).unwrap();
        assert!(s.plan().is_none());
        let live = s.request(handle).unwrap();
        (
            fence,
            CompletionRecord {
                generation: handle.generation as i64,
                submission_sequence: fence.0 as i64,
                consumed_tokens: live.consumed_tokens as i64,
                generated_tokens: live.generated_tokens as i64,
                token_id: -1,
                status: 1,
                vector: Some(TokenVector::default()),
                ..Default::default()
            },
        )
    }

    #[test]
    fn causal_prefix_stops_before_later_confidence_can_change_the_decision() {
        assert_eq!(
            causal_prefix(&[0.9, 0.8, 0.3, 0.1, 1.0, 1.0, 1.0], 0.15).unwrap(),
            3
        );
        assert_eq!(causal_prefix(&[0.0, 1.0, 1.0], 0.15).unwrap(), 0);
        assert_eq!(causal_prefix(&[1.0; 7], 0.15).unwrap(), 7);
        assert_eq!(causal_prefix(&[0.5, 0.5], 0.25).unwrap(), 2);
        assert!(causal_prefix(&[f32::NAN], 0.15).is_err());
        assert!(causal_prefix(&[1.01], 0.15).is_err());
        assert!(causal_prefix(&[0.9], 0.0).is_err());
    }

    #[test]
    fn proposal_keeps_progress_and_verification_consumes_it_once() {
        let mut s = engine();
        let handle = ready(&mut s);
        let (fence, record) = propose(&mut s, handle);
        assert!(
            s.complete_proposal(fence, &[record], Default::default(), &[f32::NAN; 7])
                .is_err()
        );
        assert!(s.submitted_step(fence).is_ok());
        let output = s
            .complete_proposal(
                fence,
                &[record],
                Default::default(),
                &[0.9, 0.8, 0.3, 0.1, 1.0, 1.0, 1.0],
            )
            .unwrap();
        assert!(output.is_empty());
        assert_eq!(s.request(handle).unwrap().consumed_tokens, 65);
        assert_eq!(s.request(handle).unwrap().generated_tokens, 1);
        let step = s.plan().unwrap();
        assert_eq!(
            (step.kind, step.token_capacity, step.lanes[0].tokens),
            (GraphKind::Verification, 4, 4)
        );
        assert!(s.commit(&step, Fence(fence.0 + 2)).is_err());
        let verify_fence = Fence(fence.0 + 1);
        s.commit(&step, verify_fence).unwrap();
        assert!(s.commit(&step, Fence(fence.0 + 2)).is_err());
        let record = CompletionRecord {
            submission_sequence: verify_fence.0 as i64,
            consumed_tokens: 67,
            generated_tokens: 3,
            token_id: 4,
            vector: Some(TokenVector {
                emitted_count: 2,
                offered_drafts: 3,
                accepted_drafts: 1,
                verification_width: 4,
                tokens: [3, 4, -1, -1, -1, -1, -1, -1],
                ..Default::default()
            }),
            ..record
        };
        assert_eq!(
            s.complete(verify_fence, &[record], Default::default())
                .unwrap()
                .len(),
            2
        );
        assert_eq!(s.plan().unwrap().kind, GraphKind::Proposal);
    }

    #[test]
    fn cancellation_during_proposal_cannot_attach_drafts_to_reused_slot() {
        let mut s = engine();
        let handle = ready(&mut s);
        let (fence, record) = propose(&mut s, handle);
        s.cancel(handle).unwrap();
        assert!(!s.has_available_slot());
        s.complete_proposal(fence, &[record], Default::default(), &[1.0; 7])
            .unwrap();
        assert!(s.has_available_slot());
        let replacement = ready(&mut s);
        assert_eq!(replacement.slot, handle.slot);
        assert_ne!(replacement.generation, handle.generation);
        assert_eq!(s.plan().unwrap().kind, GraphKind::Proposal);
        assert!(
            s.complete_proposal(fence, &[record], Default::default(), &[1.0; 7])
                .is_err()
        );
    }
}
