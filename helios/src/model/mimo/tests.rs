use super::*;
use crate::{
    cache::Fence,
    model::{Admission, SamplingParameters, pack_inputs},
    scheduler::{RequestScheduler, RequestSpec, SchedulerLimits},
};

#[test]
fn mixed_prefill_vision_indices_follow_packed_decode_tokens() {
    for width in [1, 3, 8] {
        let config = ModelConfig {
            prepared_sequence_tokens: 2048,
            contract: MiMo::metadata(),
            identity: [0; 32],
            model_id: "mimo".into(),
            limits: SchedulerLimits {
                max_active: 2,
                request_slots: 2,
                max_sequence_tokens: 2048,
                prefill_chunk_size: 128,
                pages: 64,
                snapshots: 8,
            },
            eos_token_ids: vec![1],
            sampling_defaults: SamplingParameters::default(),
            max_new_tokens: 16,
            speculative_widths: if width == 1 { vec![] } else { vec![1, width] },
            speculative_fixed_width: None,
            history_drafts: false,
            speculative_proposal_source: crate::scheduler::speculation::ProposalSource::Mtp,
            state_rows_per_request: Some(1),
        };
        let mut scheduler = RequestScheduler::new(
            config.identity,
            config.limits,
            MiMo::scheduler_capabilities(&config),
        )
        .unwrap();
        scheduler
            .enable_speculation(&config.speculative_widths)
            .unwrap();
        scheduler.mixed_graphs(&[(2, width, 128)]).unwrap();
        let graphs: Vec<_> = [
            (GraphKind::Init, 1, 1, 0),
            (GraphKind::Prefill, 1, 128, 0),
            (GraphKind::Snapshot, 1, 1, 0),
            (GraphKind::Decode, 1, 1, 0),
            (GraphKind::Speculative, 1, width, 0),
            (GraphKind::Mixed, 2, width, 128),
        ]
        .into_iter()
        .map(|(kind, batch_capacity, token_capacity, prefill_capacity)| {
            crate::execution::GraphDescriptor {
                name: format!("{kind:?}"),
                kind,
                batch_capacity,
                token_capacity,
                prefill_capacity,
                estimated_duration_ns: 0,
                sampling_mode: crate::execution::SamplingMode::General,
                handle: 1,
                input_bindings: vec![],
            }
        })
        .collect();
        scheduler.measured_graph_durations(&graphs).unwrap();
        let mut admissions: [Option<Admission<MiMo>>; 2] = [None, None];
        let mut admit = |scheduler: &mut RequestScheduler, prompt: Vec<u32>, indices: Vec<i32>| {
            let handle = scheduler
                .admit(RequestSpec {
                    prompt: prompt.clone(),
                    system_boundary: None,
                    conditioning: vec![],
                    max_new_tokens: 16,
                    grammar_constrained: false,
                    host_stops: false,
                    growable_output: false,
                })
                .unwrap();
            let admission = Admission {
                handle,
                sampling: SamplingParameters::default(),
                max_new_tokens: 16,
                prompt,
                data: PromptData {
                    vision_token_indices: indices,
                },
                grammar_mask: None,
                ignore_eos: false,
            };
            admission.validate(&config).unwrap();
            admissions[handle.slot as usize] = Some(admission);
            handle
        };
        let decode = admit(&mut scheduler, vec![7; 65], vec![-1; 65]);
        for (fence, kind, consumed, generated, token) in [
            (1, GraphKind::Init, 0, 0, -1),
            (2, GraphKind::Prefill, 65, 1, 7),
            (3, GraphKind::Snapshot, 65, 1, -1),
        ] {
            let plan = scheduler.phase_plan(kind, Some(decode));
            scheduler.commit(&plan, Fence(fence)).unwrap();
            let mut records = [CompletionRecord::default(); 2];
            records[decode.slot as usize] = CompletionRecord {
                generation: decode.generation as i64,
                submission_sequence: fence as i64,
                consumed_tokens: consumed,
                generated_tokens: generated,
                token_id: token,
                status: 1,
                ..Default::default()
            };
            scheduler
                .complete(Fence(fence), &records, Default::default())
                .unwrap();
        }
        let prefill = admit(&mut scheduler, vec![IMAGE_TOKEN; 129], (0..129).collect());
        let init = scheduler.phase_plan(GraphKind::Init, Some(prefill));
        scheduler.commit(&init, Fence(4)).unwrap();
        let mut records = [CompletionRecord::default(); 2];
        records[prefill.slot as usize] = CompletionRecord {
            generation: prefill.generation as i64,
            submission_sequence: 4,
            token_id: -1,
            status: 1,
            ..Default::default()
        };
        scheduler
            .complete(Fence(4), &records, Default::default())
            .unwrap();
        let plan = scheduler.plan().unwrap();
        assert_eq!(plan.kind, GraphKind::Mixed);
        assert_eq!(plan.mixed.as_ref().unwrap().prefill_row, Some(1));
        scheduler.commit(&plan, Fence(5)).unwrap();
        let mut inputs = inputs::Inputs::default();
        pack_inputs::<MiMo>(&mut inputs, &config, &scheduler, &plan, &admissions).unwrap();
        let uploads = inputs.uploads();
        let indices = uploads
            .iter()
            .find(|u| u.buffer == "vision_token_indices")
            .unwrap();
        assert_eq!(indices.offset, width * 4);
        let values: Vec<_> = indices
            .bytes
            .chunks_exact(4)
            .map(|v| i32::from_le_bytes(v.try_into().unwrap()))
            .collect();
        assert_eq!(values, (0..128).collect::<Vec<_>>());
        assert!(
            MiMo::required_inputs(&config, GraphKind::Mixed).any(|n| n == "vision_token_indices")
        );
        assert!(
            !MiMo::required_inputs(&config, GraphKind::Speculative)
                .any(|n| n == "vision_token_indices")
        );
    }
}
