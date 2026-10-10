use std::{
    fmt::Write,
    sync::atomic::{AtomicU64, Ordering},
};

use crate::{
    execution::{CompletionRecord, GraphKind},
    scheduler::SchedulerStats,
};

#[derive(Default)]
struct RoundMetrics {
    rounds: AtomicU64,
    gpu_ns: AtomicU64,
    emitted: AtomicU64,
    offered: AtomicU64,
    accepted: AtomicU64,
}

#[derive(Clone, Copy, Default)]
pub(super) struct Snapshot {
    pub prefill_tokens: u64,
    pub generated_tokens: u64,
    pub draft_steps: u64,
    pub verify_steps: u64,
    pub accepted_drafts: u64,
    pub draft_rounds: u64,
    pub active: u64,
    pub pending: u64,
    pub completed: u64,
    pub failed: u64,
    pub reused_tokens: u64,
    pub cached_pages: u64,
    pub free_pages: u64,
}

#[derive(Default)]
pub(super) struct Metrics {
    prefill_tokens: AtomicU64,
    generated_tokens: AtomicU64,
    accepted_drafts: AtomicU64,
    draft_rounds: AtomicU64,
    pub admitted: AtomicU64,
    pub completed: AtomicU64,
    pub cancelled: AtomicU64,
    pub failed: AtomicU64,
    launches: [AtomicU64; 8],
    sampling_launches: [AtomicU64; 4],
    speculative: [[RoundMetrics; 8]; 3],
    draft_sources: [[[AtomicU64; 2]; 2]; 7],
    mixed_grammar_decode_steps: AtomicU64,
    constrained_max_inflight: AtomicU64,
    cache_lookups: AtomicU64,
    cache_hits: AtomicU64,
    cache_misses: AtomicU64,
    reused_tokens: AtomicU64,
    snapshot_publications: AtomicU64,
    snapshot_evictions: AtomicU64,
    system_checkpoints: AtomicU64,
    prompt_checkpoints: AtomicU64,
    active_requests: AtomicU64,
    free_slots: AtomicU64,
    retiring_slots: AtomicU64,
    free_pages: AtomicU64,
    free_snapshots: AtomicU64,
    cached_checkpoints: AtomicU64,
    cached_pages: AtomicU64,
    pub pending: AtomicU64,
}

impl Metrics {
    pub fn progress(&self, prefill: u64, generated: usize, accepted: u64, rounds: u64) {
        for (counter, value) in [
            (&self.prefill_tokens, prefill),
            (&self.generated_tokens, generated as u64),
            (&self.accepted_drafts, accepted),
            (&self.draft_rounds, rounds),
        ] {
            if value != 0 {
                counter.fetch_add(value, Ordering::Relaxed);
            }
        }
    }

    pub fn snapshot(&self, learned_drafter: bool) -> Snapshot {
        let mut snapshot = Snapshot {
            prefill_tokens: self.prefill_tokens.load(Ordering::Relaxed),
            generated_tokens: self.generated_tokens.load(Ordering::Relaxed),
            accepted_drafts: self.accepted_drafts.load(Ordering::Relaxed),
            draft_rounds: self.draft_rounds.load(Ordering::Relaxed),
            active: self.active_requests.load(Ordering::Relaxed),
            pending: self.pending.load(Ordering::Relaxed),
            completed: self.completed.load(Ordering::Relaxed),
            failed: self.failed.load(Ordering::Relaxed),
            reused_tokens: self.reused_tokens.load(Ordering::Relaxed),
            cached_pages: self.cached_pages.load(Ordering::Relaxed),
            free_pages: self.free_pages.load(Ordering::Relaxed),
            ..Snapshot::default()
        };
        for widths in &self.speculative {
            for (drafts, counts) in widths.iter().enumerate() {
                let rounds = counts.rounds.load(Ordering::Relaxed);
                snapshot.verify_steps += rounds;
                if learned_drafter {
                    snapshot.draft_steps += rounds * drafts as u64;
                }
            }
        }
        snapshot
    }

    pub fn launched(&self, kind: GraphKind) {
        let index = match kind {
            GraphKind::Init => 0,
            GraphKind::Restore => 1,
            GraphKind::Resume => 2,
            GraphKind::Prefill => 3,
            GraphKind::Decode => 4,
            GraphKind::Vision => 5,
            GraphKind::Snapshot => 6,
            GraphKind::Speculative => 7,
        };
        self.launches[index].fetch_add(1, Ordering::Relaxed);
        let sampling_index = match kind {
            GraphKind::Prefill => 0,
            GraphKind::Decode => 1,
            GraphKind::Resume => 2,
            GraphKind::Speculative => 3,
            _ => return,
        };
        self.sampling_launches[sampling_index].fetch_add(1, Ordering::Relaxed);
    }

    pub fn speculative_completed(
        &self,
        batch: usize,
        width: usize,
        elapsed_ns: u64,
        records: &[CompletionRecord; 4],
    ) {
        let counters = &self.speculative[batch.trailing_zeros() as usize][width - 1];
        counters.rounds.fetch_add(1, Ordering::Relaxed);
        counters.gpu_ns.fetch_add(elapsed_ns, Ordering::Relaxed);
        for record in records.iter().filter(|r| r.status != 0) {
            let Some(v) = record.vector else {
                continue;
            };
            counters
                .emitted
                .fetch_add(v.emitted_count as u64, Ordering::Relaxed);
            counters
                .offered
                .fetch_add(v.offered_drafts as u64, Ordering::Relaxed);
            counters
                .accepted
                .fetch_add(v.accepted_drafts as u64, Ordering::Relaxed);
            for depth in 0..v.offered_drafts as usize {
                let source = usize::from(v.copy_offered_mask & (1 << depth) != 0);
                self.draft_sources[depth][source][0].fetch_add(1, Ordering::Relaxed);
                if depth < v.accepted_drafts as usize {
                    self.draft_sources[depth][source][1].fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    }

    pub fn constrained_submission(&self, outstanding: usize, mixed_decode: bool) {
        self.constrained_max_inflight
            .fetch_max(outstanding as u64, Ordering::Relaxed);
        if mixed_decode {
            self.mixed_grammar_decode_steps
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn scheduler(&self, stats: SchedulerStats) {
        for (target, value) in [
            (&self.cache_lookups, stats.cache_lookups),
            (&self.cache_hits, stats.cache_hits),
            (&self.cache_misses, stats.cache_lookups - stats.cache_hits),
            (&self.reused_tokens, stats.reused_tokens),
            (&self.snapshot_publications, stats.snapshot_publications),
            (&self.snapshot_evictions, stats.snapshot_evictions),
            (&self.system_checkpoints, stats.system_checkpoints),
            (&self.prompt_checkpoints, stats.prompt_checkpoints),
            (&self.active_requests, stats.active_requests as u64),
            (&self.free_slots, stats.free_slots as u64),
            (&self.retiring_slots, stats.retiring_slots as u64),
            (&self.free_pages, stats.free_pages as u64),
            (&self.free_snapshots, stats.free_snapshots as u64),
            (&self.cached_checkpoints, stats.cached_checkpoints as u64),
            (&self.cached_pages, stats.cached_pages as u64),
        ] {
            target.store(value, Ordering::Relaxed);
        }
    }

    pub fn render(&self) -> String {
        let mut text = String::with_capacity(4096);
        text.push_str("# HELP flare_sampling_graph_launches_total Successful native submissions by captured sampler mode; includes consumption-only prefill.\n# TYPE flare_sampling_graph_launches_total counter\n");
        for (kind, count) in ["prefill", "decode", "resume", "speculative"]
            .into_iter()
            .zip(&self.sampling_launches)
        {
            let _ = writeln!(
                text,
                "flare_sampling_graph_launches_total{{kind=\"{kind}\",mode=\"general\"}} {}",
                count.load(Ordering::Relaxed)
            );
        }
        for (metric, help) in [
            (
                "rounds_total",
                "Completed speculative graphs, including terminal no-op rounds.",
            ),
            (
                "gpu_nanoseconds_total",
                "CUDA event duration of completed full speculative graphs; excludes input upload and completion transfer.",
            ),
            (
                "emitted_tokens_total",
                "Raw emitted tokens in completed speculative records, including lanes cancelled after submission.",
            ),
            (
                "offered_drafts_total",
                "Drafts offered in completed speculative records.",
            ),
            (
                "accepted_drafts_total",
                "Accepted draft tokens after EOS and output-budget clipping.",
            ),
        ] {
            let _ = writeln!(
                text,
                "# HELP flare_speculative_{metric} {help}\n# TYPE flare_speculative_{metric} counter"
            );
            for (batch, widths) in [1, 2, 4].into_iter().zip(&self.speculative) {
                for (width, counts) in (1..=8).zip(widths) {
                    let value = match metric {
                        "rounds_total" => &counts.rounds,
                        "gpu_nanoseconds_total" => &counts.gpu_ns,
                        "emitted_tokens_total" => &counts.emitted,
                        "offered_drafts_total" => &counts.offered,
                        _ => &counts.accepted,
                    }
                    .load(Ordering::Relaxed);
                    let _ = writeln!(
                        text,
                        "flare_speculative_{metric}{{batch=\"{batch}\",width=\"{width}\"}} {value}"
                    );
                }
            }
        }
        text.push_str("# HELP flare_speculative_drafts_total Draft outcomes by one-based depth and actual proposal source; accepted counts are clipped at EOS/output budget.\n# TYPE flare_speculative_drafts_total counter\n");
        for (index, sources) in self.draft_sources.iter().enumerate() {
            let depth = index + 1;
            for (source, outcomes) in ["mtp", "copy"].into_iter().zip(sources) {
                for (outcome, count) in ["offered", "accepted"].into_iter().zip(outcomes) {
                    let value = count.load(Ordering::Relaxed);
                    let _ = writeln!(
                        text,
                        "flare_speculative_drafts_total{{depth=\"{depth}\",source=\"{source}\",outcome=\"{outcome}\"}} {value}"
                    );
                }
            }
        }
        for (name, help, value) in [
            (
                "prefill_tokens_total",
                "Prompt tokens consumed by successful completed prefill work, excluding padding and prefix reuse.",
                &self.prefill_tokens,
            ),
            (
                "generated_tokens_total",
                "Committed generated tokens, including EOS and tokens subsequently withheld by text stops.",
                &self.generated_tokens,
            ),
            (
                "accepted_draft_tokens_total",
                "Accepted draft tokens from progressing live request rounds, excluding the bonus token.",
                &self.accepted_drafts,
            ),
            (
                "draft_request_rounds_total",
                "Progressing live request rounds offering at least one draft; denominator for mean acceptance length.",
                &self.draft_rounds,
            ),
            (
                "requests_admitted_total",
                "Requests admitted into a physical model slot.",
                &self.admitted,
            ),
            (
                "requests_completed_total",
                "Requests whose model generation completed successfully, including explicit token stops.",
                &self.completed,
            ),
            (
                "requests_cancelled_total",
                "Queued or active model requests cancelled before completion.",
                &self.cancelled,
            ),
            (
                "requests_failed_total",
                "Model requests failed by admission, execution, grammar, or output backpressure.",
                &self.failed,
            ),
            (
                "mixed_grammar_decode_steps_total",
                "Successful decode submissions containing both grammar-constrained and unconstrained request lanes.",
                &self.mixed_grammar_decode_steps,
            ),
            (
                "cache_lookups_total",
                "Prefix lookups during model admission, including admissions rejected for resource pressure.",
                &self.cache_lookups,
            ),
            (
                "cache_hits_total",
                "Prefix lookups returning a GPU-ready checkpoint.",
                &self.cache_hits,
            ),
            (
                "cache_reused_tokens_total",
                "Tokens covered by checkpoints returned from prefix lookups.",
                &self.reused_tokens,
            ),
            (
                "snapshot_publications_total",
                "Unique checkpoints made visible after completed GPU writes.",
                &self.snapshot_publications,
            ),
            (
                "system_checkpoints_total",
                "Exact rendered system endpoints retained after completed snapshot copies.",
                &self.system_checkpoints,
            ),
            (
                "prompt_checkpoints_total",
                "Exact prompt endpoints retained after completed snapshot copies.",
                &self.prompt_checkpoints,
            ),
            (
                "snapshot_evictions_total",
                "Checkpoints logically evicted; GPU retirement may still be pending.",
                &self.snapshot_evictions,
            ),
        ] {
            metric(
                &mut text,
                name,
                "counter",
                help,
                value.load(Ordering::Relaxed),
            );
        }
        metric(
            &mut text,
            "cache_misses_total",
            "counter",
            "Prefix lookups without a GPU-ready checkpoint.",
            self.cache_misses.load(Ordering::Relaxed),
        );
        text.push_str("# HELP flare_graph_launches_total Successful native CUDA graph submissions.\n# TYPE flare_graph_launches_total counter\n");
        for (kind, count) in [
            "init",
            "restore",
            "resume",
            "prefill",
            "decode",
            "vision",
            "snapshot",
            "speculative",
        ]
        .into_iter()
        .zip(&self.launches)
        {
            let _ = writeln!(
                text,
                "flare_graph_launches_total{{kind=\"{kind}\"}} {}",
                count.load(Ordering::Relaxed)
            );
        }
        for (name, help, value) in [
            (
                "constrained_max_inflight",
                "Maximum outstanding executor frames immediately after a successful grammar-constrained graph submission; zero until the first such submission.",
                &self.constrained_max_inflight,
            ),
            (
                "active_requests",
                "Physical model slots with a live request.",
                &self.active_requests,
            ),
            (
                "free_request_slots",
                "Physical slots available for admission.",
                &self.free_slots,
            ),
            (
                "retiring_request_slots",
                "Cancelled slots waiting for GPU completion.",
                &self.retiring_slots,
            ),
            (
                "free_kv_pages",
                "KV pages reusable after completed GPU fences.",
                &self.free_pages,
            ),
            (
                "free_snapshots",
                "Immutable state slots reusable after completed GPU fences.",
                &self.free_snapshots,
            ),
            (
                "cached_checkpoints",
                "GPU-ready immutable checkpoints.",
                &self.cached_checkpoints,
            ),
            (
                "cached_kv_pages",
                "Pages owned by immutable checkpoints.",
                &self.cached_pages,
            ),
            (
                "pending_requests",
                "Requests waiting inside the model scheduler, excluding its ingress channel.",
                &self.pending,
            ),
        ] {
            metric(
                &mut text,
                name,
                "gauge",
                help,
                value.load(Ordering::Relaxed),
            );
        }
        text
    }
}

pub(super) fn metric(text: &mut String, name: &str, kind: &str, help: &str, value: u64) {
    let _ = writeln!(
        text,
        "# HELP flare_{name} {help}\n# TYPE flare_{name} {kind}\nflare_{name} {value}"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speculative_metrics_preserve_depth_source_and_clipped_progress() {
        let metrics = Metrics::default();
        let mut records = [CompletionRecord::default(); 4];
        records[2] = CompletionRecord {
            status: 2,
            vector: Some(crate::execution::TokenVector {
                verification_width: 8,
                emitted_count: 2,
                offered_drafts: 7,
                accepted_drafts: 2,
                copy_offered_mask: 0b101,
                ..Default::default()
            }),
            ..Default::default()
        };
        metrics.speculative_completed(1, 8, 1000, &records);
        let text = metrics.render();
        for line in [
            "flare_speculative_rounds_total{batch=\"1\",width=\"8\"} 1",
            "flare_speculative_gpu_nanoseconds_total{batch=\"1\",width=\"8\"} 1000",
            "flare_speculative_emitted_tokens_total{batch=\"1\",width=\"8\"} 2",
            "flare_speculative_offered_drafts_total{batch=\"1\",width=\"8\"} 7",
            "flare_speculative_drafts_total{depth=\"1\",source=\"copy\",outcome=\"accepted\"} 1",
            "flare_speculative_drafts_total{depth=\"2\",source=\"mtp\",outcome=\"accepted\"} 1",
            "flare_speculative_drafts_total{depth=\"3\",source=\"copy\",outcome=\"offered\"} 1",
            "flare_speculative_drafts_total{depth=\"3\",source=\"copy\",outcome=\"accepted\"} 0",
        ] {
            assert!(text.lines().any(|value| value == line), "{line}");
        }
    }

    #[test]
    fn counters_expose_fixed_names_without_request_payloads() {
        let metrics = Metrics::default();
        metrics.scheduler(SchedulerStats {
            cache_lookups: 5,
            cache_hits: 2,
            reused_tokens: 192,
            free_pages: 10,
            ..SchedulerStats::default()
        });
        metrics.launched(GraphKind::Restore);
        metrics.launched(GraphKind::Decode);
        metrics.launched(GraphKind::Decode);
        let text = metrics.render();
        assert!(text.contains("flare_cache_misses_total 3\n"));
        assert!(text.contains("flare_cache_reused_tokens_total 192\n"));
        assert!(text.contains("flare_graph_launches_total{kind=\"restore\"} 1\n"));
        assert!(!text.contains("mode=\"greedy\""));
        assert!(
            text.contains(
                "flare_sampling_graph_launches_total{kind=\"decode\",mode=\"general\"} 2\n"
            )
        );
        assert_eq!(
            text.lines()
                .filter(|line| line.starts_with("flare_graph_launches_total{"))
                .count(),
            8
        );
    }
}
