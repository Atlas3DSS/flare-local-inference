use std::{marker::PhantomData, rc::Rc};

use crate::{
    Result, Shutdown,
    cache::Fence,
    cuda::{Event, PinnedHost, Stream},
};

use super::{
    COMPLETION_BYTES, ExecutionManifest, GraphKind, MAX_OUTPUT_TOKENS, REQUEST_LANES,
    VECTOR_RECORD_FIELDS,
};

pub struct Upload<'a> {
    pub buffer: &'a str,
    pub offset: usize,
    pub bytes: &'a [u8],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TokenVector {
    pub emitted_count: i64,
    pub offered_drafts: i64,
    pub accepted_drafts: i64,
    pub copy_offered_mask: i64,
    pub candidate_consumed: i64,
    pub draft_cache_consumed: i64,
    pub verification_width: i64,
    pub tokens: [i64; MAX_OUTPUT_TOKENS],
}
impl Default for TokenVector {
    fn default() -> Self {
        Self {
            emitted_count: 0,
            offered_drafts: 0,
            accepted_drafts: 0,
            copy_offered_mask: 0,
            candidate_consumed: -1,
            draft_cache_consumed: 0,
            verification_width: 1,
            tokens: [-1; MAX_OUTPUT_TOKENS],
        }
    }
}
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CompletionRecord {
    pub generation: i64,
    pub submission_sequence: i64,
    pub consumed_tokens: i64,
    pub generated_tokens: i64,
    pub token_id: i64,
    pub status: i64,
    pub error_code: i64,
    pub reserved: i64,
    pub vector: Option<TokenVector>,
}

impl CompletionRecord {
    pub(crate) fn validate_vector(&self) -> Result<()> {
        if self.status != 0 {
            if let Some(v) = self.vector {
                if !(1..=8).contains(&v.verification_width)
                    || !(0..=v.verification_width).contains(&v.emitted_count)
                    || !(0..v.verification_width).contains(&v.offered_drafts)
                    || v.accepted_drafts < v.emitted_count.saturating_sub(1).max(0)
                    || v.accepted_drafts > v.emitted_count
                    || v.accepted_drafts > v.offered_drafts
                    || v.copy_offered_mask < 0
                    || v.copy_offered_mask >= (1 << v.offered_drafts)
                    || v.draft_cache_consumed < -1
                    || (v.candidate_consumed != -1
                        && (v.candidate_consumed <= 0
                            || v.candidate_consumed % 64 != 0
                            || v.candidate_consumed > self.consumed_tokens))
                    || v.tokens[..v.emitted_count as usize]
                        .iter()
                        .any(|id| u32::try_from(*id).is_err())
                    || v.tokens[v.emitted_count as usize..]
                        .iter()
                        .any(|id| *id != -1)
                    || self.token_id
                        != if v.emitted_count == 0 {
                            -1
                        } else {
                            v.tokens[v.emitted_count as usize - 1]
                        }
                    || (self.status == 3 && v.emitted_count != 0)
                {
                    return Err("invalid bounded token-vector completion".into());
                }
            }
        }
        Ok(())
    }

    pub fn fields(&self) -> Vec<i64> {
        let mut values = vec![
            self.generation,
            self.submission_sequence,
            self.consumed_tokens,
            self.generated_tokens,
            self.token_id,
            self.status,
            self.error_code,
            self.reserved,
        ];
        if let Some(v) = self.vector {
            values.extend([
                v.emitted_count,
                v.offered_drafts,
                v.accepted_drafts,
                v.copy_offered_mask,
                v.candidate_consumed,
                v.draft_cache_consumed,
                0,
                v.verification_width,
            ]);
            values.extend(v.tokens);
        }
        values
    }
}
#[derive(Debug)]
pub struct FrameCompletion {
    pub fence: Fence,
    pub graph_index: usize,
    pub elapsed_ns: Option<u64>,
    pub records: [CompletionRecord; REQUEST_LANES],
}

#[derive(Clone, Copy)]
struct Pending {
    fence: Fence,
    graph_index: usize,
    grammar: bool,
    timed: bool,
}

struct Frame {
    input: PinnedHost,
    output: PinnedHost,
    produced: Event,
    started: Option<Event>,
    transferred: Event,
    pending: Option<Pending>,
}

struct Resources {
    model: Stream,
    copy: Stream,
    frames: [Frame; 2],
}

/// Thread-bound executor over one borrowed prepared model and its CUDA context.
pub struct GraphExecutor {
    manifest: ExecutionManifest,
    resources: Option<Resources>,
    next_sequence: u64,
    completed: Fence,
    poison: Option<String>,
    shutdown: Option<Shutdown>,
    _thread_bound: PhantomData<Rc<()>>,
}

impl GraphExecutor {
    /// # Safety
    /// All manifest handles and allocations must belong to the given context,
    /// remain stable, and obey native/manifest.md. The caller retains the owner
    /// until close confirms both streams drained. No other executor may mutate
    /// these inputs, graph executables, scratch or completion rows concurrently.
    pub unsafe fn new(manifest: ExecutionManifest) -> Result<Self> {
        manifest.validate()?;
        let model = Stream::open(manifest.context, manifest.device)?;
        let copy = Stream::open(manifest.context, manifest.device)?;
        let timed = manifest
            .graphs
            .iter()
            .any(|g| g.kind == GraphKind::Speculative);
        let frame = || -> Result<Frame> {
            Ok(Frame {
                input: model.host_buffer(manifest.input_frame_bytes)?,
                output: model.host_buffer(manifest.completion_bytes())?,
                produced: model.event_with_timing(timed)?,
                started: timed.then(|| model.event_with_timing(true)).transpose()?,
                transferred: copy.event()?,
                pending: None,
            })
        };
        let frames = [frame()?, frame()?];
        Ok(Self {
            manifest,
            resources: Some(Resources {
                model,
                copy,
                frames,
            }),
            next_sequence: 1,
            completed: Fence(0),
            poison: None,
            shutdown: None,
            _thread_bound: PhantomData,
        })
    }

    pub fn manifest(&self) -> &ExecutionManifest {
        &self.manifest
    }
    pub fn completed_fence(&self) -> Fence {
        self.completed
    }
    /// Fence to record on resources before the next successful submission attempt.
    /// A failed CUDA enqueue poisons the executor; shutdown must drain before reuse.
    pub fn next_fence(&self) -> Result<Fence> {
        if self.resources.is_none() || self.poison.is_some() {
            return Err("executor is closed or failed".into());
        }
        if self.next_sequence > i64::MAX as u64 {
            return Err("submission sequence exhausted".into());
        }
        Ok(Fence(self.next_sequence))
    }
    pub fn can_submit(&self, grammar_constrained: bool) -> bool {
        let Some(resources) = &self.resources else {
            return false;
        };
        self.poison.is_none()
            && resources.frames.iter().any(|frame| frame.pending.is_none())
            && !resources
                .frames
                .iter()
                .filter_map(|frame| frame.pending)
                .any(|pending| pending.grammar)
            && (!grammar_constrained
                || resources.frames.iter().all(|frame| frame.pending.is_none()))
    }
    pub fn outstanding(&self) -> usize {
        self.resources.as_ref().map_or(0, |resources| {
            resources
                .frames
                .iter()
                .filter(|frame| frame.pending.is_some())
                .count()
        })
    }

    /// Returns None on frame or grammar backpressure without changing CUDA state.
    pub fn submit(
        &mut self,
        graph_index: usize,
        uploads: &[Upload<'_>],
        grammar_constrained: bool,
    ) -> Result<Option<Fence>> {
        if let Some(error) = &self.poison {
            return Err(format!("executor failed: {error}"));
        }
        let resources = self.resources.as_mut().ok_or("executor is closed")?;
        let graph = self
            .manifest
            .graphs
            .get(graph_index)
            .ok_or("graph index out of range")?;
        let grammar_constrained = grammar_constrained || graph.kind == GraphKind::Speculative;
        validate_uploads(&self.manifest, graph_index, uploads)?;
        if (resources.frames.iter().any(|frame| frame.pending.is_some()) && grammar_constrained)
            || resources
                .frames
                .iter()
                .filter_map(|frame| frame.pending)
                .any(|pending| pending.grammar)
        {
            return Ok(None);
        }
        let Some(index) = resources
            .frames
            .iter()
            .position(|frame| frame.pending.is_none())
        else {
            return Ok(None);
        };
        if self.next_sequence > i64::MAX as u64 {
            return Err("submission sequence exhausted".into());
        }
        let fence = Fence(self.next_sequence);
        let frame = &mut resources.frames[index];
        for upload in uploads {
            let binding = graph
                .input_bindings
                .iter()
                .find(|binding| binding.buffer_name == upload.buffer)
                .unwrap();
            let start = binding.host_offset + upload.offset;
            frame.input.bytes_mut()[start..start + upload.bytes.len()]
                .copy_from_slice(upload.bytes);
        }
        let frame_bytes = (index as i32).to_le_bytes();
        let sequence_bytes = (fence.0 as i64).to_le_bytes();
        for (name, bytes) in [
            ("frame_index", frame_bytes.as_slice()),
            ("submission_sequence", sequence_bytes.as_slice()),
        ] {
            let binding = graph
                .input_bindings
                .iter()
                .find(|binding| binding.buffer_name == name)
                .unwrap();
            frame.input.bytes_mut()[binding.host_offset..binding.host_offset + bytes.len()]
                .copy_from_slice(bytes);
        }
        frame.pending = Some(Pending {
            fence,
            graph_index,
            grammar: grammar_constrained,
            timed: graph.kind == GraphKind::Speculative,
        });
        self.next_sequence += 1;
        let operation = || -> Result<()> {
            for upload in uploads {
                let binding = graph
                    .input_bindings
                    .iter()
                    .find(|binding| binding.buffer_name == upload.buffer)
                    .unwrap();
                let buffer = self.manifest.buffer(upload.buffer)?;
                unsafe {
                    resources.model.upload(
                        buffer.address + upload.offset,
                        &frame.input,
                        binding.host_offset + upload.offset,
                        upload.bytes.len(),
                    )?;
                }
            }
            for name in ["frame_index", "submission_sequence"] {
                let binding = graph
                    .input_bindings
                    .iter()
                    .find(|binding| binding.buffer_name == name)
                    .unwrap();
                let buffer = self.manifest.buffer(name)?;
                unsafe {
                    resources.model.upload(
                        buffer.address,
                        &frame.input,
                        binding.host_offset,
                        buffer.nbytes,
                    )?;
                }
            }
            if graph.kind == GraphKind::Speculative {
                resources.model.record(frame.started.as_ref().unwrap())?;
            }
            resources.model.launch(graph.handle)?;
            resources.model.record(&frame.produced)?;
            resources.copy.wait(&frame.produced)?;
            let address = self.manifest.buffer("completion_records")?.address
                + index * self.manifest.completion_bytes();
            unsafe {
                resources.copy.download(
                    &frame.output,
                    address,
                    self.manifest.completion_bytes(),
                )?;
            }
            resources.copy.record(&frame.transferred)
        };
        if let Err(error) = operation() {
            self.poison = Some(error.clone());
            return Err(error);
        }
        Ok(Some(fence))
    }

    pub fn poll(&mut self) -> Result<Option<FrameCompletion>> {
        if let Some(error) = &self.poison {
            return Err(format!("executor failed: {error}"));
        }
        let resources = self.resources.as_mut().ok_or("executor is closed")?;
        let frame = resources
            .frames
            .iter_mut()
            .filter(|frame| frame.pending.is_some())
            .min_by_key(|frame| frame.pending.unwrap().fence);
        let Some(frame) = frame else {
            return Ok(None);
        };
        match frame.transferred.completed() {
            Ok(false) => return Ok(None),
            Err(error) => {
                self.poison = Some(error.clone());
                return Err(error);
            }
            Ok(true) => {}
        }
        let pending = frame.pending.take().unwrap();
        self.completed = pending.fence;
        let records = match decode_records(frame.output.bytes(), pending.fence) {
            Ok(records) => records,
            Err(error) => {
                self.poison = Some(error.clone());
                return Err(error);
            }
        };
        let elapsed_ns = if pending.timed {
            match frame.produced.elapsed_ns(frame.started.as_ref().unwrap()) {
                Ok(value) => Some(value),
                Err(error) => {
                    self.poison = Some(error.clone());
                    return Err(error);
                }
            }
        } else {
            None
        };
        Ok(Some(FrameCompletion {
            fence: pending.fence,
            graph_index: pending.graph_index,
            elapsed_ns,
            records,
        }))
    }

    pub fn close(&mut self) -> Shutdown {
        if let Some(shutdown) = &self.shutdown {
            return shutdown.clone();
        }
        let resources = self.resources.take().unwrap();
        let compute = resources.model.synchronize();
        let copy = resources.copy.synchronize();
        let drained = compute.is_ok() && copy.is_ok();
        let error = compute.err().or(copy.err()).or(self.poison.take());
        if drained {
            drop(resources);
        } else {
            std::mem::forget(resources);
        }
        let shutdown = Shutdown { drained, error };
        self.shutdown = Some(shutdown.clone());
        shutdown
    }
}

impl Drop for GraphExecutor {
    fn drop(&mut self) {
        self.close();
    }
}

pub(super) fn validate_uploads(
    manifest: &ExecutionManifest,
    graph_index: usize,
    uploads: &[Upload<'_>],
) -> Result<()> {
    let graph = manifest
        .graphs
        .get(graph_index)
        .ok_or("graph index out of range")?;
    for upload in uploads {
        if ["frame_index", "submission_sequence"].contains(&upload.buffer) {
            return Err("reserved executor inputs cannot be overridden".into());
        }
        let binding = graph
            .input_bindings
            .iter()
            .find(|binding| binding.buffer_name == upload.buffer)
            .ok_or("upload is not bound by this graph")?;
        let buffer = manifest.buffer(upload.buffer)?;
        if upload.bytes.is_empty()
            || upload.offset % buffer.dtype.size() != 0
            || upload.bytes.len() % buffer.dtype.size() != 0
            || upload
                .offset
                .checked_add(upload.bytes.len())
                .is_none_or(|end| end > binding.nbytes)
        {
            return Err(format!(
                "unaligned or out-of-range upload: {}",
                upload.buffer
            ));
        }
    }
    // A shared host input frame cannot simultaneously hold conflicting upload values.
    for (index, first) in uploads.iter().enumerate() {
        for second in &uploads[index + 1..] {
            if first.buffer == second.buffer
                && first.offset < second.offset + second.bytes.len()
                && second.offset < first.offset + first.bytes.len()
            {
                return Err("uploads overlap within one input buffer".into());
            }
        }
    }
    Ok(())
}

pub(super) fn decode_records(
    bytes: &[u8],
    fence: Fence,
) -> Result<[CompletionRecord; REQUEST_LANES]> {
    let vector = bytes.len() == REQUEST_LANES * VECTOR_RECORD_FIELDS * 8;
    if !vector && bytes.len() != COMPLETION_BYTES {
        return Err("completion extent mismatch".into());
    }
    let mut records = [CompletionRecord::default(); REQUEST_LANES];
    for (record, bytes) in records.iter_mut().zip(bytes.chunks_exact(if vector {
        VECTOR_RECORD_FIELDS * 8
    } else {
        64
    })) {
        let mut fields = [0i64; VECTOR_RECORD_FIELDS];
        for (field, bytes) in fields.iter_mut().zip(bytes.chunks_exact(8)) {
            *field = i64::from_le_bytes(bytes.try_into().unwrap());
        }
        *record = CompletionRecord {
            generation: fields[0],
            submission_sequence: fields[1],
            consumed_tokens: fields[2],
            generated_tokens: fields[3],
            token_id: fields[4],
            status: fields[5],
            error_code: fields[6],
            reserved: fields[7],
            vector: vector.then(|| TokenVector {
                emitted_count: fields[8],
                offered_drafts: fields[9],
                accepted_drafts: fields[10],
                copy_offered_mask: fields[11],
                candidate_consumed: fields[12],
                draft_cache_consumed: fields[13],
                verification_width: fields[15],
                tokens: fields[16..24].try_into().unwrap(),
            }),
        };
        if vector && record.status != 0 && fields[14] != 0 {
            return Err("reserved vector completion field must be zero".into());
        }
        record.validate_vector()?;
        if !(0..=3).contains(&record.status)
            || record.reserved != 0
            || (record.status != 0
                && (record.generation <= 0
                    || record.submission_sequence != fence.0 as i64
                    || record.consumed_tokens < 0
                    || record.generated_tokens < 0
                    || record.token_id < -1
                    || (record.status != 3 && record.error_code != 0)))
        {
            return Err("invalid or stale device completion record".into());
        }
    }
    Ok(records)
}
