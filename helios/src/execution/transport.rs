use std::{marker::PhantomData, rc::Rc};

use crate::{
    Error, ExecutionError, Failure, Result, Shutdown,
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
    pub model_reserved: i64,
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
            model_reserved: -1,
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
                    return Err(Error::execution(
                        ExecutionError::Completion,
                        "invalid bounded token-vector completion",
                    ));
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
                v.model_reserved,
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
    resolved_uploads: Vec<ResolvedUpload>,
    next_sequence: u64,
    completed: Fence,
    poison: Option<Error>,
    shutdown: Option<Shutdown>,
    _thread_bound: PhantomData<Rc<()>>,
}

impl GraphExecutor {
    /// # Safety
    /// All manifest handles and allocations must belong to the given context,
    /// remain stable, and obey helios/manifest.md. The caller retains the owner
    /// until close confirms both streams drained. No other executor may mutate
    /// these inputs, graph executables, scratch or completion rows concurrently.
    pub unsafe fn new(manifest: ExecutionManifest) -> Result<Self> {
        manifest.validate()?;
        let upload_capacity = manifest
            .graphs
            .iter()
            .map(|graph| graph.input_bindings.len())
            .max()
            .unwrap_or(0)
            .saturating_mul(REQUEST_LANES);
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
            resolved_uploads: Vec::with_capacity(upload_capacity),
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
        if let Some(error) = &self.poison {
            return Err(ExecutionError::Poisoned(Failure::caused_by(
                "executor is closed or failed",
                error.clone(),
            ))
            .into());
        }
        if self.resources.is_none() {
            return Err(Error::execution(
                ExecutionError::Closed,
                "executor is closed or failed",
            ));
        }
        if self.next_sequence > i64::MAX as u64 {
            return Err(Error::execution(
                ExecutionError::Capacity,
                "submission sequence exhausted",
            ));
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

    /// Validate a coordinated submission without touching CUDA or advancing its fence.
    pub(super) fn validate_submission(
        &mut self,
        graph: usize,
        uploads: &[Upload<'_>],
        grammar: bool,
    ) -> Result<Option<Fence>> {
        let fence = self.next_fence()?;
        validate_uploads(&self.manifest, graph, uploads, &mut self.resolved_uploads)?;
        Ok(self
            .can_submit(grammar || self.manifest.graphs[graph].kind == GraphKind::Speculative)
            .then_some(fence))
    }

    /// Retain allocations after a peer failure that may strand a GPU collective.
    pub(super) fn abandon(&mut self, error: Error) -> Shutdown {
        if let Some(resources) = self.resources.take() {
            std::mem::forget(resources);
        }
        let shutdown = Shutdown {
            drained: false,
            error: Some(error),
        };
        self.shutdown = Some(shutdown.clone());
        shutdown
    }

    /// Returns None on frame or grammar backpressure without changing CUDA state.
    pub fn submit(
        &mut self,
        graph_index: usize,
        uploads: &[Upload<'_>],
        grammar_constrained: bool,
    ) -> Result<Option<Fence>> {
        if let Some(error) = &self.poison {
            return Err(ExecutionError::Poisoned(Failure::caused_by(
                format!("executor failed: {error}"),
                error.clone(),
            ))
            .into());
        }
        let resources = self
            .resources
            .as_mut()
            .ok_or_else(|| Error::execution(ExecutionError::Closed, "executor is closed"))?;
        let graph =
            self.manifest.graphs.get(graph_index).ok_or_else(|| {
                Error::execution(ExecutionError::State, "graph index out of range")
            })?;
        let grammar_constrained = grammar_constrained || graph.kind == GraphKind::Speculative;
        let resolved = validate_uploads(
            &self.manifest,
            graph_index,
            uploads,
            &mut self.resolved_uploads,
        )?;
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
            return Err(Error::execution(
                ExecutionError::Capacity,
                "submission sequence exhausted",
            ));
        }
        let fence = Fence(self.next_sequence);
        let frame = &mut resources.frames[index];
        resolved.stage(frame.input.bytes_mut(), index as i32, fence.0 as i64);
        frame.pending = Some(Pending {
            fence,
            graph_index,
            grammar: grammar_constrained,
            timed: graph.kind == GraphKind::Speculative,
        });
        self.next_sequence += 1;
        let operation = || -> Result<()> {
            for upload in resolved.copies() {
                // SAFETY: Validation resolved aligned device/host subranges within the
                // retained manifest allocations and this pinned frame. The frame is
                // marked pending before enqueue, remains immutable until completion,
                // and cannot be reused after an enqueue error until close drains both
                // context-bound streams (or intentionally retains their resources).
                unsafe {
                    resources.model.upload(
                        upload.device_address,
                        &frame.input,
                        upload.host_offset,
                        upload.nbytes,
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
            // SAFETY: The manifest bounds both completion rows; index selects one
            // row in that retained allocation. The copy stream waits for the model
            // event, and the pinned output is not read or reused before transferred
            // completes. Shutdown drains both streams before releasing either owner.
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
            return Err(ExecutionError::Poisoned(Failure::caused_by(
                format!("executor failed: {error}"),
                error.clone(),
            ))
            .into());
        }
        let resources = self
            .resources
            .as_mut()
            .ok_or_else(|| Error::execution(ExecutionError::Closed, "executor is closed"))?;
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
        let compute = resources.model.synchronize().map_err(Error::drain);
        let copy = resources.copy.synchronize().map_err(Error::drain);
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct ResolvedUpload {
    pub(super) host_offset: usize,
    pub(super) device_address: usize,
    pub(super) nbytes: usize,
}

/// Borrowed descriptors are consumed synchronously; CUDA retains only frame bytes.
pub(super) struct ResolvedUploads<'uploads, 'bytes, 'storage> {
    uploads: &'uploads [Upload<'bytes>],
    bindings: &'storage [ResolvedUpload],
    frame_index: ResolvedUpload,
    submission_sequence: ResolvedUpload,
}

impl ResolvedUploads<'_, '_, '_> {
    pub(super) fn stage(&self, frame: &mut [u8], index: i32, sequence: i64) {
        for (upload, binding) in self.uploads.iter().zip(self.bindings) {
            frame[binding.host_offset..binding.host_offset + binding.nbytes]
                .copy_from_slice(upload.bytes);
        }
        for (binding, bytes) in [
            (self.frame_index, index.to_le_bytes().as_slice()),
            (self.submission_sequence, sequence.to_le_bytes().as_slice()),
        ] {
            frame[binding.host_offset..binding.host_offset + binding.nbytes].copy_from_slice(bytes);
        }
    }

    pub(super) fn copies(&self) -> impl Iterator<Item = ResolvedUpload> + '_ {
        self.bindings
            .iter()
            .copied()
            .chain([self.frame_index, self.submission_sequence])
    }
}

/// Resolve one command against an already validated, immutable manifest.
/// Omitted inputs and disjoint partial uploads preserve resident device ranges.
/// Storage grows only to the fragment-count high-water; valid nonempty disjoint
/// fragments are bounded by input_frame_bytes. The executor reserves four uploads
/// per binding for physical request lanes before accepting any submissions.
pub(super) fn validate_uploads<'uploads, 'bytes, 'storage>(
    manifest: &ExecutionManifest,
    graph_index: usize,
    uploads: &'uploads [Upload<'bytes>],
    storage: &'storage mut Vec<ResolvedUpload>,
) -> Result<ResolvedUploads<'uploads, 'bytes, 'storage>> {
    storage.clear();
    let graph = manifest
        .graphs
        .get(graph_index)
        .ok_or_else(|| Error::execution(ExecutionError::Upload, "graph index out of range"))?;
    if uploads.len() > manifest.input_frame_bytes {
        return Err(Error::execution(
            ExecutionError::Upload,
            "upload count exceeds the nonempty input-fragment bound",
        ));
    }
    let reserved = |name: &str| -> Result<ResolvedUpload> {
        let binding = graph
            .input_bindings
            .iter()
            .find(|binding| binding.buffer_name == name)
            .ok_or_else(|| {
                Error::execution(
                    ExecutionError::Upload,
                    "graph lacks a reserved input binding",
                )
            })?;
        let buffer = manifest.buffer(name)?;
        Ok(ResolvedUpload {
            host_offset: binding.host_offset,
            device_address: buffer.address,
            nbytes: buffer.nbytes,
        })
    };
    let frame_index = reserved("frame_index")?;
    let submission_sequence = reserved("submission_sequence")?;
    for upload in uploads {
        if ["frame_index", "submission_sequence"].contains(&upload.buffer) {
            return Err(Error::execution(
                ExecutionError::Upload,
                "reserved executor inputs cannot be overridden",
            ));
        }
        let binding = graph
            .input_bindings
            .iter()
            .find(|binding| binding.buffer_name == upload.buffer)
            .ok_or_else(|| {
                Error::execution(ExecutionError::Upload, "upload is not bound by this graph")
            })?;
        let buffer = manifest.buffer(upload.buffer)?;
        if upload.bytes.is_empty()
            || upload.offset % buffer.dtype.size() != 0
            || upload.bytes.len() % buffer.dtype.size() != 0
            || upload
                .offset
                .checked_add(upload.bytes.len())
                .is_none_or(|end| end > binding.nbytes)
        {
            return Err(Error::execution(
                ExecutionError::Upload,
                format!("unaligned or out-of-range upload: {}", upload.buffer),
            ));
        }
        let resolved = ResolvedUpload {
            host_offset: binding.host_offset + upload.offset,
            device_address: buffer.address + upload.offset,
            nbytes: upload.bytes.len(),
        };
        // Manifest validation makes different buffers' host spans disjoint.
        if storage.iter().any(|prior| {
            prior.host_offset < resolved.host_offset + resolved.nbytes
                && resolved.host_offset < prior.host_offset + prior.nbytes
        }) {
            return Err(Error::execution(
                ExecutionError::Upload,
                "uploads overlap within one input buffer",
            ));
        }
        storage.push(resolved);
    }
    Ok(ResolvedUploads {
        uploads,
        bindings: storage,
        frame_index,
        submission_sequence,
    })
}

pub(super) fn decode_records(
    bytes: &[u8],
    fence: Fence,
) -> Result<[CompletionRecord; REQUEST_LANES]> {
    let vector = bytes.len() == REQUEST_LANES * VECTOR_RECORD_FIELDS * 8;
    if !vector && bytes.len() != COMPLETION_BYTES {
        return Err(Error::execution(
            ExecutionError::Completion,
            "completion extent mismatch",
        ));
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
                model_reserved: fields[12],
                draft_cache_consumed: fields[13],
                verification_width: fields[15],
                tokens: fields[16..24].try_into().unwrap(),
            }),
        };
        if vector && record.status != 0 && fields[14] != 0 {
            return Err(Error::execution(
                ExecutionError::Completion,
                "reserved vector completion field must be zero",
            ));
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
            return Err(Error::execution(
                ExecutionError::Completion,
                "invalid or stale device completion record",
            ));
        }
    }
    Ok(records)
}
