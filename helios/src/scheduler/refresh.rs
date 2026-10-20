use super::{Fence, RequestHandle, ScheduledStep, cost};
use crate::{Result, execution::{ExecutionTiming, GraphKind, REQUEST_LANES}};

pub(super) const CREDIT_CAP_NS: f64 = 5_000_000.0;
const INCOME_RATE: f64 = 0.002 / 0.998;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum PlanOrigin {
    #[default]
    Ordinary,
    Discovery,
    RefreshStart(cost::Key),
    RefreshContinuation(u64),
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Owner {
    pub handle: RequestHandle,
    pub consumed: u64,
    pub fence: Fence,
}

pub(super) struct Episode {
    pub id: u64,
    pub target: cost::Key,
    pub owners: [Option<Owner>; REQUEST_LANES],
    inflight: usize,
}

impl Episode {
    pub fn owns(&self, handle: RequestHandle, consumed: u64, fence: Fence) -> bool {
        self.owners.get(handle.slot as usize).and_then(|owner| *owner)
            .is_some_and(|owner| owner.handle == handle && owner.consumed == consumed && owner.fence == fence)
    }

    pub fn owns_handle(&self, handle: RequestHandle) -> bool {
        self.owners.get(handle.slot as usize).and_then(|owner| *owner)
            .is_some_and(|owner| owner.handle == handle)
    }

    pub fn owned_mask(&self) -> u64 {
        self.owners.iter().enumerate().filter(|(_, owner)| owner.is_some())
            .fold(0, |mask, (slot, _)| mask | (1 << slot))
    }
}

#[derive(Default)]
pub(super) struct RefreshState {
    pub credit_ns: f64,
    active: Option<Episode>,
}

impl RefreshState {
    pub fn ready(&self) -> bool {
        self.active.is_none() && self.credit_ns >= CREDIT_CAP_NS
    }

    pub fn episode(&self) -> Option<&Episode> {
        self.active.as_ref()
    }

    pub fn validate(&self, origin: PlanOrigin) -> Result<()> {
        match origin {
            PlanOrigin::RefreshStart(_) if !self.ready() => Err("refresh requires full credit and no active episode".into()),
            PlanOrigin::RefreshContinuation(id) if self.active.as_ref().is_none_or(|episode| episode.id != id) =>
                Err("refresh continuation does not own the active episode".into()),
            _ => Ok(()),
        }
    }

    pub fn commit(&mut self, origin: PlanOrigin, fence: Fence, step: &ScheduledStep) {
        match origin {
            PlanOrigin::RefreshStart(target) => {
                let mut owners = [None; REQUEST_LANES];
                if step.kind == GraphKind::Proposal {
                    for lane in &step.lanes {
                        owners[lane.request.slot as usize] = Some(Owner {
                            handle: lane.request, consumed: lane.start_position, fence,
                        });
                    }
                }
                self.active = Some(Episode { id: fence.0, target, owners, inflight: 1 });
            }
            PlanOrigin::RefreshContinuation(_) => {
                let episode = self.active.as_mut().unwrap();
                episode.inflight += 1;
                for lane in &step.lanes {
                    if episode.owns_handle(lane.request) {
                        episode.owners[lane.request.slot as usize] = None;
                    }
                }
            }
            _ => {}
        }
    }

    pub fn validate_completion(&self, origin: PlanOrigin, timing: ExecutionTiming) -> Result<()> {
        if matches!(origin, PlanOrigin::RefreshStart(_) | PlanOrigin::RefreshContinuation(_))
            && (timing.queued || (cost::observed_duration_ns(timing).is_none() && timing.wall_ns == 0)) {
            return Err("refresh completion requires an isolated measured interval".into());
        }
        Ok(())
    }

    pub fn complete(&mut self, origin: PlanOrigin, timing: ExecutionTiming) {
        match origin {
            PlanOrigin::Ordinary => {
                if let Some(duration) = cost::observed_duration_ns(timing) {
                    self.credit_ns = CREDIT_CAP_NS.min(self.credit_ns + INCOME_RATE * duration as f64);
                }
            }
            PlanOrigin::Discovery => {}
            PlanOrigin::RefreshStart(_) | PlanOrigin::RefreshContinuation(_) => {
                let duration = cost::observed_duration_ns(timing).unwrap_or(timing.wall_ns);
                self.credit_ns -= duration as f64;
                self.active.as_mut().unwrap().inflight -= 1;
                self.close_resolved();
            }
        }
    }

    pub fn retain_owners(&mut self, keep: &[bool; REQUEST_LANES]) {
        if let Some(episode) = &mut self.active {
            for (owner, &keep) in episode.owners.iter_mut().zip(keep) {
                if !keep { *owner = None; }
            }
        }
        self.close_resolved();
    }

    fn close_resolved(&mut self) {
        if self.active.as_ref().is_some_and(|episode| episode.inflight == 0 && episode.owned_mask() == 0) {
            self.active = None;
        }
    }

    pub fn idle(&mut self) {
        self.credit_ns = self.credit_ns.min(0.0);
    }

    pub fn abort(&mut self) {
        self.active = None;
        self.idle();
    }
}
