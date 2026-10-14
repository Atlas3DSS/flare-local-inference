//! CPython preparation hooks and native ownership at the package boundary.

pub use helios_core::*;

mod allocator;
mod embedding_tables;
mod lifetime;
mod processes;
mod python;
