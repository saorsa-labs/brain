//! Shared library code for the PTG command-line binaries (`ptg`, `ptg-bench`,
//! `ptg-judge`). Hosts the column-pack parser and the topology-resolution
//! logic so all three binaries share a single source of truth.

pub mod column_pack;
pub mod setup;
pub mod topology_cli;
