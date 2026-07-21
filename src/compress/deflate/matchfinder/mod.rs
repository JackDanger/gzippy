//! Lempel-Ziv matchfinding.
//!
//! Increment 1 lands only the shared primitives in [`common`]; the hash-chain /
//! binary-tree / hash-table finders come in later increments.

pub mod bt;
pub mod common;
pub mod hc;
/// ECT-class BT4 full-Pareto matchfinder — the crown engine's (`parse::ultra`)
/// seed matchfinder for the multi-seed iterated squeeze. Not used by any
/// other level tier.
pub mod lzfind;
