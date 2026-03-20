//! Vera Core — parsing, indexing, retrieval, storage, and embedding engine.
//!
//! This crate provides the core functionality for Vera's code indexing and
//! retrieval pipeline. It is consumed by `vera-cli` (CLI interface) and
//! `vera-mcp` (MCP server).
//!
//! # Module overview
//!
//! - [`parsing`] — Tree-sitter based source code parsing and AST extraction.
//! - [`indexing`] — Building and maintaining the search index (BM25, vectors).
//! - [`retrieval`] — Hybrid search, fusion, and reranking pipeline.
//! - [`storage`] — Persistent storage backends (SQLite + sqlite-vec, Tantivy).
//! - [`embedding`] — Embedding generation via external API providers.

pub mod discovery;
pub mod embedding;
pub mod indexing;
pub mod parsing;
pub mod retrieval;
pub mod storage;

/// Shared types used across multiple modules.
pub mod types;

/// Configuration types and defaults.
pub mod config;

/// Index statistics collection.
pub mod stats;
