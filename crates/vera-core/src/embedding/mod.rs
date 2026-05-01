//! Embedding generation via external API providers.
//!
//! This module provides:
//! - [`EmbeddingProvider`] trait for abstracting embedding API calls
//! - [`OpenAiProvider`] for OpenAI-compatible embedding endpoints
//! - Batched embedding generation with configurable batch size
//! - Credential management (read from environment, never log)
//! - Error handling (auth failures, connection errors, rate limits)

mod provider;

pub use provider::{
    CachedEmbeddingProvider, EmbeddingError, EmbeddingProvider, EmbeddingProviderConfig,
    OpenAiProvider, embed_chunks, embed_chunks_concurrent, embed_chunks_concurrent_with_progress,
};

pub mod dynamic;
pub use dynamic::{DynamicProvider, create_dynamic_provider};

pub mod local_provider;

pub use local_provider::LocalEmbeddingProvider;

pub mod model2vec_provider;
pub use model2vec_provider::Model2VecProvider;

/// Test helpers for creating mock embedding providers.
#[cfg(test)]
pub(crate) mod test_helpers {
    pub use super::provider::test_helpers::MockProvider;
}

#[cfg(test)]
mod tests;
