use crate::config::{InferenceBackend, VeraConfig};
use crate::retrieval::local_reranker::LocalReranker;
use crate::retrieval::reranker::{
    ApiReranker, RerankScore, Reranker, RerankerConfig, RerankerError,
};
use anyhow::Result;
use std::time::Duration;

pub enum DynamicReranker {
    Api(ApiReranker),
    Local(LocalReranker),
}

impl Reranker for DynamicReranker {
    async fn rerank(
        &self,
        query: &str,
        documents: &[String],
    ) -> Result<Vec<RerankScore>, RerankerError> {
        match self {
            Self::Api(p) => p.rerank(query, documents).await,
            Self::Local(p) => p.rerank(query, documents).await,
        }
    }
}

pub async fn create_dynamic_reranker(
    config: &VeraConfig,
    backend: InferenceBackend,
) -> anyhow::Result<Option<DynamicReranker>> {
    if !config.retrieval.reranking_enabled {
        return Ok(None);
    }

    match backend {
        InferenceBackend::OnnxJina(ep) => {
            let p = LocalReranker::new_with_ep(ep)
                .await
                .map_err(|e| anyhow::anyhow!("Failed to initialize local reranker: {e}\nHint: check network connection or manually place model at ~/.vera/models/"))?;
            Ok(Some(DynamicReranker::Local(p)))
        }
        InferenceBackend::PotionCode => Ok(None),
        InferenceBackend::Api => match RerankerConfig::from_env() {
            Ok(cfg) => {
                let cfg = cfg
                    .with_timeout(Duration::from_secs(30))
                    .with_max_retries(2);
                let p = ApiReranker::new(cfg)
                    .map_err(|err| anyhow::anyhow!("failed to init reranker: {err}"))?;
                Ok(Some(DynamicReranker::Api(p)))
            }
            Err(_) => Ok(None),
        },
    }
}
