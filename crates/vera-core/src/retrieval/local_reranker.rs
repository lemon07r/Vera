use crate::config::OnnxExecutionProvider;
use crate::local_models::ensure_model_file;
use crate::retrieval::reranker::{RerankScore, Reranker, RerankerError};
use anyhow::{Context, Result};
use ort::session::{Session, builder::GraphOptimizationLevel};
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;
use tokio::task;

const RERANKER_REPO: &str = "jinaai/jina-reranker-v2-base-multilingual";
const TOKENIZER_FILE: &str = "tokenizer.json";
const MAX_RERANK_BATCH_SIZE: usize = 8;

#[derive(Clone)]
pub struct LocalReranker {
    session: Arc<Mutex<Session>>,
    tokenizer: Arc<Tokenizer>,
}

impl LocalReranker {
    pub async fn new_with_ep(ep: OnnxExecutionProvider) -> Result<Self, RerankerError> {
        let ort_path = crate::local_models::ensure_ort_library_for_ep(ep)
            .await
            .map_err(|e| RerankerError::ApiError {
                status: 500,
                message: e.to_string(),
            })?;
        crate::local_models::ensure_ort_runtime(Some(&ort_path)).map_err(|e| {
            RerankerError::ApiError {
                status: 500,
                message: e.to_string(),
            }
        })?;
        crate::local_models::ensure_provider_dependencies(ep, &ort_path).map_err(|e| {
            RerankerError::ApiError {
                status: 500,
                message: e.to_string(),
            }
        })?;

        let onnx_path = ensure_model_file(
            RERANKER_REPO,
            crate::local_models::reranker_onnx_file_for_ep(ep),
        )
        .await
        .map_err(|e| RerankerError::ApiError {
            status: 500,
            message: format!("Failed to download ONNX model: {}", e),
        })?;

        let tokenizer_path = ensure_model_file(RERANKER_REPO, TOKENIZER_FILE)
            .await
            .map_err(|e| RerankerError::ApiError {
                status: 500,
                message: format!("Failed to download tokenizer: {}", e),
            })?;

        let tokenizer = task::spawn_blocking(move || load_tokenizer(tokenizer_path))
            .await
            .map_err(|e| RerankerError::ApiError {
                status: 500,
                message: e.to_string(),
            })?
            .map_err(|e| RerankerError::ApiError {
                status: 500,
                message: e.to_string(),
            })?;

        let session = task::spawn_blocking(move || build_session(ep, onnx_path))
            .await
            .map_err(|e| RerankerError::ApiError {
                status: 500,
                message: e.to_string(),
            })?
            .map_err(|e| RerankerError::ApiError {
                status: 500,
                message: crate::local_models::wrap_ort_error(e),
            })?;

        Ok(Self {
            session: Arc::new(Mutex::new(session)),
            tokenizer: Arc::new(tokenizer),
        })
    }

    pub fn probe_session(ep: OnnxExecutionProvider) -> Result<()> {
        let ort_path = crate::local_models::ort_library_path_for_ep(ep)?;
        crate::local_models::ensure_ort_runtime(Some(&ort_path))?;
        let (onnx_path, _) = default_asset_paths(ep)?;
        let _ = build_session(ep, onnx_path)?;
        Ok(())
    }

    pub fn probe_inference(ep: OnnxExecutionProvider) -> Result<()> {
        let ort_path = crate::local_models::ort_library_path_for_ep(ep)?;
        crate::local_models::ensure_ort_runtime(Some(&ort_path))?;
        let (onnx_path, tokenizer_path) = default_asset_paths(ep)?;
        let mut session = build_session(ep, onnx_path)?;
        let tokenizer = load_tokenizer(tokenizer_path)?;
        run_probe_inference(&mut session, &tokenizer)
    }

    #[allow(clippy::needless_range_loop)]
    fn do_rerank_batch(
        &self,
        query: &str,
        documents: &[String],
        index_offset: usize,
    ) -> Result<Vec<RerankScore>> {
        let mut encodings = Vec::with_capacity(documents.len());
        for doc in documents {
            let encoding = self
                .tokenizer
                .encode((query.to_string(), doc.clone()), true)
                .map_err(|e| anyhow::anyhow!("Tokenizer error: {}", e))?;
            encodings.push(encoding);
        }

        let batch_size = documents.len();
        let mut max_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0);
        if max_len == 0 {
            max_len = 1;
        }

        let truncate_len = 512;
        if max_len > truncate_len {
            max_len = truncate_len;
        }

        let mut input_ids = ndarray::Array2::<i64>::zeros((batch_size, max_len));
        let mut attention_mask = ndarray::Array2::<i64>::zeros((batch_size, max_len));

        for (i, encoding) in encodings.iter().enumerate() {
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();
            let len = std::cmp::min(ids.len(), max_len);
            for j in 0..len {
                input_ids[[i, j]] = ids[j] as i64;
                attention_mask[[i, j]] = mask[j] as i64;
            }
        }

        let input_ids_tensor = ort::value::Tensor::from_array(input_ids)
            .map_err(|e| anyhow::anyhow!("Tensor error: {}", e))?;
        let attention_mask_tensor = ort::value::Tensor::from_array(attention_mask)
            .map_err(|e| anyhow::anyhow!("Tensor error: {}", e))?;

        let inputs = ort::inputs![
            "input_ids" => input_ids_tensor,
            "attention_mask" => attention_mask_tensor,
        ];

        let mut session = self.session.lock().unwrap();
        let outputs = session.run(inputs)?;

        let output_value = outputs.values().next().unwrap();
        let (shape, data) = output_value.try_extract_tensor::<f32>()?;
        let ndim = shape.len();

        let mut results = Vec::with_capacity(batch_size);
        if ndim == 2 {
            let dim = shape[1] as usize;
            for i in 0..batch_size {
                let score = data[i * dim];
                results.push(RerankScore {
                    index: index_offset + i,
                    relevance_score: score as f64,
                });
            }
        } else if ndim == 1 {
            for i in 0..batch_size {
                let score = data[i];
                results.push(RerankScore {
                    index: index_offset + i,
                    relevance_score: score as f64,
                });
            }
        } else {
            anyhow::bail!("Unexpected tensor shape for reranker: {:?}", shape);
        }

        results.sort_by(|a, b| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(results)
    }

    fn do_rerank(&self, query: &str, documents: &[String]) -> Result<Vec<RerankScore>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        let mut combined = Vec::with_capacity(documents.len());
        for (batch_index, batch) in documents.chunks(MAX_RERANK_BATCH_SIZE).enumerate() {
            let mut scores =
                self.do_rerank_batch(query, batch, batch_index * MAX_RERANK_BATCH_SIZE)?;
            combined.append(&mut scores);
        }

        combined.sort_by(|a, b| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(combined)
    }
}

fn default_asset_paths(
    ep: OnnxExecutionProvider,
) -> Result<(std::path::PathBuf, std::path::PathBuf)> {
    let model_dir = crate::local_models::vera_home_dir()?
        .join("models")
        .join(RERANKER_REPO);
    Ok((
        model_dir.join(crate::local_models::reranker_onnx_file_for_ep(ep)),
        model_dir.join(TOKENIZER_FILE),
    ))
}

fn load_tokenizer(tokenizer_path: std::path::PathBuf) -> Result<Tokenizer> {
    Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Tokenizer init failed: {}", e))
}

fn build_session(ep: OnnxExecutionProvider, onnx_path: std::path::PathBuf) -> Result<Session> {
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let builder = ort::session::builder::SessionBuilder::new()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(threads)?;
    let builder = register_execution_provider(builder, ep)?;
    builder
        .commit_from_file(&onnx_path)
        .with_context(|| format!("failed to load reranker model {}", onnx_path.display()))
}

fn run_probe_inference(session: &mut Session, tokenizer: &Tokenizer) -> Result<()> {
    let encoding = tokenizer
        .encode(
            (
                "vera doctor probe".to_string(),
                "diagnose the selected execution provider".to_string(),
            ),
            true,
        )
        .map_err(|e| anyhow::anyhow!("Tokenizer error: {}", e))?;

    let ids = encoding.get_ids();
    let mask = encoding.get_attention_mask();
    let max_len = ids.len().max(1);
    let mut input_ids = ndarray::Array2::<i64>::zeros((1, max_len));
    let mut attention_mask = ndarray::Array2::<i64>::zeros((1, max_len));

    for (index, token_id) in ids.iter().enumerate() {
        input_ids[[0, index]] = *token_id as i64;
    }
    for (index, mask_value) in mask.iter().enumerate() {
        attention_mask[[0, index]] = *mask_value as i64;
    }

    let inputs = ort::inputs![
        "input_ids" => ort::value::Tensor::from_array(input_ids)?,
        "attention_mask" => ort::value::Tensor::from_array(attention_mask)?,
    ];

    let outputs = session.run(inputs)?;
    let output = outputs
        .values()
        .next()
        .context("reranker model produced no outputs")?;
    let (_, data) = output.try_extract_tensor::<f32>()?;
    if data.is_empty() {
        anyhow::bail!("reranker output tensor was empty");
    }
    if !data.iter().all(|value| value.is_finite()) {
        anyhow::bail!("reranker output contained non-finite values");
    }
    Ok(())
}

impl Reranker for LocalReranker {
    async fn rerank(
        &self,
        query: &str,
        documents: &[String],
    ) -> Result<Vec<RerankScore>, RerankerError> {
        let provider = self.clone();
        let query = query.to_string();
        let documents = documents.to_vec();

        task::spawn_blocking(move || {
            provider
                .do_rerank(&query, &documents)
                .map_err(|e| RerankerError::ApiError {
                    status: 500,
                    message: e.to_string(),
                })
        })
        .await
        .map_err(|e| RerankerError::ApiError {
            status: 500,
            message: e.to_string(),
        })?
    }
}

/// Register the appropriate ONNX execution provider on a session builder.
fn register_execution_provider(
    builder: ort::session::builder::SessionBuilder,
    ep: OnnxExecutionProvider,
) -> ort::Result<ort::session::builder::SessionBuilder> {
    match ep {
        OnnxExecutionProvider::Cpu => Ok(builder),
        OnnxExecutionProvider::Cuda => builder.with_execution_providers([
            ort::execution_providers::CUDAExecutionProvider::default().build(),
        ]),
        OnnxExecutionProvider::Rocm => builder.with_execution_providers([
            ort::execution_providers::ROCmExecutionProvider::default().build(),
        ]),
        OnnxExecutionProvider::DirectMl => builder.with_execution_providers([
            ort::execution_providers::DirectMLExecutionProvider::default().build(),
        ]),
        OnnxExecutionProvider::CoreMl => builder.with_execution_providers([
            ort::execution_providers::CoreMLExecutionProvider::default().build(),
        ]),
        OnnxExecutionProvider::OpenVino => builder.with_execution_providers([
            ort::execution_providers::OpenVINOExecutionProvider::default().build(),
        ]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_local_reranker() {
        // Skip if ONNX Runtime is not installed (requires libonnxruntime.so)
        if crate::local_models::ensure_ort_runtime(None).is_err() {
            eprintln!("Skipping: ONNX Runtime not available");
            return;
        }
        let reranker = LocalReranker::new_with_ep(OnnxExecutionProvider::Cpu)
            .await
            .unwrap();
        let query = "How to parse JSON".to_string();
        let docs = vec![
            "This is a random document about cars.".to_string(),
            "JSON parsing in Rust can be done using serde_json::from_str.".to_string(),
        ];
        let scores = reranker.rerank(&query, &docs).await.unwrap();
        assert_eq!(scores.len(), 2);

        let relevant_score = scores
            .iter()
            .find(|s| s.index == 1)
            .unwrap()
            .relevance_score;
        let irrelevant_score = scores
            .iter()
            .find(|s| s.index == 0)
            .unwrap()
            .relevance_score;
        assert!(relevant_score > irrelevant_score);

        // Assert sorting is descending
        assert_eq!(scores[0].index, 1);
        assert_eq!(scores[1].index, 0);
        assert!(scores[0].relevance_score > scores[1].relevance_score);
    }
}
