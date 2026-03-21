use crate::local_models::ensure_model_file;
use crate::retrieval::reranker::{RerankScore, Reranker, RerankerError};
use anyhow::Result;
use ort::session::{Session, builder::GraphOptimizationLevel};
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;
use tokio::task;

const RERANKER_REPO: &str = "jinaai/jina-reranker-v2-base-multilingual";
const ONNX_FILE: &str = "onnx/model_quantized.onnx";
const TOKENIZER_FILE: &str = "tokenizer.json";

#[derive(Clone)]
pub struct LocalReranker {
    session: Arc<Mutex<Session>>,
    tokenizer: Arc<Tokenizer>,
}

impl LocalReranker {
    pub async fn new() -> Result<Self, RerankerError> {
        let onnx_path = ensure_model_file(RERANKER_REPO, ONNX_FILE)
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

        let tokenizer = task::spawn_blocking(move || {
            Tokenizer::from_file(tokenizer_path)
                .map_err(|e| anyhow::anyhow!("Tokenizer init failed: {}", e))
        })
        .await
        .map_err(|e| RerankerError::ApiError {
            status: 500,
            message: e.to_string(),
        })?
        .map_err(|e| RerankerError::ApiError {
            status: 500,
            message: e.to_string(),
        })?;

        let session = task::spawn_blocking(move || {
            let threads = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1);
            ort::session::builder::SessionBuilder::new()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_intra_threads(threads)?
                .commit_from_file(onnx_path)
        })
        .await
        .map_err(|e| RerankerError::ApiError {
            status: 500,
            message: e.to_string(),
        })?
        .map_err(|e| RerankerError::ApiError {
            status: 500,
            message: e.to_string(),
        })?;

        Ok(Self {
            session: Arc::new(Mutex::new(session)),
            tokenizer: Arc::new(tokenizer),
        })
    }

    #[allow(clippy::needless_range_loop)]
    fn do_rerank(&self, query: &str, documents: &[String]) -> Result<Vec<RerankScore>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }

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
                    index: i,
                    relevance_score: score as f64,
                });
            }
        } else if ndim == 1 {
            for i in 0..batch_size {
                let score = data[i];
                results.push(RerankScore {
                    index: i,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_local_reranker() {
        let reranker = LocalReranker::new().await.unwrap();
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
