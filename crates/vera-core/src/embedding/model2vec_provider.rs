use std::path::PathBuf;
use std::sync::Arc;

use model2vec_rs::model::StaticModel;
use tokio::task;

use crate::embedding::provider::{EmbeddingError, EmbeddingProvider};
use crate::local_models::{POTION_CODE_DIM, POTION_CODE_MAX_LENGTH};

pub struct Model2VecProvider {
    model: Arc<StaticModel>,
    dim: usize,
    max_length: usize,
    batch_size: usize,
}

impl Model2VecProvider {
    pub async fn new_potion_code() -> anyhow::Result<Self> {
        let model_dir = crate::local_models::ensure_potion_code_assets().await?;
        Self::from_cached_potion_code_dir(model_dir).await
    }

    pub async fn from_cached_potion_code_dir(model_dir: PathBuf) -> anyhow::Result<Self> {
        let load_dir = model_dir.clone();
        let model = task::spawn_blocking(move || {
            StaticModel::from_pretrained(load_dir, None, Some(true), None)
        })
        .await
        .map_err(|err| anyhow::anyhow!("failed to join potion-code loader: {err}"))??;

        Ok(Self {
            model: Arc::new(model),
            dim: POTION_CODE_DIM,
            max_length: POTION_CODE_MAX_LENGTH,
            batch_size: 1024,
        })
    }

    pub fn probe_inference() -> anyhow::Result<()> {
        let model_dir = crate::local_models::potion_code_model_dir()?;
        for asset in crate::local_models::inspect_potion_code_model_files()? {
            if !asset.exists {
                anyhow::bail!("missing {} at {}", asset.name, asset.path.display());
            }
        }

        let model = StaticModel::from_pretrained(model_dir, None, Some(true), None)?;
        let embeddings = model.encode(&["vera doctor probe".to_string()]);
        let Some(vector) = embeddings.first() else {
            anyhow::bail!("potion-code returned no embeddings");
        };
        if vector.len() != POTION_CODE_DIM || !vector.iter().all(|value| value.is_finite()) {
            anyhow::bail!(
                "potion-code returned invalid embedding: dim={}, expected={}",
                vector.len(),
                POTION_CODE_DIM
            );
        }
        Ok(())
    }
}

impl EmbeddingProvider for Model2VecProvider {
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let model = Arc::clone(&self.model);
        let texts = texts.to_vec();
        let input_len = texts.len();
        let max_length = self.max_length;
        let batch_size = self.batch_size;
        let expected_dim = self.dim;

        let vectors = task::spawn_blocking(move || {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                model.encode_with_args(&texts, Some(max_length), batch_size)
            }))
            .map_err(|_| EmbeddingError::ResponseError {
                message: "potion-code tokenization failed".to_string(),
            })
        })
        .await
        .map_err(|err| EmbeddingError::ConnectionError {
            message: format!("potion-code worker failed: {err}"),
        })??;

        if vectors.len() != input_len {
            return Err(EmbeddingError::ResponseError {
                message: format!(
                    "potion-code returned {} vectors for {} inputs",
                    vectors.len(),
                    input_len
                ),
            });
        }

        if let Some(bad_dim) = vectors
            .iter()
            .map(Vec::len)
            .find(|&dim| dim != expected_dim)
        {
            return Err(EmbeddingError::ResponseError {
                message: format!(
                    "potion-code returned {bad_dim}-dim vectors, expected {expected_dim}"
                ),
            });
        }

        Ok(vectors)
    }

    fn expected_dim(&self) -> Option<usize> {
        Some(self.dim)
    }

    fn max_batch_size(&self) -> Option<usize> {
        Some(self.batch_size)
    }
}
