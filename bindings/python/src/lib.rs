#![allow(non_local_definitions)]

use std::collections::{BTreeMap, HashMap, HashSet};
use std::env;
use std::fs;
use std::io::{ErrorKind, Read};
use std::num::NonZeroU64;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::process;
use std::sync::{Mutex, OnceLock};
use std::thread;
use std::time::Duration;

use chrono::Utc;
#[cfg(unix)]
use libc;
use memvid_ask_model::{
    run_model_inference, ModelContextFragment, ModelContextFragmentKind, ModelRunError,
};
#[cfg(feature = "encryption")]
use memvid_core::encryption::EncryptionError as CapsuleError;
use memvid_core::enrich::EnrichmentContext;
#[cfg(feature = "replay")]
use memvid_core::replay::{ReplayEngine, ReplayExecutionConfig};
use memvid_core::table::{
    export_to_csv, export_to_json, extract_tables, get_table, list_tables,
    store_table_with_embedder, TableExtractionOptions,
};
use memvid_core::types::adaptive::{AdaptiveConfig, CutoffStrategy};
use memvid_core::types::ask::{AskContextFragment, AskContextFragmentKind, AskMode, AskRequest};
use memvid_core::types::audit::{AuditOptions, AuditReport};
use memvid_core::types::embedding_identity::{EmbeddingIdentity, EmbeddingIdentitySummary};
use memvid_core::types::metadata::DocMetadata;
use memvid_core::types::search::{SearchEngineKind, SearchParams, SearchRequest, SearchResponse};
use memvid_core::types::{
    AclContext, AclEnforcementMode, DoctorOptions, Frame, MediaManifest, MemoryBinding,
    PutManyOpts, PutRequest, SearchHit, SignedTicket, Ticket, TimelineQuery, VerificationStatus,
};
use memvid_core::{
    AskResponse as CoreAskResponse, DocumentFormat, EnrichmentEngine, Memvid as MemvidCore,
    MemvidError as MemvidCoreError, ReaderHint, ReaderRegistry, Result as MemvidResult,
    RulesEngine, Stats as CoreStats, TimelineQueryBuilder, VecEmbedder,
};
#[cfg(feature = "parallel_segments")]
use memvid_core::{BuildOpts, ParallelInput, ParallelPayload};
use memvid_core::{
    MEMVID_EMBEDDING_DIMENSION_KEY, MEMVID_EMBEDDING_MODEL_KEY, MEMVID_EMBEDDING_NORMALIZED_KEY,
    MEMVID_EMBEDDING_PROVIDER_KEY,
};
use uuid::Uuid;

mod lock;
use crate::lock::{current_owner as lock_current_owner, PyLockOwner};
use pyo3::exceptions::{PyException, PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple, PyType};
use serde_json::{json, Number, Value};

const DEFAULT_FIND_K: usize = 10;
const DEFAULT_FIND_SNIPPET: usize = 800;
const DEFAULT_ASK_K: usize = 6;
const DEFAULT_ASK_SNIPPET: usize = 800;
const DEFAULT_TIMELINE_LIMIT: u64 = 50;
const MAX_HIT_SNIPPET_CHARS: usize = 50000;
const MAX_CONTEXT_SNIPPET_CHARS: usize = 50000;
const DEFAULT_LOCK_TIMEOUT_MS: u64 = 250;
const DEFAULT_API_URL: &str = "https://memvid.com";
/// Free tier file size limit: 9999 GB
const FREE_TIER_LIMIT_BYTES: u64 = 9999 * 1024 * 1024 * 1024; // 9999 GB
const OPENAI_EMBEDDINGS_PATH: &str = "/v1/embeddings";

/// How often to refresh ticket for write operations (5 minutes)
const WRITE_OP_REFRESH_SECS: i64 = 300;

/// Cached subscription ticket for SDK
#[derive(Debug, Clone)]
struct CachedSubscription {
    status: String,
    capacity_bytes: u64,
    plan_end_date: Option<String>,
    cached_at: i64,
}

impl CachedSubscription {
    /// Check if subscription is in grace period (canceled but end date is in future)
    fn is_in_grace_period(&self) -> bool {
        if self.status != "canceled" {
            return false;
        }
        if let Some(ref end_date) = self.plan_end_date {
            if let Ok(end) = chrono::DateTime::parse_from_rfc3339(end_date) {
                return end > Utc::now();
            }
        }
        false
    }

    /// Check if the ticket is too old for write operations
    fn is_stale_for_writes(&self) -> bool {
        let now = Utc::now().timestamp();
        now - self.cached_at > WRITE_OP_REFRESH_SECS
    }

    /// Check if subscription allows write operations
    fn allows_writes(&self) -> bool {
        match self.status.as_str() {
            "active" | "trialing" | "past_due" => true,
            "canceled" => self.is_in_grace_period(),
            _ => false, // inactive or other
        }
    }
}

/// Global subscription cache (per API key)
static SUBSCRIPTION_CACHE: OnceLock<Mutex<HashMap<String, CachedSubscription>>> = OnceLock::new();

fn get_subscription_cache() -> &'static Mutex<HashMap<String, CachedSubscription>> {
    SUBSCRIPTION_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Fetch subscription ticket from dashboard API
fn fetch_subscription_ticket(api_key: &str) -> PyResult<CachedSubscription> {
    let api_url = env::var("MEMVID_DASHBOARD_URL")
        .or_else(|_| env::var("MEMVID_API_URL"))
        .unwrap_or_else(|_| DEFAULT_API_URL.to_string());
    // Handle both http://localhost:3001 and http://localhost:3001/api
    let base = api_url.trim_end_matches('/').trim_end_matches("/api");
    let url = format!("{}/api/ticket", base);

    let response = ureq::get(&url)
        .set("x-api-key", api_key)
        .call()
        .map_err(|e| {
            if let ureq::Error::Status(401, _) = e {
                PyRuntimeError::new_err(
                    "Invalid API key. Get a valid key at https://memvid.com/dashboard/api-keys",
                )
            } else {
                PyRuntimeError::new_err(format!("Failed to fetch subscription: {}", e))
            }
        })?;

    let body: serde_json::Value = response
        .into_json()
        .map_err(|_| PyRuntimeError::new_err("Failed to parse ticket response"))?;

    let data = body
        .get("data")
        .ok_or_else(|| PyRuntimeError::new_err("Invalid ticket response: missing data"))?;

    let ticket = data
        .get("ticket")
        .ok_or_else(|| PyRuntimeError::new_err("Invalid ticket response: missing ticket"))?;

    let subscription = data
        .get("subscription")
        .ok_or_else(|| PyRuntimeError::new_err("Invalid ticket response: missing subscription"))?;

    let status = subscription
        .get("status")
        .and_then(|v| v.as_str())
        .unwrap_or("inactive")
        .to_string();

    let capacity_bytes = ticket
        .get("capacity_bytes")
        .and_then(|v| v.as_u64())
        .unwrap_or(FREE_TIER_LIMIT_BYTES);

    let plan_end_date = subscription
        .get("planEndDate")
        .or_else(|| subscription.get("ends_at"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    Ok(CachedSubscription {
        status,
        capacity_bytes,
        plan_end_date,
        cached_at: Utc::now().timestamp(),
    })
}

/// Get fresh subscription for write operations (smart refresh)
fn get_fresh_subscription_for_writes(api_key: &str) -> PyResult<CachedSubscription> {
    let cache = get_subscription_cache();
    let mut cache_guard = cache
        .lock()
        .map_err(|_| PyRuntimeError::new_err("Failed to acquire subscription cache lock"))?;

    // Check if we have a cached subscription
    let needs_refresh = match cache_guard.get(api_key) {
        Some(cached) => {
            // Smart refresh: active users get 5-min cache, others always refresh
            if cached.status == "active" || cached.status == "trialing" {
                cached.is_stale_for_writes()
            } else {
                // Canceled/inactive: always refresh to catch reactivation instantly
                true
            }
        }
        None => true,
    };

    if needs_refresh {
        let fresh = fetch_subscription_ticket(api_key)?;
        cache_guard.insert(api_key.to_string(), fresh.clone());
        Ok(fresh)
    } else {
        Ok(cache_guard
            .get(api_key)
            .ok_or_else(|| {
                PyRuntimeError::new_err("subscription cache entry disappeared unexpectedly")
            })?
            .clone())
    }
}

/// Check if subscription allows write operations
fn require_active_subscription(api_key: &Option<String>) -> PyResult<Option<CachedSubscription>> {
    let api_key = match api_key {
        Some(key) => key,
        None => return Ok(None), // No API key = free tier, allow
    };

    let subscription = get_fresh_subscription_for_writes(api_key)?;

    if !subscription.allows_writes() {
        return Err(PyRuntimeError::new_err(format!(
            "Your subscription has expired.\n\n\
             This operation requires an active subscription.\n\
             Your plan ended on: {}\n\n\
             You can still read your data with stats() and timeline().\n\n\
             To restore full access, reactivate your subscription:\n\
             https://memvid.com/dashboard/plan",
            subscription.plan_end_date.as_deref().unwrap_or("unknown")
        )));
    }

    Ok(Some(subscription))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum EmbeddingModelChoice {
    BgeSmall,
    BgeBase,
    Nomic,
    GteLarge,
    OpenAILarge,
    OpenAISmall,
    OpenAIAda,
}

impl EmbeddingModelChoice {
    fn is_openai(&self) -> bool {
        matches!(
            self,
            EmbeddingModelChoice::OpenAILarge
                | EmbeddingModelChoice::OpenAISmall
                | EmbeddingModelChoice::OpenAIAda
        )
    }

    fn name(&self) -> &'static str {
        match self {
            EmbeddingModelChoice::BgeSmall => "bge-small",
            EmbeddingModelChoice::BgeBase => "bge-base",
            EmbeddingModelChoice::Nomic => "nomic",
            EmbeddingModelChoice::GteLarge => "gte-large",
            EmbeddingModelChoice::OpenAILarge => "openai-large",
            EmbeddingModelChoice::OpenAISmall => "openai-small",
            EmbeddingModelChoice::OpenAIAda => "openai-ada",
        }
    }

    fn canonical_model_id(&self) -> &'static str {
        match self {
            EmbeddingModelChoice::BgeSmall => "BAAI/bge-small-en-v1.5",
            EmbeddingModelChoice::BgeBase => "BAAI/bge-base-en-v1.5",
            EmbeddingModelChoice::Nomic => "nomic-embed-text-v1.5",
            EmbeddingModelChoice::GteLarge => "thenlper/gte-large",
            EmbeddingModelChoice::OpenAILarge => "text-embedding-3-large",
            EmbeddingModelChoice::OpenAISmall => "text-embedding-3-small",
            EmbeddingModelChoice::OpenAIAda => "text-embedding-ada-002",
        }
    }

    fn dimensions(&self) -> usize {
        match self {
            EmbeddingModelChoice::BgeSmall => 384,
            EmbeddingModelChoice::BgeBase => 768,
            EmbeddingModelChoice::Nomic => 768,
            EmbeddingModelChoice::GteLarge => 1024,
            EmbeddingModelChoice::OpenAILarge => 3072,
            EmbeddingModelChoice::OpenAISmall => 1536,
            EmbeddingModelChoice::OpenAIAda => 1536,
        }
    }

    #[cfg(feature = "fastembed")]
    fn to_fastembed_model(&self) -> MemvidResult<fastembed::EmbeddingModel> {
        match self {
            EmbeddingModelChoice::BgeSmall => Ok(fastembed::EmbeddingModel::BGESmallENV15),
            EmbeddingModelChoice::BgeBase => Ok(fastembed::EmbeddingModel::BGEBaseENV15),
            EmbeddingModelChoice::Nomic => Ok(fastembed::EmbeddingModel::NomicEmbedTextV15),
            EmbeddingModelChoice::GteLarge => Ok(fastembed::EmbeddingModel::GTELargeENV15),
            EmbeddingModelChoice::OpenAILarge
            | EmbeddingModelChoice::OpenAISmall
            | EmbeddingModelChoice::OpenAIAda => Err(MemvidCoreError::EmbeddingFailed {
                reason: "OpenAI models don't use fastembed. Check is_openai() first."
                    .to_string()
                    .into_boxed_str(),
            }),
        }
    }

    fn from_dimension(dim: u32) -> Option<Self> {
        match dim {
            384 => Some(Self::BgeSmall),
            768 => Some(Self::BgeBase), // could also be Nomic
            1024 => Some(Self::GteLarge),
            1536 => Some(Self::OpenAISmall), // could also be Ada
            3072 => Some(Self::OpenAILarge),
            _ => None,
        }
    }

    fn parse(value: &str) -> MemvidResult<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "bge-small" | "bge_small" | "bgesmall" | "small" | "baai/bge-small-en-v1.5" => {
                Ok(Self::BgeSmall)
            }
            "bge-base" | "bge_base" | "bgebase" | "base" | "baai/bge-base-en-v1.5" => {
                Ok(Self::BgeBase)
            }
            "nomic" | "nomic-embed" | "nomic_embed" | "nomic-embed-text-v1.5" => Ok(Self::Nomic),
            "gte-large" | "gte_large" | "gtelarge" | "gte" | "thenlper/gte-large" => {
                Ok(Self::GteLarge)
            }
            "openai" | "openai-large" | "openai_large" | "text-embedding-3-large" => {
                Ok(Self::OpenAILarge)
            }
            "openai-small" | "openai_small" | "text-embedding-3-small" => Ok(Self::OpenAISmall),
            "openai-ada" | "openai_ada" | "text-embedding-ada-002" => Ok(Self::OpenAIAda),
            other => Err(MemvidCoreError::EmbeddingFailed {
                reason: format!(
                    "unsupported embedding model '{other}'. Expected one of: bge-small, bge-base, nomic, gte-large, openai-large, openai-small, openai-ada"
                )
                .into_boxed_str(),
            }),
        }
    }
}

#[derive(Clone)]
struct OpenAIEmbeddingProvider {
    api_key: String,
    model: EmbeddingModelChoice,
    base_url: String,
}

impl OpenAIEmbeddingProvider {
    fn from_env(model: EmbeddingModelChoice) -> MemvidResult<Self> {
        let api_key = env::var("OPENAI_API_KEY")
            .map_err(|_| MemvidCoreError::EmbeddingFailed {
                reason: "OPENAI_API_KEY environment variable is required for OpenAI embeddings"
                    .to_string()
                    .into_boxed_str(),
            })?
            .trim()
            .to_string();
        if api_key.is_empty() {
            return Err(MemvidCoreError::EmbeddingFailed {
                reason: "OPENAI_API_KEY cannot be empty"
                    .to_string()
                    .into_boxed_str(),
            });
        }

        if is_offline() {
            return Err(MemvidCoreError::EmbeddingFailed {
                reason: "OpenAI embeddings unavailable while MEMVID_OFFLINE=1"
                    .to_string()
                    .into_boxed_str(),
            });
        }

        let base_url = env::var("OPENAI_BASE_URL")
            .ok()
            .map(|value| value.trim().trim_end_matches('/').to_string())
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| "https://api.openai.com".to_string());

        Ok(Self {
            api_key,
            model,
            base_url,
        })
    }

    fn truncate_for_embedding(text: &str) -> std::borrow::Cow<'_, str> {
        const MAX_CHARS: usize = 20_000;
        if text.len() <= MAX_CHARS {
            return std::borrow::Cow::Borrowed(text);
        }
        let truncated = &text[..MAX_CHARS];
        let end = truncated
            .char_indices()
            .rev()
            .next()
            .map(|(i, c)| i + c.len_utf8())
            .unwrap_or(MAX_CHARS);
        std::borrow::Cow::Owned(text[..end].to_string())
    }

    fn embed(&self, text: &str) -> MemvidResult<Vec<f32>> {
        let text = Self::truncate_for_embedding(text);
        let url = format!("{}{}", self.base_url, OPENAI_EMBEDDINGS_PATH);
        let body = json!({
            "model": self.model.canonical_model_id(),
            "input": [&*text],
        });

        let response = ureq::post(&url)
            .set("Authorization", &format!("Bearer {}", self.api_key))
            .set("Content-Type", "application/json")
            .send_json(body);

        let response = match response {
            Ok(resp) => resp,
            Err(ureq::Error::Status(code, resp)) => {
                let message = resp.into_string().unwrap_or_default();
                return Err(MemvidCoreError::EmbeddingFailed {
                    reason: format!("OpenAI embeddings request failed ({code}): {message}")
                        .into_boxed_str(),
                });
            }
            Err(ureq::Error::Transport(err)) => {
                return Err(MemvidCoreError::EmbeddingFailed {
                    reason: format!("OpenAI embeddings transport error: {err}").into_boxed_str(),
                });
            }
        };

        let body: Value = response
            .into_json()
            .map_err(|err| MemvidCoreError::EmbeddingFailed {
                reason: format!("failed to parse OpenAI embeddings response: {err}")
                    .into_boxed_str(),
            })?;

        let embedding = body
            .get("data")
            .and_then(|data| data.as_array())
            .and_then(|items| items.first())
            .and_then(|item| item.get("embedding"))
            .and_then(|value| value.as_array())
            .ok_or_else(|| MemvidCoreError::EmbeddingFailed {
                reason: format!(
                    "OpenAI embeddings response missing data[0].embedding: {}",
                    serde_json::to_string(&body).unwrap_or_default()
                )
                .into_boxed_str(),
            })?;

        let mut vec = Vec::with_capacity(embedding.len());
        for value in embedding {
            let Some(f) = value.as_f64() else {
                return Err(MemvidCoreError::EmbeddingFailed {
                    reason: "OpenAI embeddings response contained non-numeric values"
                        .to_string()
                        .into_boxed_str(),
                });
            };
            vec.push(f as f32);
        }

        if vec.is_empty() {
            return Err(MemvidCoreError::EmbeddingFailed {
                reason: "OpenAI embeddings response returned empty vector"
                    .to_string()
                    .into_boxed_str(),
            });
        }
        Ok(vec)
    }

    fn embed_batch(&self, texts: Vec<String>) -> MemvidResult<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let input_len = texts.len();
        let inputs: Vec<String> = texts
            .into_iter()
            .map(|text| Self::truncate_for_embedding(&text).into_owned())
            .collect();

        let url = format!("{}{}", self.base_url, OPENAI_EMBEDDINGS_PATH);
        let body = json!({
            "model": self.model.canonical_model_id(),
            "input": inputs,
        });

        let response = ureq::post(&url)
            .set("Authorization", &format!("Bearer {}", self.api_key))
            .set("Content-Type", "application/json")
            .send_json(body);

        let response = match response {
            Ok(resp) => resp,
            Err(ureq::Error::Status(code, resp)) => {
                let message = resp.into_string().unwrap_or_default();
                return Err(MemvidCoreError::EmbeddingFailed {
                    reason: format!("OpenAI embeddings request failed ({code}): {message}")
                        .into_boxed_str(),
                });
            }
            Err(ureq::Error::Transport(err)) => {
                return Err(MemvidCoreError::EmbeddingFailed {
                    reason: format!("OpenAI embeddings transport error: {err}").into_boxed_str(),
                });
            }
        };

        let body: Value = response
            .into_json()
            .map_err(|err| MemvidCoreError::EmbeddingFailed {
                reason: format!("failed to parse OpenAI embeddings response: {err}")
                    .into_boxed_str(),
            })?;

        let data = body
            .get("data")
            .and_then(|value| value.as_array())
            .ok_or_else(|| MemvidCoreError::EmbeddingFailed {
                reason: format!(
                    "OpenAI embeddings response missing data array: {}",
                    serde_json::to_string(&body).unwrap_or_default()
                )
                .into_boxed_str(),
            })?;

        let expected_len = self.model.dimensions();
        let mut results: Vec<Option<Vec<f32>>> = vec![None; input_len];

        for item in data {
            let index = item
                .get("index")
                .and_then(|value| value.as_u64())
                .ok_or_else(|| MemvidCoreError::EmbeddingFailed {
                    reason: format!(
                        "OpenAI embeddings response missing item.index: {}",
                        serde_json::to_string(&body).unwrap_or_default()
                    )
                    .into_boxed_str(),
                })? as usize;
            if index >= input_len {
                return Err(MemvidCoreError::EmbeddingFailed {
                    reason: format!(
                        "OpenAI embeddings response returned out-of-range index {index} (batch size {input_len})"
                    )
                    .into_boxed_str(),
                });
            }

            let embedding = item
                .get("embedding")
                .and_then(|value| value.as_array())
                .ok_or_else(|| MemvidCoreError::EmbeddingFailed {
                    reason: format!(
                        "OpenAI embeddings response missing item.embedding: {}",
                        serde_json::to_string(&body).unwrap_or_default()
                    )
                    .into_boxed_str(),
                })?;

            let mut vec = Vec::with_capacity(embedding.len());
            for value in embedding {
                let Some(f) = value.as_f64() else {
                    return Err(MemvidCoreError::EmbeddingFailed {
                        reason: "OpenAI embeddings response contained non-numeric values"
                            .to_string()
                            .into_boxed_str(),
                    });
                };
                vec.push(f as f32);
            }

            if vec.is_empty() || vec.len() != expected_len {
                return Err(MemvidCoreError::EmbeddingFailed {
                    reason: format!(
                        "OpenAI embeddings response returned invalid vector length (expected {expected_len}, got {})",
                        vec.len()
                    )
                    .into_boxed_str(),
                });
            }

            results[index] = Some(vec);
        }

        let mut embeddings = Vec::with_capacity(results.len());
        for value in results {
            let Some(vec) = value else {
                return Err(MemvidCoreError::EmbeddingFailed {
                    reason: "OpenAI embeddings response returned incomplete batch"
                        .to_string()
                        .into_boxed_str(),
                });
            };
            embeddings.push(vec);
        }
        Ok(embeddings)
    }
}

#[derive(Clone)]
enum EmbeddingBackend {
    #[cfg(feature = "fastembed")]
    FastEmbed(std::sync::Arc<std::sync::Mutex<fastembed::TextEmbedding>>),
    OpenAI(std::sync::Arc<OpenAIEmbeddingProvider>),
}

#[derive(Clone)]
struct EmbeddingRuntime {
    backend: EmbeddingBackend,
    model: EmbeddingModelChoice,
    dimension: usize,
}

impl EmbeddingRuntime {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn model_choice(&self) -> EmbeddingModelChoice {
        self.model
    }

    fn embed(&self, text: &str) -> MemvidResult<Vec<f32>> {
        match &self.backend {
            #[cfg(feature = "fastembed")]
            EmbeddingBackend::FastEmbed(model) => {
                let mut guard = model.lock().map_err(|_| MemvidCoreError::EmbeddingFailed {
                    reason: "fastembed runtime poisoned".to_string().into_boxed_str(),
                })?;
                let outputs = guard.embed(vec![text.to_string()], None).map_err(|err| {
                    MemvidCoreError::EmbeddingFailed {
                        reason: format!("failed to compute embedding with fastembed: {err}")
                            .into_boxed_str(),
                    }
                })?;
                outputs
                    .into_iter()
                    .next()
                    .ok_or_else(|| MemvidCoreError::EmbeddingFailed {
                        reason: "fastembed returned no embedding output"
                            .to_string()
                            .into_boxed_str(),
                    })
            }
            EmbeddingBackend::OpenAI(provider) => provider.embed(text),
        }
    }

    fn embed_batch(&self, texts: Vec<String>) -> MemvidResult<Vec<Vec<f32>>> {
        match &self.backend {
            #[cfg(feature = "fastembed")]
            EmbeddingBackend::FastEmbed(model) => {
                let mut guard = model.lock().map_err(|_| MemvidCoreError::EmbeddingFailed {
                    reason: "fastembed runtime poisoned".to_string().into_boxed_str(),
                })?;
                guard
                    .embed(texts, None)
                    .map_err(|err| MemvidCoreError::EmbeddingFailed {
                        reason: format!("failed to compute embeddings with fastembed: {err}")
                            .into_boxed_str(),
                    })
            }
            EmbeddingBackend::OpenAI(provider) => provider.embed_batch(texts),
        }
    }
}

impl VecEmbedder for EmbeddingRuntime {
    fn embed_query(&self, text: &str) -> MemvidResult<Vec<f32>> {
        self.embed(text)
    }

    fn embedding_dimension(&self) -> usize {
        self.dimension()
    }
}

fn expand_path(value: &str) -> MemvidResult<PathBuf> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(MemvidCoreError::InvalidQuery {
            reason: "empty path".to_string(),
        });
    }

    if trimmed.starts_with("~/") {
        if let Ok(home) = env::var("HOME") {
            return Ok(PathBuf::from(home).join(&trimmed[2..]));
        }
    }
    Ok(PathBuf::from(trimmed))
}

fn resolve_models_dir() -> MemvidResult<PathBuf> {
    let raw = env::var("MEMVID_MODELS_DIR").unwrap_or_else(|_| "~/.memvid/models".to_string());
    expand_path(&raw)
}

fn is_offline() -> bool {
    env::var("MEMVID_OFFLINE")
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes"
            )
        })
        .unwrap_or(false)
}

#[cfg(feature = "fastembed")]
fn instantiate_fastembed_runtime(model: EmbeddingModelChoice) -> MemvidResult<EmbeddingRuntime> {
    use fastembed::{InitOptions, TextEmbedding};
    use std::fs;

    let cache_dir = resolve_models_dir()?;
    fs::create_dir_all(&cache_dir)?;

    if is_offline() {
        let mut entries = fs::read_dir(&cache_dir)?;
        if entries.next().is_none() {
            return Err(MemvidCoreError::EmbeddingFailed {
                reason: "semantic embeddings unavailable while offline; allow one connected run so fastembed can cache model weights"
                    .to_string()
                    .into_boxed_str(),
            });
        }
    }

    let options = InitOptions::new(model.to_fastembed_model()?)
        .with_cache_dir(cache_dir)
        .with_show_download_progress(false);
    let mut backend =
        TextEmbedding::try_new(options).map_err(|err| MemvidCoreError::EmbeddingFailed {
            reason: format!(
                "failed to initialize embedding model '{}': {err}",
                model.name()
            )
            .into_boxed_str(),
        })?;

    let probe = backend
        .embed(vec!["memvid probe".to_string()], None)
        .map_err(|err| MemvidCoreError::EmbeddingFailed {
            reason: format!("failed to determine embedding dimension: {err}").into_boxed_str(),
        })?;
    let dimension = probe.first().map(|vec| vec.len()).unwrap_or(0);
    if dimension == 0 {
        return Err(MemvidCoreError::EmbeddingFailed {
            reason: "fastembed reported zero-length embeddings"
                .to_string()
                .into_boxed_str(),
        });
    }

    Ok(EmbeddingRuntime {
        backend: EmbeddingBackend::FastEmbed(std::sync::Arc::new(std::sync::Mutex::new(backend))),
        model,
        dimension,
    })
}

fn instantiate_openai_runtime(model: EmbeddingModelChoice) -> MemvidResult<EmbeddingRuntime> {
    let provider = OpenAIEmbeddingProvider::from_env(model)?;
    Ok(EmbeddingRuntime {
        backend: EmbeddingBackend::OpenAI(std::sync::Arc::new(provider)),
        model,
        dimension: model.dimensions(),
    })
}

fn load_embedding_runtime(model: EmbeddingModelChoice) -> MemvidResult<EmbeddingRuntime> {
    if model.is_openai() {
        return instantiate_openai_runtime(model);
    }
    #[cfg(feature = "fastembed")]
    {
        instantiate_fastembed_runtime(model)
    }
    #[cfg(not(feature = "fastembed"))]
    {
        Err(MemvidCoreError::EmbeddingFailed {
            reason: format!(
                "local embedding model '{}' requires the 'fastembed' feature which is not available on this platform; use OpenAI embeddings instead (openai-large, openai-small, openai-ada)",
                model.name()
            ).into_boxed_str(),
        })
    }
}

fn cached_embedding_runtime(model: EmbeddingModelChoice) -> MemvidResult<EmbeddingRuntime> {
    static CACHE: OnceLock<Mutex<HashMap<EmbeddingModelChoice, EmbeddingRuntime>>> =
        OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    {
        let guard = cache.lock().map_err(|_| MemvidCoreError::EmbeddingFailed {
            reason: "embedding runtime cache poisoned"
                .to_string()
                .into_boxed_str(),
        })?;
        if let Some(runtime) = guard.get(&model) {
            return Ok(runtime.clone());
        }
    }

    let runtime = load_embedding_runtime(model)?;
    let mut guard = cache.lock().map_err(|_| MemvidCoreError::EmbeddingFailed {
        reason: "embedding runtime cache poisoned"
            .to_string()
            .into_boxed_str(),
    })?;
    guard.insert(model, runtime.clone());
    Ok(runtime)
}

fn detect_embedding_model_for_memory(
    mem: &MemvidCore,
    model_override: Option<&str>,
) -> MemvidResult<EmbeddingModelChoice> {
    if let Some(model_str) = model_override {
        return EmbeddingModelChoice::parse(model_str);
    }

    match mem.embedding_identity_summary(10_000) {
        EmbeddingIdentitySummary::Unknown => {
            let dim = mem.effective_vec_index_dimension()?.unwrap_or(0);
            EmbeddingModelChoice::from_dimension(dim).ok_or_else(|| MemvidCoreError::EmbeddingFailed {
                reason: "unable to auto-detect embedding model (missing memvid.embedding.* metadata and unknown vector dimension); provide query_embedding_model"
                    .to_string()
                    .into_boxed_str(),
            })
        }
        EmbeddingIdentitySummary::Single(identity) => {
            if let Some(model) = identity.model.as_deref() {
                if let Ok(parsed) = EmbeddingModelChoice::parse(model) {
                    return Ok(parsed);
                }
            }
            let dim = identity.dimension.or(mem.effective_vec_index_dimension()?);
            dim.and_then(EmbeddingModelChoice::from_dimension)
                .ok_or_else(|| MemvidCoreError::EmbeddingFailed {
                    reason: "unable to auto-detect embedding model from identity metadata; provide query_embedding_model"
                        .to_string()
                        .into_boxed_str(),
                })
        }
        EmbeddingIdentitySummary::Mixed(_) => Err(MemvidCoreError::EmbeddingFailed {
            reason: "memory contains mixed embedding models; semantic queries are unsafe"
                .to_string()
                .into_boxed_str(),
        }),
    }
}

fn apply_embedding_identity_metadata_from_choice(
    options: &mut memvid_core::types::PutOptions,
    model: EmbeddingModelChoice,
    dimension: usize,
) {
    let provider = if model.is_openai() {
        "openai"
    } else {
        "fastembed"
    };
    options.extra_metadata.insert(
        MEMVID_EMBEDDING_PROVIDER_KEY.to_string(),
        provider.to_string(),
    );
    options.extra_metadata.insert(
        MEMVID_EMBEDDING_MODEL_KEY.to_string(),
        model.canonical_model_id().to_string(),
    );
    options.extra_metadata.insert(
        MEMVID_EMBEDDING_DIMENSION_KEY.to_string(),
        dimension.to_string(),
    );
}

fn embed_texts_batched(
    runtime: &EmbeddingRuntime,
    texts: Vec<String>,
) -> MemvidResult<Vec<Vec<f32>>> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }

    let batch_size = if runtime.model_choice().is_openai() {
        96
    } else {
        64
    };
    let total = texts.len();
    let mut remaining = texts;
    let mut embeddings = Vec::with_capacity(total);

    while !remaining.is_empty() {
        let take = remaining.len().min(batch_size);
        let batch: Vec<String> = remaining.drain(..take).collect();
        let mut batch_embeddings = runtime.embed_batch(batch)?;
        embeddings.append(&mut batch_embeddings);
    }

    if embeddings.len() != total {
        return Err(MemvidCoreError::EmbeddingFailed {
            reason: format!(
                "embedding runtime returned unexpected output count (expected {total}, got {})",
                embeddings.len()
            )
            .into_boxed_str(),
        });
    }
    Ok(embeddings)
}

fn payload_text_for_embedding(payload: &[u8]) -> Option<&str> {
    let text = std::str::from_utf8(payload).ok()?;
    let sample = &payload[..payload.len().min(8192)];
    if sample.contains(&0) {
        return None;
    }
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }
    Some(trimmed)
}

/// Clean up text content for display (normalize whitespace, remove common artifacts)
fn clean_text_for_display(text: &str) -> String {
    // Normalize whitespace: collapse multiple spaces/newlines into single space
    let cleaned: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    cleaned
}

pyo3::create_exception!(memvid_sdk, MemvidError, PyException);
pyo3::create_exception!(memvid_sdk, CapacityExceededError, MemvidError);
pyo3::create_exception!(memvid_sdk, TicketInvalidError, MemvidError);
pyo3::create_exception!(memvid_sdk, TicketReplayError, MemvidError);
pyo3::create_exception!(memvid_sdk, LexIndexDisabledError, MemvidError);
pyo3::create_exception!(memvid_sdk, TimeIndexMissingError, MemvidError);
pyo3::create_exception!(memvid_sdk, VerifyFailedError, MemvidError);
pyo3::create_exception!(memvid_sdk, LockedError, MemvidError);
pyo3::create_exception!(memvid_sdk, ApiKeyRequiredError, MemvidError);
pyo3::create_exception!(memvid_sdk, MemoryAlreadyBoundError, MemvidError);
pyo3::create_exception!(memvid_sdk, FrameNotFoundError, MemvidError);
pyo3::create_exception!(memvid_sdk, VecIndexDisabledError, MemvidError);
pyo3::create_exception!(memvid_sdk, CorruptFileError, MemvidError);
pyo3::create_exception!(memvid_sdk, FileNotFoundError, MemvidError);
pyo3::create_exception!(memvid_sdk, VecDimensionMismatchError, MemvidError);
pyo3::create_exception!(memvid_sdk, EmbeddingFailedError, MemvidError);
pyo3::create_exception!(memvid_sdk, EncryptionError, MemvidError);
pyo3::create_exception!(memvid_sdk, ClipIndexDisabledError, MemvidError);
pyo3::create_exception!(memvid_sdk, NerModelNotAvailableError, MemvidError);

#[cfg(feature = "parallel_segments")]
#[pyclass(name = "BuildOpts", module = "memvid_sdk")]
#[derive(Clone)]
struct PyBuildOpts {
    inner: BuildOpts,
}

#[cfg(feature = "parallel_segments")]
#[pymethods]
impl PyBuildOpts {
    #[new]
    #[pyo3(signature = (segment_tokens=None, segment_pages=None, threads=None, queue_depth=None))]
    fn new(
        segment_tokens: Option<usize>,
        segment_pages: Option<usize>,
        threads: Option<usize>,
        queue_depth: Option<usize>,
    ) -> Self {
        let mut opts = BuildOpts::default();
        if let Some(tokens) = segment_tokens {
            opts.segment_tokens = tokens;
        }
        if let Some(pages) = segment_pages {
            opts.segment_pages = pages;
        }
        if let Some(t) = threads {
            opts.threads = t;
        }
        if let Some(depth) = queue_depth {
            opts.queue_depth = depth;
        }
        opts.sanitize();
        Self { inner: opts }
    }

    #[getter]
    fn segment_tokens(&self) -> usize {
        self.inner.segment_tokens
    }

    #[getter]
    fn segment_pages(&self) -> usize {
        self.inner.segment_pages
    }

    #[getter]
    fn threads(&self) -> usize {
        self.inner.threads
    }

    #[getter]
    fn queue_depth(&self) -> usize {
        self.inner.queue_depth
    }

    fn __repr__(&self) -> String {
        format!(
            "BuildOpts(segment_tokens={}, segment_pages={}, threads={}, queue_depth={})",
            self.inner.segment_tokens,
            self.inner.segment_pages,
            self.inner.threads,
            self.inner.queue_depth
        )
    }
}

#[cfg(feature = "parallel_segments")]
impl PyBuildOpts {
    fn to_core(&self) -> BuildOpts {
        self.inner.clone()
    }
}

#[derive(Debug, Clone, Copy)]
enum OpenMode {
    Create,
    Open,
    Auto,
    ReadOnly,
}

fn build_exception(_py: Python<'_>, ty: &PyType, code: &'static str, detail: String) -> PyErr {
    let message = format!("{code}: {detail}");
    match ty.call1((message,)) {
        Ok(obj) => {
            let _ = obj.setattr("code", code);
            PyErr::from_value(obj)
        }
        Err(_) => PyErr::new::<PyException, _>(format!("{code}: {detail}")),
    }
}

fn map_core_error(py: Python<'_>, err: MemvidCoreError) -> PyErr {
    match err {
        MemvidCoreError::CapacityExceeded { .. } => build_exception(
            py,
            py.get_type::<CapacityExceededError>(),
            "MV001",
            err.to_string(),
        ),
        MemvidCoreError::TicketSignatureInvalid { .. } => build_exception(
            py,
            py.get_type::<TicketInvalidError>(),
            "MV002",
            err.to_string(),
        ),
        MemvidCoreError::TicketSequence { .. } => build_exception(
            py,
            py.get_type::<TicketReplayError>(),
            "MV003",
            err.to_string(),
        ),
        MemvidCoreError::LexNotEnabled => build_exception(
            py,
            py.get_type::<LexIndexDisabledError>(),
            "MV004",
            err.to_string(),
        ),
        MemvidCoreError::InvalidTimeIndex { .. } => build_exception(
            py,
            py.get_type::<TimeIndexMissingError>(),
            "MV005",
            err.to_string(),
        ),
        MemvidCoreError::Locked(locked) => {
            let owner_hint = locked.owner.as_ref().map(|hint| {
                let pid = hint.pid.map(|pid| pid.to_string());
                let cmd = hint.cmd.clone();
                match (pid, cmd) {
                    (Some(pid), Some(cmd)) => format!("pid={pid} cmd={cmd}"),
                    (Some(pid), None) => format!("pid={pid}"),
                    (None, Some(cmd)) => format!("cmd={cmd}"),
                    (None, None) => "owner=unknown".to_string(),
                }
            });
            let suffix = owner_hint
                .map(|value| format!(" ({value})"))
                .unwrap_or_default();
            build_exception(
                py,
                py.get_type::<LockedError>(),
                "MV007",
                format!("{}{suffix}", locked.message),
            )
        }
        MemvidCoreError::Lock(message) => {
            build_exception(py, py.get_type::<LockedError>(), "MV007", message)
        }
        MemvidCoreError::ApiKeyRequired { .. } => build_exception(
            py,
            py.get_type::<ApiKeyRequiredError>(),
            "MV008",
            err.to_string(),
        ),
        MemvidCoreError::MemoryAlreadyBound { .. } => build_exception(
            py,
            py.get_type::<MemoryAlreadyBoundError>(),
            "MV009",
            err.to_string(),
        ),
        MemvidCoreError::FrameNotFound { .. } | MemvidCoreError::FrameNotFoundByUri { .. } => {
            build_exception(
                py,
                py.get_type::<FrameNotFoundError>(),
                "MV010",
                err.to_string(),
            )
        }
        MemvidCoreError::VecNotEnabled => build_exception(
            py,
            py.get_type::<VecIndexDisabledError>(),
            "MV011",
            err.to_string(),
        ),
        MemvidCoreError::EncryptedFile { .. } => build_exception(
            py,
            py.get_type::<EncryptionError>(),
            "MV016",
            err.to_string(),
        ),
        MemvidCoreError::ClipNotEnabled => build_exception(
            py,
            py.get_type::<ClipIndexDisabledError>(),
            "MV018",
            err.to_string(),
        ),
        MemvidCoreError::VecDimensionMismatch { .. } => build_exception(
            py,
            py.get_type::<VecDimensionMismatchError>(),
            "MV014",
            err.to_string(),
        ),
        MemvidCoreError::EmbeddingFailed { .. } => build_exception(
            py,
            py.get_type::<EmbeddingFailedError>(),
            "MV015",
            err.to_string(),
        ),
        MemvidCoreError::NerModelNotAvailable { .. } => build_exception(
            py,
            py.get_type::<NerModelNotAvailableError>(),
            "MV017",
            err.to_string(),
        ),
        MemvidCoreError::InvalidHeader { .. }
        | MemvidCoreError::InvalidToc { .. }
        | MemvidCoreError::ChecksumMismatch { .. }
        | MemvidCoreError::WalCorruption { .. }
        | MemvidCoreError::ManifestWalCorrupted { .. }
        | MemvidCoreError::AuxiliaryFileDetected { .. } => build_exception(
            py,
            py.get_type::<CorruptFileError>(),
            "MV012",
            err.to_string(),
        ),
        MemvidCoreError::Io { ref source, .. } if source.kind() == ErrorKind::NotFound => {
            build_exception(
                py,
                py.get_type::<FileNotFoundError>(),
                "MV013",
                format!("I/O error: {source}"),
            )
        }
        other => build_exception(py, py.get_type::<MemvidError>(), "MV999", other.to_string()),
    }
}

#[cfg(feature = "encryption")]
fn map_capsule_error(py: Python<'_>, err: CapsuleError) -> PyErr {
    match err {
        CapsuleError::Io { ref source, .. } if source.kind() == ErrorKind::NotFound => {
            build_exception(
                py,
                py.get_type::<FileNotFoundError>(),
                "MV013",
                format!("I/O error: {source}"),
            )
        }
        other => build_exception(
            py,
            py.get_type::<EncryptionError>(),
            "MV016",
            other.to_string(),
        ),
    }
}

fn model_error_to_py(err: ModelRunError) -> PyErr {
    match err {
        ModelRunError::UnsupportedModel(model) => {
            PyValueError::new_err(format!("unsupported model '{model}'"))
        }
        ModelRunError::AssetsMissing { model, missing } => {
            let paths: Vec<_> = missing
                .into_iter()
                .map(|path| path.display().to_string())
                .collect();
            PyValueError::new_err(format!(
                "model '{model}' missing required assets: {}",
                paths.join(", ")
            ))
        }
        ModelRunError::Runtime(err) => PyRuntimeError::new_err(err.to_string()),
    }
}

#[pyclass(name = "_MemvidCore", module = "memvid_sdk")]
pub struct MemvidCorePy {
    inner: Option<MemvidCore>,
    path: PathBuf,
    vec_available: bool,
    read_only: bool,
    lock_timeout_ms: u64,
    force_stale: bool,
    command: String,
    api_key: Option<String>,
    capacity_limit: Option<u64>,
    capacity_checked: bool,
}

impl MemvidCorePy {
    fn core_err(err: MemvidCoreError) -> PyErr {
        Python::with_gil(|py| map_core_error(py, err))
    }

    fn build_command_label() -> String {
        let args = env::args().collect::<Vec<_>>().join(" ");
        format!("python:{} {}", process::id(), args)
    }

    fn apply_lock_settings(
        mem: &mut MemvidCore,
        timeout_ms: u64,
        force_stale: bool,
        command: &str,
    ) {
        let settings = mem.lock_settings_mut();
        settings.timeout_ms = timeout_ms;
        settings.force_stale = force_stale;
        if settings.command.as_deref() != Some(command) {
            settings.command = Some(command.to_string());
        }
    }

    fn apply_default_lock_settings(mem: &mut MemvidCore) {
        let command = Self::build_command_label();
        Self::apply_lock_settings(mem, DEFAULT_LOCK_TIMEOUT_MS, false, &command);
    }

    #[cfg(feature = "parallel_segments")]
    fn commit_after_put(mem: &mut MemvidCore, preference: Option<bool>) -> PyResult<()> {
        let use_parallel = preference.unwrap_or_else(parallel_env_default);
        if use_parallel {
            mem.commit_parallel(BuildOpts::default())
                .map_err(Self::core_err)
        } else {
            mem.commit().map_err(Self::core_err)
        }
    }

    #[cfg(not(feature = "parallel_segments"))]
    fn commit_after_put(mem: &mut MemvidCore, _preference: Option<bool>) -> PyResult<()> {
        mem.commit().map_err(Self::core_err)
    }

    fn open_inner(path: &Path, mode: OpenMode) -> MemvidResult<MemvidCore> {
        match mode {
            OpenMode::Create => MemvidCore::create(path),
            OpenMode::Open => MemvidCore::open(path),
            OpenMode::ReadOnly => MemvidCore::open_read_only(path),
            OpenMode::Auto => match MemvidCore::open(path) {
                Ok(mem) => Ok(mem),
                Err(MemvidCoreError::Io { source, .. }) if source.kind() == ErrorKind::NotFound => {
                    MemvidCore::create(path)
                }
                Err(err) => Err(err),
            },
        }
    }

    fn try_enable_vec(inner: &mut MemvidCore) -> MemvidResult<()> {
        match catch_unwind(AssertUnwindSafe(|| inner.enable_vec())) {
            Ok(result) => result,
            Err(_) => Err(MemvidCoreError::FeatureUnavailable { feature: "vec" }),
        }
    }

    fn configure_indexes(
        inner: &mut MemvidCore,
        enable_lex: bool,
        enable_vec: bool,
    ) -> MemvidResult<()> {
        // Always call enable_lex/enable_vec if requested, even if index exists
        // These methods are idempotent and will set the necessary flags
        if enable_lex {
            const RETRIES: usize = 20;
            let mut attempt = 0;
            loop {
                match inner.enable_lex() {
                    Ok(_) => break,
                    Err(MemvidCoreError::Tantivy { .. }) if attempt < RETRIES => {
                        attempt += 1;
                        thread::sleep(Duration::from_millis(100));
                        continue;
                    }
                    Err(err) => return Err(err),
                }
            }
        }
        if enable_vec {
            Self::try_enable_vec(inner)?;
        }
        Ok(())
    }

    fn ensure_indexes_available(
        inner: &MemvidCore,
        enable_lex: bool,
        enable_vec: bool,
    ) -> MemvidResult<()> {
        let stats = inner.stats()?;
        if enable_lex && !stats.has_lex_index {
            return Err(MemvidCoreError::LexNotEnabled);
        }
        if enable_vec && !stats.has_vec_index {
            return Err(MemvidCoreError::VecNotEnabled);
        }
        Ok(())
    }

    fn ensure_open_mut(&mut self) -> PyResult<&mut MemvidCore> {
        let timeout = self.lock_timeout_ms;
        let force = self.force_stale;
        let command = self.command.clone();
        let inner = self
            .inner
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("memvid handle is closed"))?;
        Self::apply_lock_settings(inner, timeout, force, &command);
        Ok(inner)
    }

    /// Fetch capacity from API
    fn fetch_capacity_from_api(&mut self) -> PyResult<()> {
        let api_key = match &self.api_key {
            Some(key) => key,
            None => return Ok(()), // No API key, use free tier
        };

        let api_url = env::var("MEMVID_API_URL").unwrap_or_else(|_| DEFAULT_API_URL.to_string());
        let url = format!("{}/auth/capacity", api_url);

        match ureq::get(&url).set("X-API-Key", api_key).call() {
            Ok(response) => {
                let body: serde_json::Value = response.into_json().map_err(|e| {
                    PyRuntimeError::new_err(format!("Failed to parse API response: {}", e))
                })?;

                if let Some(data) = body.get("data") {
                    self.capacity_limit = data
                        .get("storage_limit_bytes")
                        .and_then(|v| v.as_i64())
                        .map(|v| v as u64);
                    self.capacity_checked = true;
                }
                Ok(())
            }
            Err(_) => {
                // Network error - silently fall back to free tier
                Ok(())
            }
        }
    }

    /// Check if write operation is allowed
    fn check_can_write(&mut self) -> PyResult<()> {
        // First check subscription status (blocks expired subscriptions)
        let subscription = require_active_subscription(&self.api_key)?;

        // Get the file's capacity from stats (not physical file size)
        let inner = self.ensure_open_ref()?;
        let stats = inner.stats().map_err(Self::core_err)?;
        let file_capacity = stats.capacity_bytes;

        // Use subscription capacity if available, otherwise fall back to stored limit
        let limit = match (&subscription, &self.api_key, &self.capacity_limit) {
            (Some(sub), _, _) => sub.capacity_bytes,
            (None, Some(_), Some(limit)) => *limit,
            (None, Some(_), None) => return Ok(()), // Unlimited
            (None, None, _) => FREE_TIER_LIMIT_BYTES,
        };

        if file_capacity > limit {
            if self.api_key.is_none() {
                return Err(ApiKeyRequiredError::new_err(format!(
                    "File capacity ({:.2} GB) exceeds 9999 GB free tier limit. \
                     Set MEMVID_API_KEY environment variable or pass api_key parameter. \
                     Get your API key at https://memvid.com/dashboard/api-keys",
                    file_capacity as f64 / 1e9
                )));
            } else {
                return Err(CapacityExceededError::new_err(format!(
                    "File capacity ({:.2} GB) exceeds plan limit ({:.2} GB). \
                     Upgrade your plan at https://memvid.com/dashboard/plan",
                    file_capacity as f64 / 1e9,
                    limit as f64 / 1e9
                )));
            }
        }

        Ok(())
    }

    fn ensure_open_ref(&self) -> PyResult<&MemvidCore> {
        self.inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("memvid handle is closed"))
    }

    fn ensure_indexes(&mut self, enable_lex: bool, enable_vec: bool) -> PyResult<()> {
        loop {
            let request_vec = enable_vec && self.vec_available && !self.read_only;
            let request_lex = enable_lex && !self.read_only;
            let result = {
                let inner = self.ensure_open_mut()?;
                Self::configure_indexes(inner, request_lex, request_vec)
            };
            match result {
                Ok(_) => {
                    if self.read_only {
                        let stats = self.ensure_open_ref()?.stats().map_err(Self::core_err)?;
                        if enable_lex && !stats.has_lex_index {
                            return Err(Self::core_err(MemvidCoreError::LexNotEnabled));
                        }
                        if enable_vec && !stats.has_vec_index {
                            return Err(Self::core_err(MemvidCoreError::VecNotEnabled));
                        }
                    }
                    return Ok(());
                }
                Err(MemvidCoreError::FeatureUnavailable { feature })
                    if feature == "vec" && request_vec =>
                {
                    self.vec_available = false;
                    drop(self.inner.take());
                    let reopened =
                        Self::open_inner(&self.path, OpenMode::Open).map_err(Self::core_err)?;
                    let timeout = self.lock_timeout_ms;
                    let force = self.force_stale;
                    let command = self.command.clone();
                    self.inner = Some(reopened);
                    if let Some(inner) = self.inner.as_mut() {
                        Self::apply_lock_settings(inner, timeout, force, &command);
                    }
                    continue;
                }
                Err(_err @ MemvidCoreError::Lock(_)) if self.read_only => {
                    drop(self.inner.take());
                    let reopened =
                        Self::open_inner(&self.path, OpenMode::Open).map_err(Self::core_err)?;
                    let timeout = self.lock_timeout_ms;
                    let force = self.force_stale;
                    let command = self.command.clone();
                    self.inner = Some(reopened);
                    if let Some(inner) = self.inner.as_mut() {
                        Self::apply_lock_settings(inner, timeout, force, &command);
                    }
                    continue;
                }
                Err(err) => return Err(Self::core_err(err)),
            }
        }
    }

    fn from_path(
        path: PathBuf,
        mode: OpenMode,
        enable_lex: bool,
        enable_vec: bool,
        read_only: bool,
        force_writable: bool,
        lock_timeout_ms: u64,
        force_stale: bool,
        api_key: Option<String>,
    ) -> PyResult<Self> {
        let effective_mode = if read_only {
            match mode {
                OpenMode::Create => OpenMode::Create,
                OpenMode::Open | OpenMode::Auto | OpenMode::ReadOnly => OpenMode::ReadOnly,
            }
        } else {
            mode
        };
        if force_writable && read_only {
            return Err(PyValueError::new_err(
                "force_writable cannot be combined with read_only=True",
            ));
        }
        let mut effective_read_only = read_only;
        let command = Self::build_command_label();
        let mut inner = if force_writable {
            let _ = Self::open_inner(&path, OpenMode::ReadOnly).map_err(Self::core_err)?;
            let mut writable = Self::open_inner(&path, OpenMode::Open).map_err(Self::core_err)?;
            Self::apply_lock_settings(&mut writable, lock_timeout_ms, force_stale, &command);
            effective_read_only = false;
            writable
        } else {
            let mut opened = Self::open_inner(&path, effective_mode).map_err(Self::core_err)?;
            let applied_force = force_stale && !effective_read_only;
            Self::apply_lock_settings(&mut opened, lock_timeout_ms, applied_force, &command);
            opened
        };
        let applied_force = force_stale && !effective_read_only;
        let mut vec_available = enable_vec;
        // Auto-detect vec from disk state: if file already has a vec index, enable vec
        {
            let stats = inner.stats().map_err(Self::core_err)?;
            if stats.has_vec_index {
                vec_available = true;
            }
        }
        let request_lex = enable_lex && !effective_read_only;
        let request_vec = enable_vec && !effective_read_only;
        match Self::configure_indexes(&mut inner, request_lex, request_vec) {
            Ok(()) => {}
            Err(MemvidCoreError::FeatureUnavailable { feature }) if feature == "vec" => {
                vec_available = false;
                let mut reopened =
                    Self::open_inner(&path, OpenMode::Open).map_err(Self::core_err)?;
                Self::apply_lock_settings(&mut reopened, lock_timeout_ms, applied_force, &command);
                inner = reopened;
            }
            Err(err) => return Err(Self::core_err(err)),
        }
        Self::apply_lock_settings(&mut inner, lock_timeout_ms, applied_force, &command);
        if effective_read_only {
            let stats = inner.stats().map_err(Self::core_err)?;
            if enable_lex && !stats.has_lex_index {
                return Err(Self::core_err(MemvidCoreError::LexNotEnabled));
            }
            if enable_vec && !stats.has_vec_index {
                return Err(Self::core_err(MemvidCoreError::VecNotEnabled));
            }
            vec_available = stats.has_vec_index;
        }
        let mut instance = Self {
            inner: Some(inner),
            path,
            vec_available,
            read_only: effective_read_only,
            lock_timeout_ms,
            force_stale: applied_force,
            command,
            api_key: api_key.clone(),
            capacity_limit: None,
            capacity_checked: false,
        };

        // Fetch capacity if API key is set
        if api_key.is_some() {
            let _ = instance.fetch_capacity_from_api();
        }

        Ok(instance)
    }

    fn ensure_mutable(&self) -> PyResult<()> {
        if self.read_only {
            Err(Python::with_gil(|py| {
                build_exception(
                    py,
                    py.get_type::<MemvidError>(),
                    "MV999",
                    "memvid handle opened with read_only=True; mutation is not allowed".to_string(),
                )
            }))
        } else {
            Ok(())
        }
    }
}

#[allow(non_local_definitions)]
#[pymethods]
impl MemvidCorePy {
    #[new]
    #[pyo3(signature = (path, mode = "auto", *, enable_lex = true, enable_vec = false, read_only = true, lock_timeout_ms = DEFAULT_LOCK_TIMEOUT_MS, force = None, force_writable = false, api_key = None))]
    fn new(
        path: String,
        mode: &str,
        enable_lex: bool,
        enable_vec: bool,
        read_only: bool,
        lock_timeout_ms: u64,
        force: Option<&str>,
        force_writable: bool,
        api_key: Option<String>,
    ) -> PyResult<Self> {
        // Get API key from parameter or environment variable
        let api_key = api_key.or_else(|| env::var("MEMVID_API_KEY").ok());

        let path_buf = PathBuf::from(path);
        let behavior = match mode {
            "auto" | "open_or_create" => OpenMode::Auto,
            "create" | "w" => OpenMode::Create,
            "open" | "r" => OpenMode::Open,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unsupported mode '{other}'; choose 'create', 'open', or 'auto'"
                )));
            }
        };
        let force_stale = match force {
            Some("stale_only") => true,
            Some(value) => {
                return Err(PyValueError::new_err(format!(
                    "unsupported force option '{value}'; expected 'stale_only'"
                )));
            }
            None => false,
        };
        Self::from_path(
            path_buf,
            behavior,
            enable_lex,
            enable_vec,
            read_only,
            force_writable,
            lock_timeout_ms,
            force_stale,
            api_key,
        )
    }

    #[classmethod]
    #[pyo3(signature = (path, *, enable_lex = true, enable_vec = false, mode = "auto", read_only = false, force_writable = false))]
    fn open(
        _cls: &PyType,
        py: Python<'_>,
        path: &str,
        enable_lex: bool,
        enable_vec: bool,
        mode: &str,
        read_only: bool,
        force_writable: bool,
    ) -> PyResult<Py<MemvidCorePy>> {
        let instance = Self::new(
            path.to_string(),
            mode,
            enable_lex,
            enable_vec,
            read_only,
            DEFAULT_LOCK_TIMEOUT_MS,
            None,
            force_writable,
            None, // api_key - will be read from env
        )?;
        Py::new(py, instance)
    }

    #[classmethod]
    #[pyo3(signature = (path, *, enable_lex = true, enable_vec = false))]
    fn create(
        _cls: &PyType,
        py: Python<'_>,
        path: &str,
        enable_lex: bool,
        enable_vec: bool,
    ) -> PyResult<Py<MemvidCorePy>> {
        let instance = Self::from_path(
            PathBuf::from(path),
            OpenMode::Create,
            enable_lex,
            enable_vec,
            false,
            false,
            DEFAULT_LOCK_TIMEOUT_MS,
            false,
            None, // api_key - will be read from env
        )?;
        Py::new(py, instance)
    }

    fn stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        let inner = self.ensure_open_ref()?;
        let stats = inner.stats().map_err(Self::core_err)?;
        let obj = stats_to_py(py, &stats)?;
        let dict: &PyDict = obj.as_ref(py).downcast()?;

        let effective_vec_dimension = inner
            .effective_vec_index_dimension()
            .map_err(Self::core_err)?;
        dict.set_item("effective_vec_dimension", effective_vec_dimension)?;

        let summary = inner.embedding_identity_summary(10_000);
        dict.set_item(
            "embedding_identity_summary",
            embedding_identity_summary_to_py(py, &summary)?,
        )?;

        Ok(obj)
    }

    fn put(&mut self, payload: &PyDict) -> PyResult<u64> {
        self.ensure_mutable()?;

        // Check capacity before write
        self.check_can_write()?;

        let request = parse_put_payload(payload)?;
        if request.enable_embedding && !self.vec_available {
            return Err(Self::core_err(MemvidCoreError::VecNotEnabled));
        }
        let enable_enrichment = request.enable_enrichment;
        self.ensure_indexes(true, request.enable_embedding)?;
        let inner = self.ensure_open_mut()?;

        // Get frame count before put - after commit, new frame_id = this count
        // (frame IDs are indices into the TOC frames array)
        let frame_id = inner.stats().map_err(Self::core_err)?.frame_count;

        let _wal_seq = perform_put(inner, &request)?;
        Self::commit_after_put(inner, request.parallel)?;

        // Run rules-based enrichment if enabled
        if enable_enrichment {
            if let Ok(frame) = inner.frame_by_id(frame_id) {
                let text = frame.search_text.as_deref().unwrap_or("");
                if !text.is_empty() {
                    let mut rules_engine = RulesEngine::new();
                    let _ = rules_engine.init();
                    let ctx = EnrichmentContext::new(
                        frame_id,
                        frame.uri.clone().unwrap_or_default(),
                        text.to_string(),
                        frame.title.clone(),
                        frame.timestamp,
                        None,
                    );
                    let result = rules_engine.enrich(&ctx);
                    let mut cards_added = false;
                    for card in result.cards {
                        if inner.put_memory_card(card).is_ok() {
                            cards_added = true;
                        }
                    }
                    if cards_added {
                        let _ = inner.commit();
                    }
                }
            }
        }

        Ok(frame_id)
    }

    /// Batch put operation - much faster than individual puts
    ///
    /// Ingests multiple documents in a single call, eliminating Python FFI overhead.
    /// Expected performance: 500-1000 docs/sec (100-200x faster than individual puts)
    ///
    /// # Arguments
    ///
    /// * `requests` - List of document dicts, each with keys: title, label, text, (optional: uri, metadata, tags, labels)
    /// * `embeddings` - Optional list of embedding vectors (one per document, or None)
    /// * `opts` - Optional dict with keys: compression_level (0-11), disable_auto_checkpoint (bool), skip_sync (bool)
    ///
    /// # Returns
    ///
    /// List of frame IDs for the ingested documents
    #[pyo3(signature = (requests, embeddings = None, opts = None))]
    fn put_many(
        &mut self,
        py: Python<'_>,
        requests: &PyList,
        embeddings: Option<&PyList>,
        opts: Option<&PyDict>,
    ) -> PyResult<Vec<u64>> {
        self.ensure_mutable()?;
        self.check_can_write()?;

        // Parse options
        // Check if OPENAI_API_KEY is available for auto-embedding (for default case)
        let default_has_openai_key = env::var("OPENAI_API_KEY")
            .ok()
            .map(|k| !k.trim().is_empty())
            .unwrap_or(false);

        let (put_opts, auto_enabled_via_openai) = if let Some(opts_dict) = opts {
            parse_put_many_opts(opts_dict)?
        } else {
            // When no options provided, auto-enable embeddings if OPENAI_API_KEY exists
            let mut opts = PutManyOpts::default();
            opts.enable_embedding = default_has_openai_key;
            (opts, default_has_openai_key)
        };

        let embedding_model_override = if let Some(opts_dict) = opts {
            opts_dict
                .get_item("embedding_model")?
                .map(|value| value.extract::<String>())
                .transpose()?
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty())
        } else {
            None
        };

        // Parse embeddings if provided
        let embeddings_vec: Option<Vec<Option<Vec<f32>>>> = if let Some(emb_list) = embeddings {
            if emb_list.len() != requests.len() {
                return Err(PyValueError::new_err(format!(
                    "Embeddings list length ({}) must match requests length ({})",
                    emb_list.len(),
                    requests.len()
                )));
            }
            let mut parsed_embeddings = Vec::with_capacity(emb_list.len());
            for i in 0..emb_list.len() {
                let item = emb_list.get_item(i)?;
                if item.is_none() {
                    parsed_embeddings.push(None);
                } else {
                    match item.extract::<Vec<f32>>() {
                        Ok(vec) => {
                            parsed_embeddings.push(Some(vec));
                        }
                        Err(_) => {
                            parsed_embeddings.push(None);
                        }
                    }
                }
            }
            Some(parsed_embeddings)
        } else {
            None
        };

        // Parse requests and attach embeddings by index
        let mut put_requests = Vec::with_capacity(requests.len());
        for i in 0..requests.len() {
            let item = requests.get_item(i)?;
            let doc_dict = item
                .downcast::<PyDict>()
                .map_err(|_| PyValueError::new_err("Each request must be a dict"))?;

            let mut request = parse_put_request(doc_dict)?;

            // Attach embedding from separate parameter
            if let Some(ref emb_vec) = embeddings_vec {
                if let Some(ref embedding) = emb_vec[i] {
                    request.embedding = Some(embedding.clone());
                } else {
                }
            }

            put_requests.push(request);
        }

        let wants_vec =
            put_opts.enable_embedding || put_requests.iter().any(|req| req.embedding.is_some());
        if wants_vec && !self.vec_available {
            return Err(Self::core_err(MemvidCoreError::VecNotEnabled));
        }
        self.ensure_indexes(true, wants_vec)?;

        let runtime = if put_opts.enable_embedding {
            // Choose model:
            // - If model specified, use it
            // - If auto-enabled via OPENAI_API_KEY (and no model specified), use OpenAISmall
            // - Otherwise fall back to BgeSmall (local)
            let model_choice = match embedding_model_override.as_deref() {
                Some(value) => EmbeddingModelChoice::parse(value).map_err(Self::core_err)?,
                None if auto_enabled_via_openai => EmbeddingModelChoice::OpenAISmall,
                None => EmbeddingModelChoice::BgeSmall,
            };
            Some(cached_embedding_runtime(model_choice).map_err(Self::core_err)?)
        } else {
            None
        };

        // eprintln!("[put_many FFI] Calling Rust put_parallel_inputs with {} requests", put_requests.len());

        // Convert PutRequests to ParallelInputs
        #[cfg(feature = "parallel_segments")]
        let mut parallel_inputs: Vec<ParallelInput> = Vec::with_capacity(put_requests.len());

        #[cfg(feature = "parallel_segments")]
        for req in &put_requests {
            // Create PutOptions from PutRequest fields
            let mut options = memvid_core::types::PutOptions::default();
            options.title = Some(req.title.clone());
            options.uri = req.uri.clone();
            options.tags = req.tags.clone();
            options.labels = req.labels.clone();
            options.auto_tag = put_opts.auto_tag;
            options.extract_dates = put_opts.extract_dates;
            options.enable_embedding = put_opts.enable_embedding;
            options.no_raw = put_opts.no_raw;

            // Split metadata into DocMetadata + extra_metadata (matches put()).
            if !req.metadata.is_empty() {
                let mut map = serde_json::Map::with_capacity(req.metadata.len());
                for (key, value) in req.metadata.iter() {
                    map.insert(key.clone(), value.clone());
                }
                let (doc_meta, extras) = split_metadata(Some(Value::Object(map)))?;
                options.metadata = doc_meta;

                for (key, value) in extras {
                    // Store embedding identity fields as raw strings (no JSON quotes),
                    // so memvid-core can auto-detect cross-SDK.
                    let is_identity_key = matches!(
                        key.as_str(),
                        MEMVID_EMBEDDING_PROVIDER_KEY
                            | MEMVID_EMBEDDING_MODEL_KEY
                            | MEMVID_EMBEDDING_DIMENSION_KEY
                            | MEMVID_EMBEDDING_NORMALIZED_KEY
                    );
                    if is_identity_key {
                        if let Value::String(s) = value {
                            options.extra_metadata.insert(key, s);
                            continue;
                        }
                    }
                    options.extra_metadata.insert(key, value.to_string());
                }
            }

            let mut embedding = req.embedding.clone();
            if embedding.is_none() {
                if let Some(runtime) = runtime.as_ref() {
                    apply_embedding_identity_metadata_from_choice(
                        &mut options,
                        runtime.model_choice(),
                        runtime.dimension(),
                    );
                    embedding = Some(runtime.embed(&req.text).map_err(Self::core_err)?);
                }
            }

            parallel_inputs.push(ParallelInput {
                payload: ParallelPayload::Bytes(req.text.as_bytes().to_vec()),
                options,
                embedding,
                chunk_embeddings: None,
            });
        }

        // Convert PutManyOpts to BuildOpts
        #[cfg(feature = "parallel_segments")]
        let build_opts = BuildOpts {
            zstd_level: put_opts.compression_level,
            ..BuildOpts::default()
        };

        // Release GIL during Rust processing
        #[cfg(feature = "parallel_segments")]
        let frame_ids = py.allow_threads(|| {
            let inner = self.ensure_open_mut()?;
            inner
                .put_parallel_inputs(&parallel_inputs, build_opts)
                .map_err(Self::core_err)
        })?;

        #[cfg(not(feature = "parallel_segments"))]
        let frame_ids: Vec<u64> = {
            return Err(PyRuntimeError::new_err(
                "put_many requires the parallel_segments feature to be enabled",
            ));
        };

        // Run rules-based enrichment if enabled
        #[cfg(feature = "parallel_segments")]
        if put_opts.enable_enrichment && !frame_ids.is_empty() {
            let inner = self.ensure_open_mut()?;
            let mut rules_engine = RulesEngine::new();
            let _ = rules_engine.init();
            let mut cards_added = false;

            for frame_id in &frame_ids {
                if let Ok(frame) = inner.frame_by_id(*frame_id) {
                    let text = frame.search_text.as_deref().unwrap_or("");
                    if !text.is_empty() {
                        let ctx = EnrichmentContext::new(
                            *frame_id,
                            frame.uri.clone().unwrap_or_default(),
                            text.to_string(),
                            frame.title.clone(),
                            frame.timestamp,
                            None,
                        );
                        let result = rules_engine.enrich(&ctx);
                        for card in result.cards {
                            if inner.put_memory_card(card).is_ok() {
                                cards_added = true;
                            }
                        }
                    }
                }
            }

            if cards_added {
                let _ = inner.commit();
            }
        }

        Ok(frame_ids)
    }

    /// Remove a frame by its ID.
    ///
    /// This performs a soft delete - the frame is marked as deleted and removed
    /// from search indexes, but remains in the file for audit purposes.
    ///
    /// Args:
    ///     frame_id: The frame ID to remove (returned by put())
    ///
    /// Returns:
    ///     The sequence number of the deletion operation
    ///
    /// Example:
    ///     >>> frame_id = mem.put(title="Doc", text="content")
    ///     >>> mem.remove(frame_id)  # Remove the frame
    fn remove(&mut self, frame_id: u64) -> PyResult<u64> {
        self.ensure_mutable()?;
        self.check_can_write()?;

        let inner = self.ensure_open_mut()?;
        let seq = inner.delete_frame(frame_id).map_err(Self::core_err)?;
        inner.commit().map_err(Self::core_err)?;

        Ok(seq)
    }

    #[pyo3(signature = (query, *, k = DEFAULT_FIND_K, snippet_chars = DEFAULT_FIND_SNIPPET, scope = None, cursor = None, mode = None, query_embedding = None, query_embedding_model = None, adaptive = None, min_relevancy = None, max_k = None, adaptive_strategy = None, as_of_frame = None, as_of_ts = None, acl_context = None, acl_enforcement_mode = "audit"))]
    fn find(
        &mut self,
        py: Python<'_>,
        query: &str,
        k: usize,
        snippet_chars: usize,
        scope: Option<&str>,
        cursor: Option<&str>,
        mode: Option<&str>,
        query_embedding: Option<Vec<f32>>,
        query_embedding_model: Option<&str>,
        adaptive: Option<bool>,
        min_relevancy: Option<f64>,
        max_k: Option<usize>,
        adaptive_strategy: Option<&str>,
        as_of_frame: Option<u64>,
        as_of_ts: Option<i64>,
        acl_context: Option<&PyDict>,
        acl_enforcement_mode: &str,
    ) -> PyResult<PyObject> {
        // Check subscription before allowing find operation
        require_active_subscription(&self.api_key)?;

        let mode_normalized = mode.unwrap_or("auto").to_ascii_lowercase();
        let acl_context = parse_acl_context_py(acl_context)?;
        let acl_enforcement_mode = parse_acl_enforcement_mode(acl_enforcement_mode)?;
        let model_override = query_embedding_model
            .map(|value| value.trim())
            .filter(|value| !value.is_empty());
        let has_override = model_override.is_some();

        let limit = snippet_chars.min(MAX_HIT_SNIPPET_CHARS);

        match mode_normalized.as_str() {
            "lex" | "lexical" => {
                self.ensure_indexes(true, false)?;
                let inner = self.ensure_open_mut()?;
                let request = SearchRequest {
                    query: query.to_string(),
                    top_k: k,
                    snippet_chars,
                    uri: None,
                    scope: scope.map(|s| s.to_string()),
                    cursor: cursor.map(|c| c.to_string()),
                    temporal: None,
                    as_of_frame,
                    as_of_ts,
                    no_sketch: false,
                    acl_context: acl_context.clone(),
                    acl_enforcement_mode,
                };
                let response = inner.search(request).map_err(Self::core_err)?;
                build_find_result(py, inner, response, limit)
            }
            "sem" | "semantic" => {
                if !self.vec_available {
                    return Err(Self::core_err(MemvidCoreError::VecNotEnabled));
                }
                self.ensure_indexes(false, true)?;
                let inner = self.ensure_open_mut()?;

                let embedding = match query_embedding {
                    Some(embedding) => embedding,
                    None => {
                        let model = detect_embedding_model_for_memory(inner, model_override)
                            .map_err(Self::core_err)?;
                        let runtime = cached_embedding_runtime(model).map_err(Self::core_err)?;
                        runtime.embed(query).map_err(Self::core_err)?
                    }
                };
                if adaptive.unwrap_or(false) {
                    let min_relevancy = min_relevancy.unwrap_or(0.5) as f32;
                    let max_k = max_k.unwrap_or(100);
                    let strategy = parse_cutoff_strategy(adaptive_strategy, min_relevancy)
                        .map_err(Self::core_err)?;
                    let config = AdaptiveConfig {
                        enabled: true,
                        max_results: max_k,
                        min_results: 1,
                        strategy,
                        normalize_scores: true,
                    };

                    let start = std::time::Instant::now();
                    let result = inner
                        .search_adaptive_acl(
                            query,
                            &embedding,
                            config,
                            snippet_chars,
                            scope,
                            acl_context.as_ref(),
                            acl_enforcement_mode,
                        )
                        .map_err(Self::core_err)?;
                    let elapsed_ms = start.elapsed().as_millis();

                    // Normalize cosine similarity scores from [-1, 1] to [0, 1]
                    let hits: Vec<_> = result
                        .results
                        .into_iter()
                        .map(|mut hit| {
                            if let Some(score) = hit.score {
                                hit.score = Some((score + 1.0) / 2.0);
                            }
                            hit
                        })
                        .collect();
                    let mut response = SearchResponse {
                        query: query.to_string(),
                        elapsed_ms,
                        total_hits: hits.len(),
                        params: SearchParams {
                            top_k: hits.len(),
                            snippet_chars,
                            cursor: None,
                        },
                        hits,
                        context: String::new(),
                        next_cursor: None,
                        engine: SearchEngineKind::Hybrid,
                        stale_index_skips: 0,
                    };
                    response.context = build_context_for_hits(&response.hits);
                    build_find_result(py, inner, response, limit)
                } else {
                    let mut response = inner
                        .vec_search_with_embedding_acl(
                            query,
                            &embedding,
                            k,
                            snippet_chars,
                            scope,
                            acl_context.as_ref(),
                            acl_enforcement_mode,
                        )
                        .map_err(Self::core_err)?;
                    // Normalize cosine similarity scores from [-1, 1] to [0, 1]
                    for hit in &mut response.hits {
                        if let Some(score) = hit.score {
                            hit.score = Some((score + 1.0) / 2.0);
                        }
                    }
                    build_find_result(py, inner, response, limit)
                }
            }
            "auto" | "hybrid" => {
                // Hybrid = lexical search + semantic rerank (best-effort).
                self.ensure_indexes(true, false)?;
                let vec_available = self.vec_available;
                let inner = self.ensure_open_mut()?;

                let request = SearchRequest {
                    query: query.to_string(),
                    top_k: k,
                    snippet_chars,
                    uri: None,
                    scope: scope.map(|s| s.to_string()),
                    cursor: cursor.map(|c| c.to_string()),
                    temporal: None,
                    as_of_frame,
                    as_of_ts,
                    no_sketch: false,
                    acl_context: acl_context.clone(),
                    acl_enforcement_mode,
                };
                let mut response = inner.search(request).map_err(Self::core_err)?;

                match query_embedding {
                    Some(query_embedding) => {
                        if !vec_available {
                            return Err(Self::core_err(MemvidCoreError::VecNotEnabled));
                        }

                        // Validate dimension mismatch up-front so we don't silently skip rerank.
                        if let Some(expected) = inner
                            .effective_vec_index_dimension()
                            .map_err(Self::core_err)?
                        {
                            if expected > 0 && query_embedding.len() as u32 != expected {
                                return Err(Self::core_err(
                                    MemvidCoreError::VecDimensionMismatch {
                                        expected,
                                        actual: query_embedding.len(),
                                    },
                                ));
                            }
                        }

                        // If lexical search returned results, try to rerank with semantic scores
                        if !response.hits.is_empty() {
                            if apply_semantic_rerank_with_embedding(
                                inner,
                                &query_embedding,
                                &mut response,
                            )
                            .map_err(Self::core_err)?
                            {
                                response.engine =
                                    memvid_core::types::search::SearchEngineKind::Hybrid;
                                response.context = build_context_for_hits(&response.hits);
                            }
                        } else {
                            // Lexical returned no results - fall back to pure vector search
                            let mut vec_response = inner
                                .vec_search_with_embedding_acl(
                                    query,
                                    &query_embedding,
                                    k,
                                    snippet_chars,
                                    scope,
                                    acl_context.as_ref(),
                                    acl_enforcement_mode,
                                )
                                .map_err(Self::core_err)?;
                            // Normalize cosine similarity scores from [-1, 1] to [0, 1]
                            for hit in &mut vec_response.hits {
                                if let Some(score) = hit.score {
                                    hit.score = Some((score + 1.0) / 2.0);
                                }
                            }
                            vec_response.engine =
                                memvid_core::types::search::SearchEngineKind::Hybrid;
                            vec_response.context = build_context_for_hits(&vec_response.hits);
                            return build_find_result(py, inner, vec_response, limit);
                        }
                    }
                    None => {
                        if !vec_available {
                            if has_override {
                                return Err(Self::core_err(MemvidCoreError::VecNotEnabled));
                            }
                            return build_find_result(py, inner, response, limit);
                        }

                        let has_vec_index = inner
                            .stats()
                            .map(|stats| stats.has_vec_index)
                            .unwrap_or(false);
                        if !has_vec_index {
                            if has_override {
                                return Err(Self::core_err(MemvidCoreError::VecNotEnabled));
                            }
                            return build_find_result(py, inner, response, limit);
                        }

                        let model = match detect_embedding_model_for_memory(inner, model_override) {
                            Ok(model) => model,
                            Err(err) => {
                                if has_override {
                                    return Err(Self::core_err(err));
                                }
                                return build_find_result(py, inner, response, limit);
                            }
                        };

                        let runtime = match cached_embedding_runtime(model) {
                            Ok(runtime) => runtime,
                            Err(err) => {
                                if has_override {
                                    return Err(Self::core_err(err));
                                }
                                return build_find_result(py, inner, response, limit);
                            }
                        };

                        let query_embedding = match runtime.embed(query) {
                            Ok(vec) => vec,
                            Err(err) => {
                                if has_override {
                                    return Err(Self::core_err(err));
                                }
                                return build_find_result(py, inner, response, limit);
                            }
                        };

                        if apply_semantic_rerank_with_embedding(
                            inner,
                            &query_embedding,
                            &mut response,
                        )
                        .map_err(Self::core_err)?
                        {
                            response.engine = memvid_core::types::search::SearchEngineKind::Hybrid;
                            response.context = build_context_for_hits(&response.hits);
                        }
                    }
                }

                build_find_result(py, inner, response, limit)
            }
            other => Err(PyValueError::new_err(format!("invalid mode: {other}"))),
        }
    }

    #[pyo3(signature = (question, *, k = DEFAULT_ASK_K, mode = "auto", snippet_chars = DEFAULT_ASK_SNIPPET, scope = None, since = None, until = None, context_only = false, query_embedding = None, query_embedding_model = None, adaptive = None, min_relevancy = None, max_k = None, adaptive_strategy = None, model = None, llm_context_chars = None, api_key = None, return_sources = false, show_chunks = false, acl_context = None, acl_enforcement_mode = "audit"))]
    fn ask(
        &mut self,
        py: Python<'_>,
        question: &str,
        k: usize,
        mode: &str,
        snippet_chars: usize,
        scope: Option<&str>,
        since: Option<i64>,
        until: Option<i64>,
        context_only: bool,
        query_embedding: Option<Vec<f32>>,
        query_embedding_model: Option<&str>,
        adaptive: Option<bool>,
        min_relevancy: Option<f64>,
        max_k: Option<usize>,
        adaptive_strategy: Option<&str>,
        model: Option<&str>,
        llm_context_chars: Option<usize>,
        api_key: Option<&str>,
        return_sources: bool,
        show_chunks: bool,
        acl_context: Option<&PyDict>,
        acl_enforcement_mode: &str,
    ) -> PyResult<PyObject> {
        // Check subscription before allowing ask operation
        require_active_subscription(&self.api_key)?;

        let requested_mode = parse_ask_mode(mode)?;
        let acl_context = parse_acl_context_py(acl_context)?;
        let acl_enforcement_mode = parse_acl_enforcement_mode(acl_enforcement_mode)?;
        let model_override = query_embedding_model
            .map(|value| value.trim())
            .filter(|value| !value.is_empty());
        let has_override = model_override.is_some();

        // Ask always needs lexical search for context assembly. Semantic retrieval is best-effort
        // in hybrid mode, and strict in semantic mode.
        self.ensure_indexes(true, false)?;
        let vec_available = self.vec_available;
        let inner = self.ensure_open_mut()?;

        let stats = inner.stats().map_err(Self::core_err)?;
        let has_vec_index = stats.has_vec_index;
        let mut ask_mode = requested_mode;
        let scope_owned = scope.map(|s| s.to_string());
        let adaptive_config = if adaptive.unwrap_or(false) {
            let min_relevancy = min_relevancy.unwrap_or(0.5) as f32;
            let max_k = max_k.unwrap_or(100);
            let strategy =
                parse_cutoff_strategy(adaptive_strategy, min_relevancy).map_err(Self::core_err)?;
            Some(AdaptiveConfig {
                enabled: true,
                max_results: max_k,
                min_results: 1,
                strategy,
                normalize_scores: true,
            })
        } else {
            None
        };

        let has_query_embedding = query_embedding.is_some();

        let build_request = |mode_override: AskMode, adaptive: Option<AdaptiveConfig>| AskRequest {
            question: question.to_string(),
            top_k: k,
            snippet_chars,
            uri: None,
            scope: scope_owned.clone(),
            cursor: None,
            start: since,
            end: until,
            temporal: None,
            context_only,
            mode: mode_override,
            as_of_frame: None,
            as_of_ts: None,
            adaptive,
            acl_context: acl_context.clone(),
            acl_enforcement_mode,
        };
        let runtime = if query_embedding.is_none() && ask_mode != AskMode::Lex {
            if !vec_available || !has_vec_index {
                if ask_mode == AskMode::Sem {
                    return Err(Self::core_err(MemvidCoreError::VecNotEnabled));
                }
                // Hybrid fallback.
                ask_mode = AskMode::Lex;
                None
            } else {
                match detect_embedding_model_for_memory(inner, model_override) {
                    Ok(model) => match cached_embedding_runtime(model) {
                        Ok(runtime) => Some(runtime),
                        Err(err) => {
                            if ask_mode == AskMode::Sem || has_override {
                                return Err(Self::core_err(err));
                            }
                            None
                        }
                    },
                    Err(err) => {
                        if ask_mode == AskMode::Sem || has_override {
                            return Err(Self::core_err(err));
                        }
                        None
                    }
                }
            }
        } else {
            None
        };

        if runtime.is_none() && query_embedding.is_none() && ask_mode != AskMode::Lex {
            // Hybrid fallback should be explicit.
            ask_mode = AskMode::Lex;
        }

        let effective_adaptive = if has_query_embedding || runtime.is_some() {
            adaptive_config.clone()
        } else {
            None
        };

        let mut response = if let Some(query_embedding) = query_embedding {
            if !vec_available || !has_vec_index {
                return Err(Self::core_err(MemvidCoreError::VecNotEnabled));
            }
            if let Some(expected) = inner
                .effective_vec_index_dimension()
                .map_err(Self::core_err)?
            {
                if expected > 0 && query_embedding.len() as u32 != expected {
                    return Err(Self::core_err(MemvidCoreError::VecDimensionMismatch {
                        expected,
                        actual: query_embedding.len(),
                    }));
                }
            }
            let embedder = StaticEmbedder {
                embedding: query_embedding,
            };
            inner
                .ask(build_request(ask_mode, effective_adaptive), Some(&embedder))
                .map_err(Self::core_err)?
        } else {
            match runtime.as_ref() {
                Some(runtime) => inner
                    .ask(build_request(ask_mode, effective_adaptive), Some(runtime))
                    .map_err(Self::core_err)?,
                None => inner
                    .ask::<dyn VecEmbedder>(build_request(ask_mode, None), None::<&dyn VecEmbedder>)
                    .map_err(Self::core_err)?,
            }
        };
        let mut model_info = None;
        if let Some(model_name) = model {
            match run_model_inference(
                model_name,
                question,
                &response.retrieval.context,
                response.retrieval.hits.as_slice(),
                llm_context_chars,
                api_key,
                None, // system_prompt_override
            ) {
                Ok(inference) => {
                    model_info = Some((
                        inference.answer.requested.clone(),
                        inference.answer.model.clone(),
                    ));
                    response.answer = Some(inference.answer.answer.clone());
                    response.retrieval.context = inference.context_body.clone();
                    apply_model_context_fragments(&mut response, inference.context_fragments);

                    // Record ASK action if a session is active
                    #[cfg(feature = "replay")]
                    if inner.is_recording() {
                        let retrieved_frames: Vec<u64> = response
                            .retrieval
                            .hits
                            .iter()
                            .map(|hit| hit.frame_id)
                            .collect();
                        inner.record_ask_action(
                            question,
                            model_name,
                            &inference.answer.model,
                            inference.answer.answer.as_bytes(),
                            0, // duration_ms not tracked
                            retrieved_frames,
                        );
                    }
                }
                Err(err) => return Err(model_error_to_py(err)),
            }
        }
        build_ask_result(py, inner, response, model_info, return_sources, show_chunks)
    }

    /// Generate an audit report for a question.
    ///
    /// This method performs a retrieval-augmented query and returns a structured
    /// report containing all sources used, along with provenance metadata.
    ///
    /// Args:
    ///     question: The question to audit
    ///     k: Number of sources to retrieve (default: 10)
    ///     mode: Search mode - "auto", "lex", or "sem" (default: "auto")
    ///     snippet_chars: Max chars per snippet (default: 500)
    ///     scope: Optional URI scope filter
    ///     since: Optional start timestamp filter
    ///     until: Optional end timestamp filter
    ///     include_snippets: Whether to include text snippets (default: True)
    ///     out: Optional output path to write the report
    ///     format: Output format - "text", "markdown", or "json" (default: "json")
    ///
    /// Returns:
    ///     A dict with keys: version, generated_at, question, answer, mode,
    ///     retriever, sources, total_hits, stats, notes
    #[pyo3(signature = (question, *, k = 10, mode = "auto", snippet_chars = 500, scope = None, since = None, until = None, include_snippets = true, out = None, format = "json"))]
    fn audit(
        &mut self,
        py: Python<'_>,
        question: &str,
        k: usize,
        mode: &str,
        snippet_chars: usize,
        scope: Option<&str>,
        since: Option<i64>,
        until: Option<i64>,
        include_snippets: bool,
        out: Option<&str>,
        format: &str,
    ) -> PyResult<PyObject> {
        let stats = self.ensure_open_ref()?.stats().map_err(Self::core_err)?;
        if stats.has_vec_index {
            self.vec_available = true;
        }
        let mut ask_mode = parse_ask_mode(mode)?;
        let want_vec = matches!(ask_mode, AskMode::Hybrid | AskMode::Sem);
        let can_use_vec = stats.has_vec_index || (self.vec_available && !self.read_only);
        let enable_vec = want_vec && can_use_vec;
        self.ensure_indexes(true, enable_vec)?;
        if want_vec && !enable_vec {
            ask_mode = AskMode::Lex;
        }

        let inner = self.ensure_open_mut()?;

        let options = AuditOptions {
            top_k: Some(k),
            snippet_chars: Some(snippet_chars),
            mode: Some(ask_mode),
            scope: scope.map(|s| s.to_string()),
            start: since,
            end: until,
            include_snippets,
        };

        let report = inner
            .audit::<dyn VecEmbedder>(question, Some(options), None::<&dyn VecEmbedder>)
            .map_err(Self::core_err)?;

        // Handle output file if specified
        if let Some(out_path) = out {
            let content = match format {
                "text" => report.to_text(),
                "markdown" | "md" => report.to_markdown(),
                "json" => serde_json::to_string_pretty(&report).map_err(|e| {
                    PyValueError::new_err(format!("JSON serialization error: {}", e))
                })?,
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "Unknown format: {}. Expected 'text', 'markdown', or 'json'",
                        format
                    )));
                }
            };
            fs::write(out_path, content)
                .map_err(|e| PyIOError::new_err(format!("Failed to write output file: {}", e)))?;
        }

        // Build Python dict result
        build_audit_result(py, report)
    }

    #[pyo3(signature = (*, limit = DEFAULT_TIMELINE_LIMIT, since = None, until = None, reverse = false, as_of_frame = None, as_of_ts = None))]
    fn timeline(
        &mut self,
        py: Python<'_>,
        limit: u64,
        since: Option<i64>,
        until: Option<i64>,
        reverse: bool,
        as_of_frame: Option<u64>,
        as_of_ts: Option<i64>,
    ) -> PyResult<PyObject> {
        let inner = self.ensure_open_mut()?;
        let query = make_timeline_query(limit, since, until, reverse);
        let mut entries = inner.timeline(query).map_err(Self::core_err)?;

        // Apply Replay filters
        if as_of_frame.is_some() || as_of_ts.is_some() {
            entries.retain(|entry| {
                if let Some(cutoff_frame) = as_of_frame {
                    if entry.frame_id > cutoff_frame {
                        return false;
                    }
                }
                if let Some(cutoff_ts) = as_of_ts {
                    if entry.timestamp > cutoff_ts {
                        return false;
                    }
                }
                true
            });
        }

        build_timeline(py, entries)
    }

    #[pyo3(signature = (uri))]
    fn frame(&mut self, py: Python<'_>, uri: &str) -> PyResult<PyObject> {
        let inner = self.ensure_open_mut()?;
        let frame = inner.frame_by_uri(uri).map_err(Self::core_err)?;
        let manifest = inner.media_manifest_by_uri(uri).map_err(Self::core_err)?;
        frame_to_py(py, &frame, manifest)
    }

    #[pyo3(signature = (uri))]
    fn blob(&mut self, py: Python<'_>, uri: &str) -> PyResult<PyObject> {
        let inner = self.ensure_open_mut()?;
        let mut reader = inner.blob_reader_by_uri(uri).map_err(Self::core_err)?;
        let mut buffer = Vec::new();
        reader
            .read_to_end(&mut buffer)
            .map_err(|err| PyIOError::new_err(err.to_string()))?;
        Ok(PyBytes::new(py, &buffer).to_object(py))
    }

    fn enable_lex(&mut self) -> PyResult<()> {
        self.ensure_mutable()?;
        let inner = self.ensure_open_mut()?;
        inner.enable_lex().map_err(Self::core_err)
    }

    fn enable_vec(&mut self) -> PyResult<()> {
        self.ensure_mutable()?;
        let inner = self.ensure_open_mut()?;
        inner.enable_vec().map_err(Self::core_err)?;
        self.vec_available = true;
        Ok(())
    }

    /// Set vector compression mode for embeddings
    /// Must be called before ingesting documents with embeddings
    fn set_vector_compression(&mut self, enabled: bool) -> PyResult<()> {
        self.ensure_mutable()?;
        let inner = self.ensure_open_mut()?;
        let compression = if enabled {
            memvid_core::VectorCompression::Pq96
        } else {
            memvid_core::VectorCompression::None
        };
        inner.set_vector_compression(compression);
        Ok(())
    }

    fn apply_ticket(&mut self, ticket: &str) -> PyResult<()> {
        self.ensure_mutable()?;
        let parsed: Ticket = match serde_json::from_str(ticket) {
            Ok(ticket) => ticket,
            Err(err) => {
                return Err(Python::with_gil(|py| {
                    build_exception(
                        py,
                        py.get_type::<TicketInvalidError>(),
                        "MV002",
                        format!("invalid ticket: {err}"),
                    )
                }));
            }
        };
        let inner = self.ensure_open_mut()?;
        #[allow(deprecated)]
        inner.apply_ticket(parsed).map_err(Self::core_err)
    }

    /// Apply a cryptographically signed ticket to this memory.
    /// This verifies the signature against the embedded Memvid public key.
    fn apply_signed_ticket(&mut self, ticket: &str) -> PyResult<()> {
        self.ensure_mutable()?;
        let parsed: SignedTicket = match serde_json::from_str(ticket) {
            Ok(ticket) => ticket,
            Err(err) => {
                return Err(Python::with_gil(|py| {
                    build_exception(
                        py,
                        py.get_type::<TicketInvalidError>(),
                        "MV002",
                        format!("invalid signed ticket: {err}"),
                    )
                }));
            }
        };
        let inner = self.ensure_open_mut()?;
        inner.apply_signed_ticket(parsed).map_err(Self::core_err)
    }

    /// Get the current memory binding, if any
    fn get_memory_binding(&self, py: Python<'_>) -> PyResult<PyObject> {
        let inner = self.ensure_open_ref()?;
        match inner.get_memory_binding() {
            Some(binding) => {
                let dict = PyDict::new(py);
                dict.set_item("memory_id", binding.memory_id.to_string())?;
                dict.set_item("memory_name", &binding.memory_name)?;
                dict.set_item("bound_at", binding.bound_at.to_rfc3339())?;
                dict.set_item("api_url", &binding.api_url)?;
                Ok(dict.to_object(py))
            }
            None => Ok(py.None()),
        }
    }

    /// Unbind the memory from the dashboard
    fn unbind_memory(&mut self) -> PyResult<()> {
        self.ensure_mutable()?;
        let inner = self.ensure_open_mut()?;
        inner.unbind_memory().map_err(Self::core_err)?;
        inner.commit().map_err(Self::core_err)
    }

    /// Get the current capacity in bytes
    fn get_capacity(&self) -> PyResult<u64> {
        let inner = self.ensure_open_ref()?;
        Ok(inner.get_capacity())
    }

    /// Get the current ticket information
    fn current_ticket(&self, py: Python<'_>) -> PyResult<PyObject> {
        let inner = self.ensure_open_ref()?;
        let ticket = inner.current_ticket();
        let dict = PyDict::new(py);
        dict.set_item("issuer", &ticket.issuer)?;
        dict.set_item("seq_no", ticket.seq_no)?;
        dict.set_item("expires_in_secs", ticket.expires_in_secs)?;
        dict.set_item("capacity_bytes", ticket.capacity_bytes)?;
        dict.set_item("verified", ticket.verified)?;
        Ok(dict.to_object(py))
    }

    /// Sync tickets from the API and apply to this file
    #[pyo3(signature = (memory_id, api_key, api_url = None))]
    fn sync_tickets(
        &mut self,
        py: Python<'_>,
        memory_id: &str,
        api_key: &str,
        api_url: Option<&str>,
    ) -> PyResult<PyObject> {
        self.ensure_mutable()?;

        // Parse memory_id - support both UUID and MongoDB ObjectId (24-char hex)
        let memory_uuid = if memory_id.contains('-') {
            // Already in UUID format
            Uuid::parse_str(memory_id)
                .map_err(|e| PyValueError::new_err(format!("invalid memory_id UUID: {e}")))?
        } else if memory_id.len() == 24 && memory_id.chars().all(|c| c.is_ascii_hexdigit()) {
            // MongoDB ObjectId: pad with 8 zeros and format as UUID
            let padded = format!("{}00000000", memory_id);
            let uuid_str = format!(
                "{}-{}-{}-{}-{}",
                &padded[0..8],
                &padded[8..12],
                &padded[12..16],
                &padded[16..20],
                &padded[20..32]
            );
            Uuid::parse_str(&uuid_str).map_err(|_| {
                PyValueError::new_err(format!("failed to convert ObjectId to UUID: {}", memory_id))
            })?
        } else {
            return Err(PyValueError::new_err(format!(
                "invalid memory_id format: expected UUID or 24-char hex ObjectId, got: {}",
                memory_id
            )));
        };

        let base_url = api_url.unwrap_or(DEFAULT_API_URL);
        let url = format!("{}/api/memories/{}/tickets/sync", base_url, memory_uuid);

        // Make HTTP request to sync tickets
        let response = ureq::post(&url)
            .set("X-API-Key", api_key)
            .set("Content-Type", "application/json")
            .call()
            .map_err(|e| {
                match e {
                    ureq::Error::Status(401, _) => {
                        PyRuntimeError::new_err(
                            "Invalid API key. Get a valid key at https://memvid.com/dashboard/api-keys"
                        )
                    }
                    ureq::Error::Status(403, _) => {
                        PyRuntimeError::new_err(format!(
                            "Access denied to memory '{}'. Make sure the memory ID is correct and belongs to your organisation.",
                            memory_id
                        ))
                    }
                    ureq::Error::Status(404, _) => {
                        PyRuntimeError::new_err(format!(
                            "Memory '{}' not found. Check that the memory ID exists in your dashboard at https://memvid.com/dashboard",
                            memory_id
                        ))
                    }
                    _ => PyRuntimeError::new_err(format!("API request failed: {}", e))
                }
            })?;

        let body: serde_json::Value = response
            .into_json()
            .map_err(|e| PyRuntimeError::new_err(format!("failed to parse API response: {e}")))?;

        // The API wraps responses in a signed envelope with data.ticket
        let ticket_data = body
            .get("data")
            .and_then(|d| d.get("ticket"))
            .ok_or_else(|| {
                PyRuntimeError::new_err(format!(
                    "API response missing 'data.ticket' field. Response: {}",
                    serde_json::to_string(&body).unwrap_or_default()
                ))
            })?;

        // Accept both "sequence" and "seq_no" field names (dashboard uses seq_no)
        let seq_no = ticket_data
            .get("sequence")
            .or_else(|| ticket_data.get("seq_no"))
            .and_then(|v| v.as_i64())
            .ok_or_else(|| PyRuntimeError::new_err("ticket missing 'sequence' or 'seq_no'"))?;

        let issuer = ticket_data
            .get("issuer")
            .and_then(|v| v.as_str())
            .ok_or_else(|| PyRuntimeError::new_err("ticket missing 'issuer'"))?
            .to_string();

        let expires_in = ticket_data
            .get("expires_in")
            .and_then(|v| v.as_u64())
            .unwrap_or(3600);

        let capacity_bytes = ticket_data.get("capacity_bytes").and_then(|v| v.as_u64());

        // Extract signature from API response (for signed ticket verification)
        let signature_b64 = ticket_data.get("signature").and_then(|v| v.as_str());

        // Check if already bound with same or higher sequence
        let inner = self.ensure_open_mut()?;
        let current = inner.current_ticket();
        if current.seq_no >= seq_no {
            // Still register file with control plane
            let _ = self.register_file_with_api(base_url, memory_id, api_key);

            // Already bound, return current info
            let dict = PyDict::new(py);
            dict.set_item("memory_id", memory_id)?;
            dict.set_item("issuer", &current.issuer)?;
            dict.set_item("seq_no", current.seq_no)?;
            dict.set_item("capacity_bytes", current.capacity_bytes)?;
            dict.set_item("verified", current.verified)?;
            dict.set_item("already_bound", true)?;
            return Ok(dict.to_object(py));
        }

        // Create memory binding if not already bound
        let binding = MemoryBinding {
            memory_id: memory_uuid,
            memory_name: issuer.clone(),
            bound_at: Utc::now(),
            api_url: base_url.to_string(),
        };

        // Apply ticket with signature verification if signature is present
        let verified = if let Some(sig_b64) = signature_b64 {
            // Decode base64 signature
            use base64::Engine;
            let signature = base64::engine::general_purpose::STANDARD
                .decode(sig_b64)
                .map_err(|e| PyRuntimeError::new_err(format!("invalid signature base64: {e}")))?;

            // Create signed ticket
            let signed_ticket = SignedTicket::new(
                &issuer,
                seq_no,
                expires_in,
                capacity_bytes,
                memory_uuid,
                signature,
            );

            // First ensure binding exists (without applying a ticket)
            // We use set_memory_binding_only to avoid the sequence number conflict
            // that occurs when bind_memory applies a temp ticket before apply_signed_ticket
            if inner.get_memory_binding().is_none() {
                inner
                    .set_memory_binding_only(binding.clone())
                    .map_err(Self::core_err)?;
            }

            // Now apply the signed ticket with verification
            // This handles both setting the seq_no and verifying the signature
            inner
                .apply_signed_ticket(signed_ticket)
                .map_err(Self::core_err)?;
            true
        } else {
            // Fallback to unsigned ticket (legacy dashboard compatibility)
            let mut ticket = Ticket::new(&issuer, seq_no).expires_in_secs(expires_in);
            if let Some(cap) = capacity_bytes {
                ticket = ticket.capacity_bytes(cap);
            }

            // Bind memory with unsigned ticket
            #[allow(deprecated)]
            inner.bind_memory(binding, ticket).map_err(Self::core_err)?;
            false
        };

        inner.commit().map_err(Self::core_err)?;

        // Register file with control plane
        let _ = self.register_file_with_api(base_url, memory_id, api_key);

        // Return result
        let dict = PyDict::new(py);
        dict.set_item("memory_id", memory_id)?;
        dict.set_item("issuer", &issuer)?;
        dict.set_item("seq_no", seq_no)?;
        dict.set_item("capacity_bytes", capacity_bytes.unwrap_or(0))?;
        dict.set_item("verified", verified)?;
        dict.set_item("already_bound", false)?;
        Ok(dict.to_object(py))
    }

    /// Helper to register file with the control plane API
    fn register_file_with_api(&self, base_url: &str, memory_id: &str, api_key: &str) {
        let file_name = self
            .path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown.mv2");

        let file_path = self
            .path
            .canonicalize()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|_| self.path.to_string_lossy().to_string());

        let file_size = std::fs::metadata(&self.path)
            .map(|m| m.len() as i64)
            .unwrap_or(0);

        let machine_id = hostname::get()
            .map(|h| h.to_string_lossy().to_string())
            .unwrap_or_else(|_| "unknown".to_string());

        let url = format!("{}/api/memories/{}/files", base_url, memory_id);

        let body = serde_json::json!({
            "file_name": file_name,
            "file_path": file_path,
            "file_size": file_size,
            "machine_id": machine_id,
        });

        // Silently ignore errors - file registration is best-effort
        let _ = ureq::post(&url)
            .set("X-API-Key", api_key)
            .set("Content-Type", "application/json")
            .send_json(body);
    }

    fn seal(&mut self) -> PyResult<()> {
        self.ensure_mutable()?;
        let inner = self.ensure_open_mut()?;
        inner.commit().map_err(Self::core_err)
    }

    /// Explicitly commit pending WAL/index changes to disk.
    /// Alias for seal() — flushes Tantivy index, WAL, and TOC.
    fn commit(&mut self) -> PyResult<()> {
        self.ensure_mutable()?;
        let inner = self.ensure_open_mut()?;
        inner.commit().map_err(Self::core_err)
    }

    #[cfg(feature = "parallel_segments")]
    fn commit_parallel(&mut self, opts: &PyBuildOpts) -> PyResult<()> {
        self.ensure_mutable()?;
        let inner = self.ensure_open_mut()?;
        inner
            .commit_parallel(opts.to_core())
            .map_err(Self::core_err)
    }

    /// Extract tables from a PDF file and store them in the memory.
    ///
    /// # Arguments
    /// * `pdf_path` - Path to the PDF file
    /// * `embed_rows` - Whether to generate embeddings for row content (default: True)
    ///
    /// # Returns
    /// Dict with 'tables_count' and list of table info dicts
    #[pyo3(signature = (pdf_path, *, embed_rows = true))]
    fn put_pdf_tables(
        &mut self,
        py: Python<'_>,
        pdf_path: &str,
        embed_rows: bool,
    ) -> PyResult<PyObject> {
        self.ensure_mutable()?;
        self.check_can_write()?;

        if embed_rows {
            // Table row embeddings imply semantic search should be available afterwards.
            self.vec_available = true;
            self.ensure_indexes(false, true)?;
        }

        // Read PDF bytes
        let pdf_bytes = fs::read(pdf_path).map_err(|e| PyIOError::new_err(e.to_string()))?;
        let filename = Path::new(pdf_path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown.pdf");

        // Extract tables
        let options = TableExtractionOptions::default();
        let result = extract_tables(&pdf_bytes, filename, &options)
            .map_err(|e| MemvidError::new_err(format!("Table extraction failed: {}", e)))?;

        if result.tables.is_empty() {
            let dict = PyDict::new(py);
            dict.set_item("tables_count", 0)?;
            dict.set_item("tables", PyList::empty(py))?;
            return Ok(dict.into());
        }

        // Store each table
        let inner = self.ensure_open_mut()?;
        let mut stored_tables = Vec::new();

        let (runtime, embedding_identity) = if embed_rows {
            let model_choice = match inner.embedding_identity_summary(10_000) {
                EmbeddingIdentitySummary::Unknown => inner
                    .effective_vec_index_dimension()
                    .map_err(Self::core_err)?
                    .and_then(EmbeddingModelChoice::from_dimension)
                    .unwrap_or(EmbeddingModelChoice::BgeSmall),
                EmbeddingIdentitySummary::Single(identity) => identity
                    .model
                    .as_deref()
                    .and_then(|model| EmbeddingModelChoice::parse(model).ok())
                    .or_else(|| {
                        identity
                            .dimension
                            .or(inner.effective_vec_index_dimension().ok().flatten())
                            .and_then(EmbeddingModelChoice::from_dimension)
                    })
                    .unwrap_or(EmbeddingModelChoice::BgeSmall),
                EmbeddingIdentitySummary::Mixed(_) => {
                    return Err(Self::core_err(MemvidCoreError::EmbeddingFailed {
                        reason:
                            "memory contains mixed embedding models; table row embedding is unsafe"
                                .to_string()
                                .into_boxed_str(),
                    }));
                }
            };

            let runtime = cached_embedding_runtime(model_choice).map_err(Self::core_err)?;
            let provider = if model_choice.is_openai() {
                "openai"
            } else {
                "fastembed"
            };
            let identity = EmbeddingIdentity {
                provider: Some(provider.to_string().into_boxed_str()),
                model: Some(
                    model_choice
                        .canonical_model_id()
                        .to_string()
                        .into_boxed_str(),
                ),
                dimension: Some(runtime.dimension() as u32),
                normalized: None,
            };
            (Some(runtime), Some(identity))
        } else {
            (None, None)
        };

        for table in &result.tables {
            let embedder = runtime.as_ref().map(|value| value as &dyn VecEmbedder);
            match store_table_with_embedder(
                inner,
                table,
                embed_rows,
                embedder,
                embedding_identity.as_ref(),
            ) {
                Ok((meta_id, row_ids)) => {
                    let table_dict = PyDict::new(py);
                    table_dict.set_item("table_id", &table.table_id)?;
                    table_dict.set_item("source_file", &table.source_file)?;
                    table_dict.set_item("n_rows", table.n_rows)?;
                    table_dict.set_item("n_cols", table.n_cols)?;
                    table_dict.set_item("page_start", table.page_start)?;
                    table_dict.set_item("page_end", table.page_end)?;
                    table_dict.set_item("quality", table.quality.to_string())?;
                    table_dict.set_item("detection_mode", table.detection_mode.to_string())?;
                    table_dict.set_item("meta_frame_id", meta_id)?;
                    table_dict.set_item("row_frame_count", row_ids.len())?;
                    let headers_list = PyList::new(py, &table.headers);
                    table_dict.set_item("headers", headers_list)?;
                    stored_tables.push(table_dict);
                }
                Err(e) => {
                    eprintln!("Warning: Failed to store table {}: {}", table.table_id, e);
                }
            }
        }

        inner.commit().map_err(Self::core_err)?;

        let result_dict = PyDict::new(py);
        result_dict.set_item("tables_count", stored_tables.len())?;
        result_dict.set_item("tables", PyList::new(py, stored_tables))?;
        Ok(result_dict.into())
    }

    /// List all tables stored in the memory.
    ///
    /// # Returns
    /// List of table summary dicts
    fn list_tables(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        let inner = self.ensure_open_mut()?;
        let summaries = list_tables(inner)
            .map_err(|e| MemvidError::new_err(format!("Failed to list tables: {}", e)))?;

        let result = PyList::empty(py);
        for summary in summaries {
            let dict = PyDict::new(py);
            dict.set_item("table_id", &summary.table_id)?;
            dict.set_item("source_file", &summary.source_file)?;
            dict.set_item("n_rows", summary.n_rows)?;
            dict.set_item("n_cols", summary.n_cols)?;
            dict.set_item("page_start", summary.page_start)?;
            dict.set_item("page_end", summary.page_end)?;
            dict.set_item("quality", summary.quality.to_string())?;
            dict.set_item("frame_id", summary.frame_id)?;
            let headers_list = PyList::new(py, &summary.headers);
            dict.set_item("headers", headers_list)?;
            result.append(dict)?;
        }
        Ok(result.into())
    }

    /// Get a table by its ID.
    ///
    /// # Arguments
    /// * `table_id` - The table ID
    /// * `format` - Output format: "dict" (default), "csv", or "json"
    ///
    /// # Returns
    /// Table data in the specified format
    #[pyo3(signature = (table_id, *, format = "dict"))]
    fn get_table(&mut self, py: Python<'_>, table_id: &str, format: &str) -> PyResult<PyObject> {
        let inner = self.ensure_open_mut()?;
        let table = get_table(inner, table_id)
            .map_err(|e| MemvidError::new_err(format!("Failed to get table: {}", e)))?;

        let table = match table {
            Some(t) => t,
            None => {
                return Err(PyValueError::new_err(format!(
                    "Table not found: {}",
                    table_id
                )));
            }
        };

        match format {
            "csv" => {
                let csv = export_to_csv(&table);
                Ok(csv.into_py(py))
            }
            "json" => {
                let json = export_to_json(&table, false)
                    .map_err(|e| MemvidError::new_err(format!("JSON export failed: {}", e)))?;
                Ok(json.into_py(py))
            }
            "dict" | _ => {
                let dict = PyDict::new(py);
                dict.set_item("table_id", &table.table_id)?;
                dict.set_item("source_file", &table.source_file)?;
                dict.set_item("n_rows", table.n_rows)?;
                dict.set_item("n_cols", table.n_cols)?;
                dict.set_item("page_start", table.page_start)?;
                dict.set_item("page_end", table.page_end)?;
                dict.set_item("quality", table.quality.to_string())?;
                dict.set_item("detection_mode", table.detection_mode.to_string())?;
                dict.set_item("confidence_score", table.confidence_score)?;

                let headers_list = PyList::new(py, &table.headers);
                dict.set_item("headers", headers_list)?;

                // Build rows as list of dicts (header -> value)
                let rows_list = PyList::empty(py);
                for row in table.data_rows() {
                    let row_dict = PyDict::new(py);
                    for (i, header) in table.headers.iter().enumerate() {
                        let value = row.cells.get(i).map(|c| c.text.as_str()).unwrap_or("");
                        row_dict.set_item(header, value)?;
                    }
                    rows_list.append(row_dict)?;
                }
                dict.set_item("rows", rows_list)?;

                Ok(dict.into())
            }
        }
    }

    fn close(&mut self) -> PyResult<()> {
        if let Some(mut inner) = self.inner.take() {
            if !self.read_only {
                let _ = inner.commit();
            }
        }
        Ok(())
    }

    // ==================== Session/Replay Methods ====================

    /// Start a new recording session
    #[cfg(feature = "replay")]
    #[pyo3(signature = (name = None))]
    fn session_start(&mut self, name: Option<String>) -> PyResult<String> {
        let inner = self.ensure_open_mut()?;
        let session_id = inner.start_session(name, None).map_err(Self::core_err)?;
        Ok(session_id.to_string())
    }

    /// End the current recording session
    #[cfg(feature = "replay")]
    fn session_end(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        let inner = self.ensure_open_mut()?;
        let session = inner.end_session().map_err(Self::core_err)?;

        // Save the session to file
        inner.save_replay_sessions().map_err(Self::core_err)?;

        // Convert session to Python dict
        let dict = PyDict::new(py);
        dict.set_item("session_id", session.session_id.to_string())?;
        dict.set_item("name", session.name.as_deref().unwrap_or(""))?;
        dict.set_item("action_count", session.actions.len())?;
        dict.set_item("checkpoint_count", session.checkpoints.len())?;
        dict.set_item("created_secs", session.created_secs)?;
        if let Some(ended) = session.ended_secs {
            dict.set_item("ended_secs", ended)?;
        }
        Ok(dict.into())
    }

    /// List all recorded sessions
    #[cfg(feature = "replay")]
    fn session_list(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        let inner = self.ensure_open_mut()?;

        // Load sessions from file first
        inner.load_replay_sessions().map_err(Self::core_err)?;

        let sessions = inner.list_sessions();
        let list = PyList::empty(py);
        for summary in sessions {
            let dict = PyDict::new(py);
            dict.set_item("session_id", summary.session_id.to_string())?;
            dict.set_item("name", summary.name.as_deref().unwrap_or(""))?;
            dict.set_item("action_count", summary.action_count)?;
            dict.set_item("checkpoint_count", summary.checkpoint_count)?;
            dict.set_item("created_secs", summary.created_secs)?;
            if let Some(ended) = summary.ended_secs {
                dict.set_item("ended_secs", ended)?;
            }
            list.append(dict)?;
        }
        Ok(list.into())
    }

    /// Replay a session with different parameters
    /// Supports audit mode for frozen retrieval and model comparison
    #[cfg(feature = "replay")]
    #[pyo3(signature = (session_id, *, top_k = None, adaptive = None, audit = None, use_model = None, diff = None))]
    fn session_replay(
        &mut self,
        py: Python<'_>,
        session_id: String,
        top_k: Option<usize>,
        adaptive: Option<bool>,
        audit: Option<bool>,
        use_model: Option<String>,
        diff: Option<bool>,
    ) -> PyResult<PyObject> {
        let inner = self.ensure_open_mut()?;

        // Load sessions from file first
        inner.load_replay_sessions().map_err(Self::core_err)?;

        // Parse session ID
        let uuid = Uuid::parse_str(&session_id)
            .map_err(|e| PyValueError::new_err(format!("Invalid session ID: {}", e)))?;

        // Get the session
        let session = inner
            .get_session(uuid)
            .ok_or_else(|| PyValueError::new_err(format!("Session {} not found", session_id)))?
            .clone();

        // Build replay config with audit mode options
        let config = ReplayExecutionConfig {
            top_k,
            adaptive: adaptive.unwrap_or(false),
            audit_mode: audit.unwrap_or(false),
            use_model,
            generate_diff: diff.unwrap_or(false),
            ..Default::default()
        };

        // Create replay engine and run
        let mut engine = ReplayEngine::new(inner, config);
        let result = engine.replay_session(&session).map_err(Self::core_err)?;

        // Convert result to Python dict
        let dict = PyDict::new(py);
        dict.set_item("total_actions", result.total_actions)?;
        dict.set_item("matched_actions", result.matched_actions)?;
        dict.set_item("mismatched_actions", result.mismatched_actions)?;
        dict.set_item("skipped_actions", result.skipped_actions)?;
        dict.set_item("match_rate", result.match_rate())?;
        dict.set_item("total_duration_ms", result.total_duration_ms)?;
        dict.set_item("success", result.is_success())?;
        dict.set_item("audit_mode", audit.unwrap_or(false))?;

        // Add action results with enhanced diff info
        let action_results = PyList::empty(py);
        for ar in &result.action_results {
            let ar_dict = PyDict::new(py);
            ar_dict.set_item("sequence", ar.sequence)?;
            ar_dict.set_item("action_type", format!("{:?}", ar.action_type))?;
            ar_dict.set_item("matched", ar.matched)?;
            ar_dict.set_item("duration_ms", ar.duration_ms)?;
            if let Some(ref diff_str) = ar.diff {
                ar_dict.set_item("diff", diff_str.clone())?;
            }
            action_results.append(ar_dict)?;
        }
        dict.set_item("action_results", action_results)?;

        Ok(dict.into())
    }

    /// Delete a recorded session
    #[cfg(feature = "replay")]
    fn session_delete(&mut self, session_id: String) -> PyResult<()> {
        let inner = self.ensure_open_mut()?;

        // Load sessions from file first
        inner.load_replay_sessions().map_err(Self::core_err)?;

        // Parse session ID
        let uuid = Uuid::parse_str(&session_id)
            .map_err(|e| PyValueError::new_err(format!("Invalid session ID: {}", e)))?;

        // Delete the session
        inner.delete_session(uuid).map_err(Self::core_err)?;

        // Save updated sessions
        inner.save_replay_sessions().map_err(Self::core_err)?;

        Ok(())
    }

    /// Create a checkpoint in the current session
    #[cfg(feature = "replay")]
    fn session_checkpoint(&mut self) -> PyResult<String> {
        let inner = self.ensure_open_mut()?;
        let checkpoint_id = inner.create_checkpoint().map_err(Self::core_err)?;
        Ok(checkpoint_id.to_string())
    }

    // ==================== End Session/Replay Methods ====================

    // ==================== Memory Cards & Enrichment ====================

    /// Get memory cards (triplets) with optional filtering.
    ///
    /// Returns extracted facts, preferences, events, and relationships as memory cards.
    ///
    /// Args:
    ///     entity: Optional entity name to filter by (e.g., "alice")
    ///     slot: Optional slot/predicate to filter by (e.g., "employer")
    ///
    /// Returns:
    ///     Dict with "cards" list and "count"
    fn memories(
        &self,
        py: Python<'_>,
        entity: Option<String>,
        slot: Option<String>,
    ) -> PyResult<PyObject> {
        let inner = self.ensure_open_ref()?;

        let mut cards_list: Vec<PyObject> = Vec::new();

        // Get all entities or filter by specific entity
        let entities: Vec<String> = if let Some(ref e) = entity {
            vec![e.to_lowercase()]
        } else {
            inner.memory_entities()
        };

        for ent in entities {
            for card in inner.get_entity_memories(&ent) {
                // Filter by slot if specified
                if let Some(ref s) = slot {
                    if card.slot.to_lowercase() != s.to_lowercase() {
                        continue;
                    }
                }

                let card_dict = PyDict::new(py);
                card_dict.set_item("id", card.id)?;
                card_dict.set_item("kind", card.kind.as_str())?;
                card_dict.set_item("entity", &card.entity)?;
                card_dict.set_item("slot", &card.slot)?;
                card_dict.set_item("value", &card.value)?;
                let polarity_str = card.polarity.map(|p| p.as_str().to_string());
                card_dict.set_item("polarity", polarity_str.as_deref())?;
                card_dict.set_item("confidence", card.confidence)?;
                card_dict.set_item("source_frame_id", card.source_frame_id)?;
                card_dict.set_item("source_uri", &card.source_uri)?;
                card_dict.set_item("document_date", card.document_date)?;
                card_dict.set_item("event_date", card.event_date)?;
                card_dict.set_item(
                    "engine",
                    if card.engine.is_empty() {
                        None
                    } else {
                        Some(&card.engine)
                    },
                )?;
                card_dict.set_item(
                    "engine_version",
                    if card.engine_version.is_empty() {
                        None
                    } else {
                        Some(&card.engine_version)
                    },
                )?;

                cards_list.push(card_dict.into());
            }
        }

        let result = PyDict::new(py);
        let count = cards_list.len();
        result.set_item("cards", PyList::new(py, cards_list))?;
        result.set_item("count", count)?;
        Ok(result.into())
    }

    /// Get memory statistics.
    ///
    /// Returns counts of entities, cards, and other memory-related metrics.
    fn memories_stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        let inner = self.ensure_open_ref()?;
        let stats = inner.memories_stats();

        let result = PyDict::new(py);
        result.set_item("entity_count", stats.entity_count)?;
        result.set_item("card_count", stats.card_count)?;
        result.set_item("slot_count", stats.slot_count)?;
        result.set_item("enriched_frames", stats.enriched_frames)?;
        result.set_item("last_enrichment", stats.last_enrichment)?;

        // Convert cards_by_kind HashMap to Python dict
        let by_kind = PyDict::new(py);
        for (kind, count) in &stats.cards_by_kind {
            by_kind.set_item(kind, count)?;
        }
        result.set_item("cards_by_kind", by_kind)?;

        Ok(result.into())
    }

    /// Get all entity names in the memory.
    fn memory_entities(&self) -> PyResult<Vec<String>> {
        let inner = self.ensure_open_ref()?;
        Ok(inner.memory_entities())
    }

    /// Get the current state of an entity (O(1) lookup).
    ///
    /// Returns the latest value for each slot of the entity.
    ///
    /// Args:
    ///     entity: Entity name to query (e.g., "alice")
    ///     slot: Optional slot to query (e.g., "employer")
    ///
    /// Returns:
    ///     Dict with entity, found, and slots
    fn state(&self, py: Python<'_>, entity: String, slot: Option<String>) -> PyResult<PyObject> {
        let inner = self.ensure_open_ref()?;
        let entity_lower = entity.to_lowercase();
        let cards = inner.get_entity_memories(&entity_lower);

        let result = PyDict::new(py);
        result.set_item("entity", &entity_lower)?;

        if cards.is_empty() {
            result.set_item("found", false)?;
            result.set_item("slots", PyDict::new(py))?;
            return Ok(result.into());
        }

        // Build state map (latest value per slot)
        let slots = PyDict::new(py);
        let mut seen_slots: std::collections::HashSet<String> = std::collections::HashSet::new();

        for card in cards {
            // Filter by slot if specified
            if let Some(ref s) = slot {
                if card.slot.to_lowercase() != s.to_lowercase() {
                    continue;
                }
            }

            // Only keep the first (latest) value per slot
            if seen_slots.contains(&card.slot) {
                continue;
            }
            seen_slots.insert(card.slot.clone());

            let slot_dict = PyDict::new(py);
            slot_dict.set_item("value", &card.value)?;
            slot_dict.set_item("kind", card.kind.as_str())?;
            let polarity_str = card.polarity.map(|p| p.as_str().to_string());
            slot_dict.set_item("polarity", polarity_str.as_deref())?;
            slot_dict.set_item("source_frame_id", card.source_frame_id)?;
            slot_dict.set_item("document_date", card.document_date)?;
            slot_dict.set_item(
                "engine",
                if card.engine.is_empty() {
                    None
                } else {
                    Some(&card.engine)
                },
            )?;

            slots.set_item(&card.slot, slot_dict)?;
        }

        result.set_item("found", true)?;
        result.set_item("slots", slots)?;
        Ok(result.into())
    }

    /// Run enrichment to extract memory cards from frames.
    ///
    /// Args:
    ///     engine: Engine to use - "rules" (default, fast). For LLM enrichment, use CLI.
    ///     force: Re-enrich all frames, ignoring previous enrichment records
    ///
    /// Returns:
    ///     Dict with enrichment stats
    fn enrich(
        &mut self,
        py: Python<'_>,
        engine: Option<String>,
        force: Option<bool>,
    ) -> PyResult<PyObject> {
        let engine_name = engine.unwrap_or_else(|| "rules".to_string());
        let force_enrich = force.unwrap_or(false);

        let inner = self.ensure_open_mut()?;

        // Get initial stats
        let initial_stats = inner.memories_stats();

        // Clear existing memories if force mode
        if force_enrich {
            inner.clear_memories();
        }

        // Run the appropriate engine
        let (kind, version, frames, cards) = match engine_name.to_lowercase().as_str() {
            "rules" => {
                let engine = RulesEngine::new();
                let kind = engine.kind().to_string();
                let version = engine.version().to_string();
                let (frames, cards) = inner.run_enrichment(&engine).map_err(Self::core_err)?;
                (kind, version, frames, cards)
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "Engine '{}' is not yet supported in the SDK. Available: rules. \
                     For LLM enrichment, use the CLI: memvid enrich --engine {}",
                    other, other
                )));
            }
        };

        // Commit changes
        inner.commit().map_err(Self::core_err)?;

        // Get final stats
        let final_stats = inner.memories_stats();

        let result = PyDict::new(py);
        result.set_item("engine", kind)?;
        result.set_item("version", version)?;
        result.set_item("frames_processed", frames)?;
        result.set_item("cards_extracted", cards)?;
        result.set_item("total_cards", final_stats.card_count)?;
        result.set_item("total_entities", final_stats.entity_count)?;
        result.set_item(
            "new_cards",
            final_stats
                .card_count
                .saturating_sub(initial_stats.card_count),
        )?;

        Ok(result.into())
    }

    /// Export memory cards (facts/triplets) to various formats.
    ///
    /// Args:
    ///     format: Output format - "json" (default), "csv", "ntriples"
    ///     entity: Optional entity filter
    ///     with_provenance: Include source frame info (default: False)
    ///
    /// Returns:
    ///     String in the requested format
    fn export_facts(
        &self,
        format: Option<String>,
        entity: Option<String>,
        with_provenance: Option<bool>,
    ) -> PyResult<String> {
        let fmt = format.unwrap_or_else(|| "json".to_string());
        let include_provenance = with_provenance.unwrap_or(false);

        let inner = self.ensure_open_ref()?;

        let mut facts: Vec<serde_json::Value> = Vec::new();

        // Get all entities or filter by specific entity
        let entities: Vec<String> = if let Some(ref e) = entity {
            vec![e.to_lowercase()]
        } else {
            inner.memory_entities()
        };

        for ent in entities {
            for card in inner.get_entity_memories(&ent) {
                let mut fact = json!({
                    "subject": card.entity,
                    "predicate": card.slot,
                    "object": card.value,
                });

                if include_provenance {
                    if let serde_json::Value::Object(ref mut map) = fact {
                        map.insert("source_frame_id".to_string(), json!(card.source_frame_id));
                        if let Some(ts) = card.document_date {
                            map.insert("timestamp".to_string(), json!(ts));
                        }
                        if !card.engine.is_empty() {
                            let engine_str = if !card.engine_version.is_empty() {
                                format!("{}:{}", card.engine, card.engine_version)
                            } else {
                                card.engine.clone()
                            };
                            map.insert("engine".to_string(), json!(engine_str));
                        }
                    }
                }

                facts.push(fact);
            }
        }

        match fmt.to_lowercase().as_str() {
            "json" => Ok(serde_json::to_string_pretty(&facts).unwrap_or_else(|_| "[]".to_string())),
            "csv" => {
                let mut csv = String::from("subject,predicate,object");
                if include_provenance {
                    csv.push_str(",source_frame_id,timestamp,engine");
                }
                csv.push('\n');

                for fact in &facts {
                    let subject = fact.get("subject").and_then(|v| v.as_str()).unwrap_or("");
                    let predicate = fact.get("predicate").and_then(|v| v.as_str()).unwrap_or("");
                    let object = fact.get("object").and_then(|v| v.as_str()).unwrap_or("");

                    // Escape CSV values
                    let escape_csv = |s: &str| -> String {
                        if s.contains(',') || s.contains('"') || s.contains('\n') {
                            format!("\"{}\"", s.replace('"', "\"\""))
                        } else {
                            s.to_string()
                        }
                    };

                    csv.push_str(&format!(
                        "{},{},{}",
                        escape_csv(subject),
                        escape_csv(predicate),
                        escape_csv(object)
                    ));

                    if include_provenance {
                        let frame_id = fact
                            .get("source_frame_id")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0);
                        let timestamp = fact.get("timestamp").and_then(|v| v.as_i64()).unwrap_or(0);
                        let engine = fact.get("engine").and_then(|v| v.as_str()).unwrap_or("");
                        csv.push_str(&format!(
                            ",{},{},{}",
                            frame_id,
                            timestamp,
                            escape_csv(engine)
                        ));
                    }

                    csv.push('\n');
                }

                Ok(csv)
            }
            "ntriples" | "nt" => {
                let mut nt = String::new();
                let base_uri = "mv2://entity/";

                for fact in &facts {
                    let subject = fact.get("subject").and_then(|v| v.as_str()).unwrap_or("");
                    let predicate = fact.get("predicate").and_then(|v| v.as_str()).unwrap_or("");
                    let object = fact.get("object").and_then(|v| v.as_str()).unwrap_or("");

                    // Escape N-Triples string
                    let escape_nt = |s: &str| -> String {
                        s.replace('\\', "\\\\")
                            .replace('"', "\\\"")
                            .replace('\n', "\\n")
                            .replace('\r', "\\r")
                            .replace('\t', "\\t")
                    };

                    nt.push_str(&format!(
                        "<{}{}> <{}pred/{}> \"{}\" .\n",
                        base_uri,
                        subject,
                        base_uri,
                        predicate,
                        escape_nt(object)
                    ));
                }

                Ok(nt)
            }
            other => Err(PyValueError::new_err(format!(
                "Unknown export format '{}'. Use: json, csv, ntriples",
                other
            ))),
        }
    }

    /// Add memory cards (SPO triplets) directly.
    ///
    /// This allows manual addition of extracted facts, useful when using
    /// external LLM enrichment or custom extraction logic.
    ///
    /// For automated LLM enrichment, use the CLI: memvid enrich --engine claude
    ///
    /// Args:
    ///     cards: List of dicts with keys: entity, slot, value, and optional kind, polarity, source_frame_id, engine
    ///
    /// Returns:
    ///     Dict with 'added' count and 'ids' list
    ///
    /// Example:
    ///     >>> mem.add_memory_cards([
    ///     ...     {"entity": "Alice", "slot": "employer", "value": "Acme Corp", "kind": "Fact"},
    ///     ...     {"entity": "Alice", "slot": "role", "value": "Engineer", "kind": "Profile"},
    ///     ... ])
    fn add_memory_cards(&mut self, py: Python<'_>, cards: Vec<Py<PyDict>>) -> PyResult<PyObject> {
        use memvid_core::types::{MemoryCard, MemoryCardBuilder, MemoryKind, Polarity};

        let inner = self.ensure_open_mut()?;
        let mut built_cards: Vec<MemoryCard> = Vec::new();

        for (i, card_dict) in cards.iter().enumerate() {
            let card: &PyDict = card_dict.as_ref(py).downcast()?;

            // Required fields
            let entity: String = card
                .get_item("entity")?
                .ok_or_else(|| PyValueError::new_err(format!("Card {} missing 'entity'", i)))?
                .extract()?;
            let slot: String = card
                .get_item("slot")?
                .ok_or_else(|| PyValueError::new_err(format!("Card {} missing 'slot'", i)))?
                .extract()?;
            let value: String = card
                .get_item("value")?
                .ok_or_else(|| PyValueError::new_err(format!("Card {} missing 'value'", i)))?
                .extract()?;

            // Optional fields
            let kind_str: String = card
                .get_item("kind")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or_else(|| "Fact".to_string());
            let kind = match kind_str.to_lowercase().as_str() {
                "fact" => MemoryKind::Fact,
                "preference" => MemoryKind::Preference,
                "event" => MemoryKind::Event,
                "profile" => MemoryKind::Profile,
                "relationship" => MemoryKind::Relationship,
                _ => MemoryKind::Other,
            };

            let polarity_str: Option<String> = card
                .get_item("polarity")?
                .map(|v| v.extract())
                .transpose()?;
            let polarity = match polarity_str.as_deref().map(|s| s.to_lowercase()).as_deref() {
                Some("positive") => Polarity::Positive,
                Some("negative") => Polarity::Negative,
                _ => Polarity::Neutral,
            };

            let source_frame_id: u64 = card
                .get_item("source_frame_id")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or(0);
            let engine: String = card
                .get_item("engine")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or_else(|| "sdk".to_string());
            let engine_version: String = card
                .get_item("engine_version")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or_else(|| "1.0.0".to_string());

            let memory_card = MemoryCardBuilder::new()
                .kind(kind)
                .entity(&entity)
                .slot(&slot)
                .value(&value)
                .polarity(polarity)
                .source(source_frame_id, None)
                .engine(&engine, &engine_version)
                .build(0)
                .map_err(|e| PyValueError::new_err(format!("Failed to build card {}: {}", i, e)))?;

            built_cards.push(memory_card);
        }

        let ids = inner
            .put_memory_cards(built_cards)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to add cards: {}", e)))?;
        inner
            .commit()
            .map_err(|e| PyRuntimeError::new_err(format!("Commit failed: {}", e)))?;

        let result = PyDict::new(py);
        result.set_item("added", ids.len())?;
        result.set_item("ids", ids)?;
        Ok(result.into())
    }

    // ==================== End Memory Cards & Enrichment ====================

    fn path(&self) -> PyResult<String> {
        Ok(self.path.to_string_lossy().to_string())
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __exit__(&mut self, _exc_type: &PyAny, _exc: &PyAny, _tb: &PyAny) -> PyResult<()> {
        self.close()
    }
}

struct PutPayload {
    text: Option<String>,
    file: Option<String>,
    uri: Option<String>,
    title: String,
    kind: Option<String>,
    track: Option<String>,
    tags: Vec<String>,
    labels: Vec<String>,
    doc_metadata: Option<DocMetadata>,
    extra_metadata: BTreeMap<String, Value>,
    search_text: Option<String>,
    enable_embedding: bool,
    /// Whether enable_embedding was auto-enabled via OPENAI_API_KEY
    auto_enabled_via_openai: bool,
    embedding_model: Option<String>,
    auto_tag: bool,
    extract_dates: bool,
    parallel: Option<bool>,
    /// Don't store raw binary content (default: true).
    /// Only extracted text + BLAKE3 hash is stored for space efficiency.
    no_raw: bool,
    /// Original source file path (for no_raw reference tracking).
    source_path: Option<String>,
    /// Unix timestamp for the frame (defaults to current time if not provided).
    /// Accepts epoch seconds (i64) or human-readable strings like "Jan 15, 2023".
    timestamp: Option<i64>,
    /// Run rules-based memory extraction after ingestion (default: true).
    enable_enrichment: bool,
}

/// Parse a timestamp from Python value (int or string).
/// Accepts:
/// - Epoch seconds as int: 1673740800
/// - Epoch seconds as string: "1673740800"
/// - ISO format: "2023-01-15"
/// - US format: "Jan 15, 2023" or "January 15, 2023"
/// - Slash format: "01/15/2023" or "1/15/2023"
fn parse_timestamp_value(value: &PyAny) -> PyResult<i64> {
    // Try as integer first
    if let Ok(ts) = value.extract::<i64>() {
        return Ok(ts);
    }

    // Try as float (JavaScript-style milliseconds or Python float)
    if let Ok(ts) = value.extract::<f64>() {
        return Ok(ts as i64);
    }

    // Parse as string with various formats
    let s: String = value.extract().map_err(|_| {
        PyValueError::new_err("timestamp must be an integer (epoch seconds) or date string")
    })?;
    let s = s.trim();

    // Try parsing as epoch timestamp first (pure digits, optionally negative)
    if s.chars().all(|c| c.is_ascii_digit())
        || (s.starts_with('-') && s[1..].chars().all(|c| c.is_ascii_digit()))
    {
        return s
            .parse::<i64>()
            .map_err(|e| PyValueError::new_err(format!("Invalid epoch timestamp: {e}")));
    }

    // Try various date formats
    use chrono::{NaiveDate, NaiveDateTime, NaiveTime, TimeZone};

    // Parse date and return midnight UTC
    fn date_to_epoch(date: NaiveDate) -> i64 {
        let datetime = NaiveDateTime::new(date, NaiveTime::from_hms_opt(0, 0, 0).unwrap());
        Utc.from_utc_datetime(&datetime).timestamp()
    }

    // Try ISO format: "2023-01-15"
    if let Ok(date) = NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        return Ok(date_to_epoch(date));
    }

    // Try US formats: "Jan 15, 2023" or "January 15, 2023"
    if let Ok(date) = NaiveDate::parse_from_str(s, "%b %d, %Y") {
        return Ok(date_to_epoch(date));
    }
    if let Ok(date) = NaiveDate::parse_from_str(s, "%B %d, %Y") {
        return Ok(date_to_epoch(date));
    }

    // Try slash format: "01/15/2023" or "1/15/2023"
    if let Ok(date) = NaiveDate::parse_from_str(s, "%m/%d/%Y") {
        return Ok(date_to_epoch(date));
    }

    // Try European format: "15-01-2023" or "15/01/2023"
    if let Ok(date) = NaiveDate::parse_from_str(s, "%d-%m-%Y") {
        return Ok(date_to_epoch(date));
    }
    if let Ok(date) = NaiveDate::parse_from_str(s, "%d/%m/%Y") {
        return Ok(date_to_epoch(date));
    }

    Err(PyValueError::new_err(format!(
        "Unable to parse timestamp '{}'. Supported formats: epoch seconds, \
         YYYY-MM-DD, Jan 15, 2023, MM/DD/YYYY",
        s
    )))
}

fn py_to_value(value: &PyAny) -> PyResult<Value> {
    if value.is_none() {
        return Ok(Value::Null);
    }
    if let Ok(boolean) = value.extract::<bool>() {
        return Ok(Value::Bool(boolean));
    }
    if let Ok(integer) = value.extract::<i64>() {
        return Ok(Value::Number(Number::from(integer)));
    }
    if let Ok(unsigned) = value.extract::<u64>() {
        return Ok(Value::Number(Number::from(unsigned)));
    }
    if let Ok(float) = value.extract::<f64>() {
        return Number::from_f64(float)
            .map(Value::Number)
            .ok_or_else(|| PyValueError::new_err("metadata float must be finite"));
    }
    if let Ok(string) = value.extract::<String>() {
        return Ok(Value::String(string));
    }
    if let Ok(dict) = value.downcast::<PyDict>() {
        let mut map = serde_json::Map::new();
        for (key, entry) in dict {
            let key_str = key.extract::<String>()?;
            map.insert(key_str, py_to_value(entry)?);
        }
        return Ok(Value::Object(map));
    }
    if let Ok(list) = value.downcast::<PyList>() {
        let mut elements = Vec::with_capacity(list.len());
        for item in list {
            elements.push(py_to_value(item)?);
        }
        return Ok(Value::Array(elements));
    }
    if let Ok(tuple) = value.downcast::<PyTuple>() {
        let mut elements = Vec::with_capacity(tuple.len());
        for item in tuple.iter() {
            elements.push(py_to_value(item)?);
        }
        return Ok(Value::Array(elements));
    }

    Err(PyValueError::new_err("unsupported metadata value type"))
}

fn is_doc_metadata_key(key: &str) -> bool {
    matches!(
        key,
        "mime"
            | "bytes"
            | "hash"
            | "width"
            | "height"
            | "colors"
            | "caption"
            | "exif"
            | "audio"
            | "media"
    )
}

fn split_metadata(
    value: Option<Value>,
) -> PyResult<(Option<DocMetadata>, BTreeMap<String, Value>)> {
    match value {
        None | Some(Value::Null) => Ok((None, BTreeMap::new())),
        Some(Value::Object(map)) => {
            let mut doc_map = serde_json::Map::new();
            let mut extras = BTreeMap::new();
            for (key, val) in map.into_iter() {
                if is_doc_metadata_key(&key) {
                    doc_map.insert(key, val);
                } else {
                    extras.insert(key, val);
                }
            }
            let doc_metadata = if doc_map.is_empty() {
                None
            } else {
                Some(
                    serde_json::from_value(Value::Object(doc_map)).map_err(|err| {
                        PyValueError::new_err(format!("invalid metadata payload: {err}"))
                    })?,
                )
            };
            Ok((doc_metadata, extras))
        }
        Some(other) => {
            let meta: DocMetadata = serde_json::from_value(other.clone()).map_err(|err| {
                PyValueError::new_err(format!(
                    "metadata must be an object compatible with DocMetadata: {err}"
                ))
            })?;
            Ok((Some(meta), BTreeMap::new()))
        }
    }
}

fn media_manifest_to_py<'py>(py: Python<'py>, manifest: &MediaManifest) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("kind", manifest.kind.clone())?;
    dict.set_item("mime", manifest.mime.clone())?;
    dict.set_item("bytes", manifest.bytes)?;
    dict.set_item("filename", manifest.filename.clone())?;
    dict.set_item("duration_ms", manifest.duration_ms)?;
    dict.set_item("width", manifest.width)?;
    dict.set_item("height", manifest.height)?;
    dict.set_item("codec", manifest.codec.clone())?;
    Ok(dict.to_object(py))
}

fn frame_to_py<'py>(
    py: Python<'py>,
    frame: &Frame,
    manifest: Option<MediaManifest>,
) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("id", frame.id)?;
    dict.set_item("uri", frame.uri.clone())?;
    dict.set_item("title", frame.title.clone())?;
    dict.set_item("timestamp", frame.timestamp)?;
    dict.set_item("kind", frame.kind.clone())?;
    dict.set_item("track", frame.track.clone())?;
    dict.set_item("tags", PyList::new(py, &frame.tags))?;
    dict.set_item("labels", PyList::new(py, &frame.labels))?;
    dict.set_item("payload_length", frame.payload_length)?;
    if let Some(meta) = frame.metadata.as_ref() {
        if let Some(mime) = &meta.mime {
            dict.set_item("mime", mime.clone())?;
        }
        if let Some(bytes) = meta.bytes {
            dict.set_item("bytes", bytes)?;
        }
        if let Some(hash) = &meta.hash {
            dict.set_item("hash", hash.clone())?;
        }
        if let Some(width) = meta.width {
            dict.set_item("width", width)?;
        }
        if let Some(height) = meta.height {
            dict.set_item("height", height)?;
        }
        if let Some(caption) = &meta.caption {
            dict.set_item("caption", caption.clone())?;
        }
        if let Some(colors) = &meta.colors {
            dict.set_item("colors", colors.clone())?;
        }
    }
    if !frame.extra_metadata.is_empty() {
        dict.set_item("extra_metadata", frame.extra_metadata.clone())?;
    }
    if let Some(media) = manifest {
        dict.set_item("media", media_manifest_to_py(py, &media)?)?;
    }
    Ok(dict.to_object(py))
}

fn parse_put_payload(payload: &PyDict) -> PyResult<PutPayload> {
    fn get_opt_string(dict: &PyDict, key: &str) -> PyResult<Option<String>> {
        dict.get_item(key)?
            .map(|value| value.extract::<String>())
            .transpose()
    }

    fn get_string(dict: &PyDict, key: &str) -> PyResult<String> {
        match dict.get_item(key)? {
            Some(value) => value.extract::<String>(),
            None => Err(PyValueError::new_err(format!(
                "put payload requires a '{key}' field"
            ))),
        }
    }

    fn get_opt_vec(dict: &PyDict, key: &str) -> PyResult<Vec<String>> {
        dict.get_item(key)?
            .map(|value| value.extract::<Vec<String>>())
            .transpose()
            .map(|opt| opt.unwrap_or_default())
    }

    let text = get_opt_string(payload, "text")?;
    let file = get_opt_string(payload, "file")?;
    let uri = get_opt_string(payload, "uri")?;
    let title = get_string(payload, "title")?;
    let kind = get_opt_string(payload, "kind")?;
    let track = get_opt_string(payload, "track")?;

    let mut labels = get_opt_vec(payload, "labels")?;
    if let Some(label_value) = get_opt_string(payload, "label")? {
        if !labels.contains(&label_value) {
            labels.insert(0, label_value);
        }
    }
    if labels.is_empty() {
        return Err(PyValueError::new_err(
            "put payload requires either 'label' or 'labels' entries",
        ));
    }

    let tags = get_opt_vec(payload, "tags")?;

    let metadata_value = payload
        .get_item("metadata")?
        .map(|value| {
            let dict = value.downcast::<PyDict>()?;
            let mut map = serde_json::Map::new();
            for (key, entry) in dict {
                let key_str = key.extract::<String>()?;
                map.insert(key_str, py_to_value(entry)?);
            }
            Ok::<Value, PyErr>(Value::Object(map))
        })
        .transpose()?;
    let (doc_metadata, extra_metadata) = split_metadata(metadata_value)?;

    let search_text = get_opt_string(payload, "search_text")?;

    let enable_embedding_explicit = payload
        .get_item("enable_embedding")?
        .map(|value| value.extract::<bool>())
        .transpose()?;

    // Check if OPENAI_API_KEY is available for auto-embedding
    let has_openai_key = env::var("OPENAI_API_KEY")
        .ok()
        .map(|k| !k.trim().is_empty())
        .unwrap_or(false);

    // Determine if embeddings should be enabled:
    // - If enable_embedding is explicitly set, use that value
    // - If enable_embedding is None and OPENAI_API_KEY exists, auto-enable with OpenAI
    let (enable_embedding, auto_enabled_via_openai) = match enable_embedding_explicit {
        Some(enabled) => (enabled, false),
        None => (has_openai_key, has_openai_key), // Auto-enable if OPENAI_API_KEY exists
    };

    let embedding_model = get_opt_string(payload, "embedding_model")?;
    let auto_tag = payload
        .get_item("auto_tag")?
        .map(|value| value.extract::<bool>())
        .transpose()?;
    let extract_dates = payload
        .get_item("extract_dates")?
        .map(|value| value.extract::<bool>())
        .transpose()?;
    let parallel = payload
        .get_item("parallel")?
        .map(|value| value.extract::<bool>())
        .transpose()?;
    // Default to no_raw=true (text-only storage) for space efficiency
    let no_raw = payload
        .get_item("no_raw")?
        .map(|value| value.extract::<bool>())
        .transpose()?
        .unwrap_or(true);
    let source_path = payload
        .get_item("source_path")?
        .map(|value| value.extract::<String>())
        .transpose()?;
    let timestamp = payload
        .get_item("timestamp")?
        .map(|value| parse_timestamp_value(value))
        .transpose()?;
    // Default to enable_enrichment=true (rules-based memory extraction)
    let enable_enrichment = payload
        .get_item("enable_enrichment")?
        .map(|value| value.extract::<bool>())
        .transpose()?
        .unwrap_or(true);

    Ok(PutPayload {
        text,
        file,
        uri,
        title,
        kind,
        track,
        tags,
        labels,
        doc_metadata,
        extra_metadata,
        search_text,
        enable_embedding,
        auto_enabled_via_openai,
        embedding_model,
        auto_tag: auto_tag.unwrap_or(true),
        extract_dates: extract_dates.unwrap_or(true),
        parallel,
        no_raw,
        source_path,
        timestamp,
        enable_enrichment,
    })
}

/// Parse a single document from Python dict to PutRequest
fn parse_put_request(doc: &PyDict) -> PyResult<PutRequest> {
    // Helper to extract required string
    fn get_string(dict: &PyDict, key: &str) -> PyResult<String> {
        dict.get_item(key)?
            .ok_or_else(|| PyValueError::new_err(format!("Document requires '{key}' field")))?
            .extract::<String>()
    }

    // Helper to extract optional string
    fn get_opt_string(dict: &PyDict, key: &str) -> PyResult<Option<String>> {
        dict.get_item(key)?
            .map(|value| value.extract::<String>())
            .transpose()
    }

    // Helper to extract optional vec
    fn get_opt_vec(dict: &PyDict, key: &str) -> PyResult<Vec<String>> {
        dict.get_item(key)?
            .map(|value| value.extract::<Vec<String>>())
            .transpose()
            .map(|opt| opt.unwrap_or_default())
    }

    // Helper to extract optional embedding vector
    fn get_opt_embedding(dict: &PyDict, key: &str) -> PyResult<Option<Vec<f32>>> {
        // Try direct access first (works around PyO3 dict iteration issues)
        match dict.get_item(key) {
            Ok(Some(value)) => match value.extract::<Vec<f32>>() {
                Ok(vec) => Ok(Some(vec)),
                Err(_) => Ok(None),
            },
            Ok(None) => Ok(None),
            Err(e) => Err(e),
        }
    }

    let title = get_string(doc, "title")?;
    let label = get_string(doc, "label")?;
    let text = get_string(doc, "text")?;
    let uri = get_opt_string(doc, "uri")?;
    let tags = get_opt_vec(doc, "tags")?;
    let labels = get_opt_vec(doc, "labels")?;
    let embedding = get_opt_embedding(doc, "embedding")?;
    // DEBUG: Log result

    // Parse metadata from Python dict to BTreeMap<String, serde_json::Value>
    let metadata = doc
        .get_item("metadata")?
        .map(|value| {
            let dict = value.downcast::<PyDict>()?;
            let mut map = BTreeMap::new();
            for (key, entry) in dict {
                let key_str = key.extract::<String>()?;
                map.insert(key_str, py_to_value(entry)?);
            }
            Ok::<BTreeMap<String, Value>, PyErr>(map)
        })
        .transpose()?
        .unwrap_or_default();

    Ok(PutRequest {
        title,
        label,
        text,
        uri,
        metadata,
        tags,
        labels,
        embedding,
    })
}

/// Parse PutManyOpts from Python dict
/// Returns (PutManyOpts, auto_enabled_via_openai)
fn parse_put_many_opts(opts: &PyDict) -> PyResult<(PutManyOpts, bool)> {
    let compression_level = opts
        .get_item("compression_level")?
        .map(|v| v.extract::<i32>())
        .transpose()?
        .unwrap_or(3); // Default compression level

    let disable_auto_checkpoint = opts
        .get_item("disable_auto_checkpoint")?
        .map(|v| v.extract::<bool>())
        .transpose()?
        .unwrap_or(true); // Default to batch mode

    let skip_sync = opts
        .get_item("skip_sync")?
        .map(|v| v.extract::<bool>())
        .transpose()?
        .unwrap_or(false); // Default to safe mode

    let enable_embedding_explicit = opts
        .get_item("enable_embedding")?
        .map(|v| v.extract::<bool>())
        .transpose()?;

    // Check if OPENAI_API_KEY is available for auto-embedding
    let has_openai_key = env::var("OPENAI_API_KEY")
        .ok()
        .map(|k| !k.trim().is_empty())
        .unwrap_or(false);

    // Determine if embeddings should be enabled:
    // - If enable_embedding is explicitly set, use that value
    // - If enable_embedding is None and OPENAI_API_KEY exists, auto-enable with OpenAI
    let (enable_embedding, auto_enabled_via_openai) = match enable_embedding_explicit {
        Some(enabled) => (enabled, false),
        None => (has_openai_key, has_openai_key),
    };

    let auto_tag = opts
        .get_item("auto_tag")?
        .map(|v| v.extract::<bool>())
        .transpose()?
        .unwrap_or(false);

    let extract_dates = opts
        .get_item("extract_dates")?
        .map(|v| v.extract::<bool>())
        .transpose()?
        .unwrap_or(false);

    // Default to no_raw=true (text-only storage) for space efficiency
    let no_raw = opts
        .get_item("no_raw")?
        .map(|v| v.extract::<bool>())
        .transpose()?
        .unwrap_or(true);

    // Default to enable_enrichment=true (rules-based memory extraction)
    let enable_enrichment = opts
        .get_item("enable_enrichment")?
        .map(|v| v.extract::<bool>())
        .transpose()?
        .unwrap_or(true);

    Ok((
        PutManyOpts {
            compression_level,
            disable_auto_checkpoint,
            skip_sync,
            enable_embedding,
            auto_tag,
            extract_dates,
            no_raw,
            enable_enrichment,
            wal_pre_size_bytes: 0,
        },
        auto_enabled_via_openai,
    ))
}

fn perform_put(mem: &mut MemvidCore, payload: &PutPayload) -> PyResult<u64> {
    let bytes = resolve_payload(payload.text.as_deref(), payload.file.as_deref())?;

    let mut options = memvid_core::types::PutOptions::default();
    options.title = Some(payload.title.clone());
    // Use file path as uri fallback for mime detection in core
    options.uri = payload.uri.clone().or_else(|| payload.file.clone());
    options.kind = payload.kind.clone();
    options.track = payload.track.clone();
    options.tags = payload.tags.clone();
    options.labels = payload.labels.clone();
    options.metadata = payload.doc_metadata.clone();
    // For files: let memvid-core handle extraction via ReaderRegistry (enables chunking)
    // For text: use provided search_text or fall back to text content
    options.search_text = if payload.file.is_some() {
        // Don't set search_text for files - let put_internal extract and chunk
        payload.search_text.clone()
    } else {
        // For text payloads, use search_text or the text itself
        payload.search_text.clone().or_else(|| payload.text.clone())
    };
    options.enable_embedding = payload.enable_embedding;
    options.auto_tag = payload.auto_tag;
    options.extract_dates = payload.extract_dates;
    options.no_raw = payload.no_raw;
    options.source_path = payload.source_path.clone();
    options.timestamp = payload.timestamp;

    for (key, value) in payload.extra_metadata.iter() {
        let is_identity_key = matches!(
            key.as_str(),
            MEMVID_EMBEDDING_PROVIDER_KEY
                | MEMVID_EMBEDDING_MODEL_KEY
                | MEMVID_EMBEDDING_DIMENSION_KEY
                | MEMVID_EMBEDDING_NORMALIZED_KEY
        );
        if is_identity_key {
            if let Value::String(s) = value {
                options.extra_metadata.insert(key.clone(), s.clone());
                continue;
            }
        }
        options
            .extra_metadata
            .insert(key.clone(), value.to_string());
    }

    if !payload.enable_embedding {
        return mem
            .put_bytes_with_options(&bytes, options)
            .map_err(MemvidCorePy::core_err);
    }

    // Choose model:
    // - If model specified, use it
    // - If auto-enabled via OPENAI_API_KEY (and no model specified), use OpenAISmall
    // - Otherwise fall back to BgeSmall (local)
    let model_choice = match payload
        .embedding_model
        .as_deref()
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
        .map(EmbeddingModelChoice::parse)
        .transpose()
        .map_err(MemvidCorePy::core_err)?
    {
        Some(model) => model,
        None if payload.auto_enabled_via_openai => EmbeddingModelChoice::OpenAISmall,
        None => EmbeddingModelChoice::BgeSmall,
    };

    let runtime = cached_embedding_runtime(model_choice).map_err(MemvidCorePy::core_err)?;
    apply_embedding_identity_metadata_from_choice(
        &mut options,
        runtime.model_choice(),
        runtime.dimension(),
    );

    // Use options.search_text which may contain extracted text from document
    if let Some(search_text) = options
        .search_text
        .as_deref()
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
    {
        let embedding = runtime.embed(search_text).map_err(MemvidCorePy::core_err)?;
        return mem
            .put_with_embedding_and_options(&bytes, embedding, options)
            .map_err(MemvidCorePy::core_err);
    }

    // Fall back to parsing bytes as text
    let text = payload_text_for_embedding(&bytes).ok_or_else(|| {
        MemvidCorePy::core_err(MemvidCoreError::EmbeddingFailed {
            reason: "enable_embedding=true requires text content (from text payload, search_text, or extractable document like PDF/DOCX/XLSX)"
                .to_string()
                .into_boxed_str(),
        })
    })?;

    if let Some(chunks) = mem.preview_chunks(&bytes) {
        let chunk_embeddings =
            embed_texts_batched(&runtime, chunks).map_err(MemvidCorePy::core_err)?;
        return mem
            .put_with_chunk_embeddings(&bytes, None, chunk_embeddings, options)
            .map_err(MemvidCorePy::core_err);
    }

    let embedding = runtime.embed(text).map_err(MemvidCorePy::core_err)?;
    mem.put_with_embedding_and_options(&bytes, embedding, options)
        .map_err(MemvidCorePy::core_err)
}

fn resolve_payload(text: Option<&str>, file: Option<&str>) -> PyResult<Vec<u8>> {
    match (text, file) {
        (Some(t), None) => Ok(t.as_bytes().to_vec()),
        (None, Some(path)) => fs::read(path).map_err(|err| PyIOError::new_err(err.to_string())),
        (Some(_), Some(_)) => Err(PyValueError::new_err(
            "provide either 'text' or 'file', not both",
        )),
        (None, None) => Err(PyValueError::new_err(
            "put requires either 'text' or 'file' payload",
        )),
    }
}

/// Static reader registry for document extraction
fn default_reader_registry() -> &'static ReaderRegistry {
    static REGISTRY: OnceLock<ReaderRegistry> = OnceLock::new();
    REGISTRY.get_or_init(ReaderRegistry::default)
}

/// Detect MIME type from file bytes
#[allow(dead_code)]
fn detect_mime_type(bytes: &[u8], file_path: Option<&str>) -> String {
    // Check magic bytes first
    if bytes.len() >= 4 {
        // PDF: %PDF
        if bytes.starts_with(b"%PDF") {
            return "application/pdf".to_string();
        }
        // ZIP-based formats (XLSX, DOCX, PPTX)
        if bytes.starts_with(&[0x50, 0x4B, 0x03, 0x04]) {
            if let Some(path) = file_path {
                let path_lower = path.to_lowercase();
                if path_lower.ends_with(".xlsx") {
                    return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        .to_string();
                }
                if path_lower.ends_with(".docx") {
                    return "application/vnd.openxmlformats-officedocument.wordprocessingml.document".to_string();
                }
                if path_lower.ends_with(".pptx") {
                    return "application/vnd.openxmlformats-officedocument.presentationml.presentation".to_string();
                }
            }
        }
    }
    // Fall back to extension-based detection
    if let Some(path) = file_path {
        let path_lower = path.to_lowercase();
        if path_lower.ends_with(".pdf") {
            return "application/pdf".to_string();
        }
        if path_lower.ends_with(".xlsx") {
            return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet".to_string();
        }
        if path_lower.ends_with(".docx") {
            return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                .to_string();
        }
        if path_lower.ends_with(".pptx") {
            return "application/vnd.openxmlformats-officedocument.presentationml.presentation"
                .to_string();
        }
        if path_lower.ends_with(".csv") {
            return "text/csv".to_string();
        }
        if path_lower.ends_with(".txt") {
            return "text/plain".to_string();
        }
    }
    // Default to octet-stream
    "application/octet-stream".to_string()
}

/// Infer document format from MIME type
#[allow(dead_code)]
fn infer_document_format(mime: &str) -> Option<DocumentFormat> {
    match mime {
        "application/pdf" => Some(DocumentFormat::Pdf),
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" => {
            Some(DocumentFormat::Xlsx)
        }
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document" => {
            Some(DocumentFormat::Docx)
        }
        "application/vnd.openxmlformats-officedocument.presentationml.presentation" => {
            Some(DocumentFormat::Pptx)
        }
        "text/csv" | "text/plain" => Some(DocumentFormat::PlainText),
        "text/markdown" => Some(DocumentFormat::Markdown),
        "text/html" => Some(DocumentFormat::Html),
        _ => None,
    }
}

/// Extract text from document bytes (PDF, XLSX, DOCX, etc.)
/// Returns the extracted text if successful, None otherwise
#[allow(dead_code)]
fn extract_text_from_document(bytes: &[u8], file_path: Option<&str>) -> Option<String> {
    let mime = detect_mime_type(bytes, file_path);

    // Only process known document types
    let format = infer_document_format(&mime)?;

    let registry = default_reader_registry();
    let magic = bytes.get(..64);
    let hint = ReaderHint::new(Some(&mime), Some(format))
        .with_uri(file_path)
        .with_magic(magic);

    // Try to find a reader for this document type
    let reader = registry.find_reader(&hint)?;

    // Extract text using the reader
    match reader.extract(bytes, &hint) {
        Ok(output) => {
            if let Some(text) = output.document.text {
                if !text.trim().is_empty() {
                    return Some(text);
                }
            }
            None
        }
        Err(_err) => {
            // Document extraction failed - silently fall back to raw bytes
            None
        }
    }
}

fn value_to_py(py: Python<'_>, value: Value) -> PyResult<PyObject> {
    match value {
        Value::Null => Ok(py.None()),
        Value::Bool(v) => Ok(v.into_py(py)),
        Value::Number(num) => {
            if let Some(int_val) = num.as_i64() {
                Ok(int_val.into_py(py))
            } else if let Some(u) = num.as_u64() {
                Ok(u.into_py(py))
            } else if let Some(f) = num.as_f64() {
                Ok(f.into_py(py))
            } else {
                Ok(py.None())
            }
        }
        Value::String(text) => Ok(text.into_py(py)),
        Value::Array(items) => {
            let list = PyList::empty(py);
            for item in items {
                let py_item = value_to_py(py, item)?;
                list.append(py_item)?;
            }
            Ok(list.into())
        }
        Value::Object(entries) => {
            let dict = PyDict::new(py);
            for (key, item) in entries {
                let py_value = value_to_py(py, item)?;
                dict.set_item(key, py_value)?;
            }
            Ok(dict.into())
        }
    }
}

fn parse_ask_mode(mode: &str) -> PyResult<AskMode> {
    match mode.to_ascii_lowercase().as_str() {
        "lex" | "lexical" => Ok(AskMode::Lex),
        "sem" | "semantic" => Ok(AskMode::Sem),
        "hybrid" | "auto" => Ok(AskMode::Hybrid),
        other => Err(PyValueError::new_err(format!(
            "invalid ask mode '{other}' (choose 'lex', 'semantic', or 'hybrid')"
        ))),
    }
}

fn parse_acl_enforcement_mode(mode: &str) -> PyResult<AclEnforcementMode> {
    match mode.trim().to_ascii_lowercase().as_str() {
        "" | "audit" => Ok(AclEnforcementMode::Audit),
        "enforce" => Ok(AclEnforcementMode::Enforce),
        other => Err(PyValueError::new_err(format!(
            "invalid acl_enforcement_mode '{other}' (choose 'audit' or 'enforce')"
        ))),
    }
}

fn parse_acl_context_py(acl_context: Option<&PyDict>) -> PyResult<Option<AclContext>> {
    let Some(acl_context) = acl_context else {
        return Ok(None);
    };

    let tenant_id = extract_optional_string_with_aliases(acl_context, &["tenant_id", "tenantId"])?;
    let subject_id =
        extract_optional_string_with_aliases(acl_context, &["subject_id", "subjectId"])?;
    let roles = extract_string_list_with_aliases(acl_context, &["roles"])?;
    let group_ids = extract_string_list_with_aliases(acl_context, &["group_ids", "groupIds"])?;

    if tenant_id.is_none() && subject_id.is_none() && roles.is_empty() && group_ids.is_empty() {
        return Ok(None);
    }

    Ok(Some(AclContext {
        tenant_id,
        subject_id,
        roles,
        group_ids,
    }))
}

fn extract_optional_string_with_aliases(dict: &PyDict, keys: &[&str]) -> PyResult<Option<String>> {
    for key in keys {
        if let Some(value) = dict.get_item(*key)? {
            let parsed = value.extract::<String>()?;
            let trimmed = parsed.trim();
            if trimmed.is_empty() {
                return Ok(None);
            }
            return Ok(Some(trimmed.to_string()));
        }
    }
    Ok(None)
}

fn extract_string_list_with_aliases(dict: &PyDict, keys: &[&str]) -> PyResult<Vec<String>> {
    for key in keys {
        if let Some(value) = dict.get_item(*key)? {
            return value.extract::<Vec<String>>();
        }
    }
    Ok(Vec::new())
}

#[derive(Clone)]
struct StaticEmbedder {
    embedding: Vec<f32>,
}

impl VecEmbedder for StaticEmbedder {
    fn embed_query(&self, _text: &str) -> MemvidResult<Vec<f32>> {
        Ok(self.embedding.clone())
    }

    fn embedding_dimension(&self) -> usize {
        self.embedding.len()
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut sum_a = 0.0f32;
    let mut sum_b = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        sum_a += x * x;
        sum_b += y * y;
    }
    if sum_a <= f32::EPSILON || sum_b <= f32::EPSILON {
        0.0
    } else {
        dot / (sum_a.sqrt() * sum_b.sqrt())
    }
}

fn parse_cutoff_strategy(
    strategy: Option<&str>,
    min_relevancy: f32,
) -> MemvidResult<CutoffStrategy> {
    match strategy
        .unwrap_or("relative")
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "relative" => Ok(CutoffStrategy::RelativeThreshold {
            min_ratio: min_relevancy,
        }),
        "absolute" => Ok(CutoffStrategy::AbsoluteThreshold {
            min_score: min_relevancy,
        }),
        "cliff" => Ok(CutoffStrategy::ScoreCliff {
            max_drop_ratio: 0.35,
        }),
        "elbow" => Ok(CutoffStrategy::Elbow { sensitivity: 1.0 }),
        "combined" => Ok(CutoffStrategy::Combined {
            relative_threshold: min_relevancy,
            max_drop_ratio: 0.35,
            absolute_min: 0.3,
        }),
        other => Err(MemvidCoreError::InvalidQuery {
            reason: format!("invalid adaptive strategy '{other}'"),
        }),
    }
}

fn apply_semantic_rerank_with_embedding(
    mem: &mut MemvidCore,
    query_embedding: &[f32],
    response: &mut SearchResponse,
) -> MemvidResult<bool> {
    use std::cmp::Ordering;

    if response.hits.is_empty() || query_embedding.is_empty() {
        return Ok(false);
    }

    let mut semantic_scores: HashMap<u64, f32> = HashMap::new();
    for hit in &response.hits {
        if let Some(embedding) = mem.frame_embedding(hit.frame_id)? {
            if embedding.len() == query_embedding.len() {
                let score = cosine_similarity(query_embedding, &embedding);
                semantic_scores.insert(hit.frame_id, score);
            }
        }
    }

    if semantic_scores.is_empty() {
        return Ok(false);
    }

    // Sort by semantic score to get semantic ranks.
    let mut sorted_semantic: Vec<(u64, f32)> = semantic_scores
        .iter()
        .map(|(frame_id, score)| (*frame_id, *score))
        .collect();
    sorted_semantic.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

    let mut semantic_rank: HashMap<u64, usize> = HashMap::new();
    for (idx, (frame_id, _)) in sorted_semantic.iter().enumerate() {
        semantic_rank.insert(*frame_id, idx + 1);
    }

    // Pure RRF: use ranks only.
    const RRF_K: f32 = 60.0;

    let mut ordering: Vec<(usize, f32, usize)> = response
        .hits
        .iter()
        .enumerate()
        .map(|(idx, hit)| {
            let lexical_rank = hit.rank;
            let lexical_rrf = 1.0 / (RRF_K + lexical_rank as f32);
            let semantic_rrf = semantic_rank
                .get(&hit.frame_id)
                .map(|rank| 1.0 / (RRF_K + *rank as f32))
                .unwrap_or(0.0);
            let combined = lexical_rrf + semantic_rrf;
            (idx, combined, lexical_rank)
        })
        .collect();

    ordering.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(Ordering::Equal)
            .then(a.2.cmp(&b.2))
    });

    let mut reordered = Vec::with_capacity(response.hits.len());
    for (rank_idx, (idx, combined_score, _)) in ordering.into_iter().enumerate() {
        let mut hit = response.hits[idx].clone();
        hit.rank = rank_idx + 1;
        // Replace score with RRF combined score (always positive)
        hit.score = Some(combined_score);
        reordered.push(hit);
    }

    response.hits = reordered;
    Ok(true)
}

fn build_context_for_hits(hits: &[SearchHit]) -> String {
    if hits.is_empty() {
        return String::new();
    }

    const MAX_CONTEXT_HITS: usize = 24;

    struct GroupSummary {
        indices: Vec<usize>,
        total_matches: usize,
        best_rank: usize,
    }
    impl Default for GroupSummary {
        fn default() -> Self {
            Self {
                indices: Vec::new(),
                total_matches: 0,
                best_rank: usize::MAX,
            }
        }
    }

    // Group hits by base URI using BTreeMap for deterministic iteration order.
    let mut groups: BTreeMap<String, GroupSummary> = BTreeMap::new();
    for (idx, hit) in hits.iter().enumerate() {
        let base = hit
            .uri
            .split('#')
            .next()
            .unwrap_or(&hit.uri)
            .to_ascii_lowercase();
        let entry = groups.entry(base).or_default();
        entry.indices.push(idx);
        entry.total_matches += hit.matches.max(1);
        entry.best_rank = entry.best_rank.min(hit.rank);
    }

    let mut selected_indices: Vec<usize> = Vec::with_capacity(MAX_CONTEXT_HITS);
    let mut seen_uris: HashSet<String> = HashSet::new();

    let mut sorted_groups: Vec<(String, GroupSummary)> = groups.into_iter().collect();
    sorted_groups.sort_by(|a, b| {
        a.1.best_rank
            .cmp(&b.1.best_rank)
            .then(b.1.total_matches.cmp(&a.1.total_matches))
    });

    // First pass: take best hit from each unique document.
    for (uri, group) in &sorted_groups {
        if selected_indices.len() >= MAX_CONTEXT_HITS {
            break;
        }
        if !seen_uris.contains(uri) {
            if let Some(&best_idx) = group.indices.first() {
                selected_indices.push(best_idx);
                seen_uris.insert(uri.clone());
            }
        }
    }

    // Second pass: fill remaining slots by rank order.
    if selected_indices.len() < MAX_CONTEXT_HITS {
        let mut remaining: Vec<(usize, usize)> = hits
            .iter()
            .enumerate()
            .filter(|(idx, _)| !selected_indices.contains(idx))
            .map(|(idx, hit)| (idx, hit.rank))
            .collect();
        remaining.sort_by_key(|(_, rank)| *rank);
        for (idx, _) in remaining {
            if selected_indices.len() >= MAX_CONTEXT_HITS {
                break;
            }
            selected_indices.push(idx);
        }
    }

    selected_indices.sort_unstable();
    selected_indices
        .into_iter()
        .filter_map(|idx| hits.get(idx))
        .map(|hit| {
            let display_uri = hit.uri.strip_prefix("mv2://").unwrap_or(&hit.uri);
            let heading = hit.title.as_deref().unwrap_or(display_uri);
            format!(
                "### [{}] {} — {}\n{}\n(matches: {})",
                hit.rank, display_uri, heading, hit.text, hit.matches
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn stats_to_py(py: Python<'_>, stats: &CoreStats) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("frame_count", stats.frame_count)?;
    dict.set_item("size_bytes", stats.size_bytes)?;
    dict.set_item("tier", format!("{:?}", stats.tier).to_ascii_lowercase())?;
    dict.set_item("has_lex_index", stats.has_lex_index)?;
    dict.set_item("has_vec_index", stats.has_vec_index)?;
    dict.set_item("has_time_index", stats.has_time_index)?;
    dict.set_item("seq_no", stats.seq_no)?;
    dict.set_item("capacity_bytes", stats.capacity_bytes)?;
    dict.set_item("active_frame_count", stats.active_frame_count)?;
    dict.set_item("payload_bytes", stats.payload_bytes)?;
    dict.set_item("logical_bytes", stats.logical_bytes)?;
    dict.set_item("saved_bytes", stats.saved_bytes)?;
    dict.set_item("compression_ratio_percent", stats.compression_ratio_percent)?;
    dict.set_item("savings_percent", stats.savings_percent)?;
    dict.set_item(
        "storage_utilisation_percent",
        stats.storage_utilisation_percent,
    )?;
    dict.set_item("remaining_capacity_bytes", stats.remaining_capacity_bytes)?;
    dict.set_item(
        "average_frame_payload_bytes",
        stats.average_frame_payload_bytes,
    )?;
    dict.set_item(
        "average_frame_logical_bytes",
        stats.average_frame_logical_bytes,
    )?;
    dict.set_item("wal_bytes", stats.wal_bytes)?;
    dict.set_item("lex_index_bytes", stats.lex_index_bytes)?;
    dict.set_item("vec_index_bytes", stats.vec_index_bytes)?;
    dict.set_item("time_index_bytes", stats.time_index_bytes)?;
    dict.set_item("lex_enabled", stats.lex_enabled)?;
    dict.set_item("vec_enabled", stats.vec_enabled)?;
    Ok(dict.into())
}

fn embedding_identity_to_py(py: Python<'_>, identity: &EmbeddingIdentity) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("provider", identity.provider.as_deref())?;
    dict.set_item("model", identity.model.as_deref())?;
    dict.set_item("dimension", identity.dimension)?;
    dict.set_item("normalized", identity.normalized)?;
    Ok(dict.into())
}

fn embedding_identity_summary_to_py(
    py: Python<'_>,
    summary: &EmbeddingIdentitySummary,
) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    match summary {
        EmbeddingIdentitySummary::Unknown => {
            dict.set_item("kind", "unknown")?;
        }
        EmbeddingIdentitySummary::Single(identity) => {
            dict.set_item("kind", "single")?;
            dict.set_item("identity", embedding_identity_to_py(py, identity)?)?;
        }
        EmbeddingIdentitySummary::Mixed(identities) => {
            dict.set_item("kind", "mixed")?;
            let list = PyList::empty(py);
            for item in identities {
                let entry = PyDict::new(py);
                entry.set_item("identity", embedding_identity_to_py(py, &item.identity)?)?;
                entry.set_item("count", item.count)?;
                list.append(entry)?;
            }
            dict.set_item("identities", list)?;
        }
    }
    Ok(dict.into())
}

fn build_find_result(
    py: Python<'_>,
    mem: &mut MemvidCore,
    response: SearchResponse,
    snippet_limit: usize,
) -> PyResult<PyObject> {
    let hits_list = PyList::empty(py);
    for hit in &response.hits {
        let frame = mem
            .frame_by_id(hit.frame_id)
            .map_err(MemvidCorePy::core_err)?;
        // Get full text content - prefer search_text, fall back to chunk_text, then hit.text
        let full_text = frame
            .search_text
            .clone()
            .filter(|t| !t.is_empty())
            .or_else(|| hit.chunk_text.clone())
            .or_else(|| mem.frame_text_by_id(hit.frame_id).ok())
            .unwrap_or_else(|| hit.text.clone());
        // Clean the text (normalize whitespace)
        let cleaned_text = clean_text_for_display(&full_text);
        // Clip for snippet
        let snippet = clip_text(&cleaned_text, snippet_limit);
        let hit_obj = make_hit_dict(py, hit, &frame, Some(snippet), Some(cleaned_text))?;
        hits_list.append(hit_obj)?;
    }

    let result = PyDict::new(py);
    result.set_item("query", response.query)?;
    result.set_item("hits", hits_list)?;
    result.set_item("took_ms", response.elapsed_ms as u64)?;
    result.set_item("total_hits", response.total_hits as u64)?;
    result.set_item(
        "engine",
        format!("{:?}", response.engine).to_ascii_lowercase(),
    )?;
    if let Some(cursor) = response.next_cursor {
        result.set_item("next_cursor", cursor)?;
    }
    if response.stale_index_skips > 0 {
        result.set_item("stale_index_skips", response.stale_index_skips)?;
    }
    result.set_item("context", response.context)?;
    Ok(result.into())
}

fn make_hit_dict(
    py: Python<'_>,
    hit: &SearchHit,
    frame: &Frame,
    snippet: Option<String>,
    text: Option<String>,
) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("frame_id", hit.frame_id)?;
    dict.set_item(
        "uri",
        frame
            .uri
            .clone()
            .unwrap_or_else(|| format!("mv2://frames/{}", hit.frame_id)),
    )?;
    dict.set_item("title", frame.title.clone())?;
    dict.set_item("rank", hit.rank as u64)?;
    dict.set_item("score", hit.score.unwrap_or(0.0))?;
    dict.set_item("matches", hit.matches as u64)?;
    if let Some(snippet) = snippet {
        dict.set_item("snippet", snippet)?;
    } else {
        dict.set_item("snippet", py.None())?;
    }
    // Add full cleaned text
    if let Some(text) = text {
        dict.set_item("text", text)?;
    } else {
        dict.set_item("text", py.None())?;
    }
    dict.set_item("tags", frame.tags.clone())?;
    dict.set_item("labels", frame.labels.clone())?;
    if let Some(meta) = &hit.metadata {
        dict.set_item("track", meta.track.clone())?;
        dict.set_item("created_at", meta.created_at.clone())?;
        dict.set_item("content_dates", meta.content_dates.clone())?;
    }
    Ok(dict.into())
}

fn apply_model_context_fragments(
    response: &mut CoreAskResponse,
    fragments: Vec<ModelContextFragment>,
) {
    if fragments.is_empty() {
        return;
    }

    response.context_fragments = fragments
        .into_iter()
        .map(|fragment| AskContextFragment {
            rank: fragment.rank,
            frame_id: fragment.frame_id,
            uri: fragment.uri,
            title: fragment.title,
            score: fragment.score,
            matches: fragment.matches,
            range: Some(fragment.range),
            chunk_range: fragment.chunk_range,
            text: fragment.text,
            kind: Some(match fragment.kind {
                ModelContextFragmentKind::Full => AskContextFragmentKind::Full,
                ModelContextFragmentKind::Summary => AskContextFragmentKind::Summary,
            }),
            #[cfg(feature = "temporal_track")]
            temporal: None,
        })
        .collect();
}

fn clip_text(text: &str, limit: usize) -> String {
    if limit == 0 {
        return String::new();
    }
    let mut iter = text.chars();
    let mut out = String::new();
    for _ in 0..limit {
        match iter.next() {
            Some(ch) => out.push(ch),
            None => return out,
        }
    }
    if iter.next().is_some() {
        out.push_str("...");
    }
    out
}

fn build_ask_result(
    py: Python<'_>,
    mem: &mut MemvidCore,
    response: CoreAskResponse,
    model_info: Option<(String, String)>,
    return_sources: bool,
    show_chunks: bool,
) -> PyResult<PyObject> {
    let top_fragment = response
        .context_fragments
        .iter()
        .min_by_key(|fragment| fragment.rank)
        .cloned();

    let mut primary_source_obj: Option<PyObject> = None;
    let mut primary_context_title: Option<String> = None;
    let mut primary_context_uri: Option<String> = None;
    let mut primary_context_snippet: Option<String> = None;

    let hits_list = PyList::empty(py);
    for hit in response.retrieval.hits.iter().take(1) {
        let frame = mem
            .frame_by_id(hit.frame_id)
            .map_err(MemvidCorePy::core_err)?;
        // Get full text content
        let full_text = frame
            .search_text
            .clone()
            .filter(|t| !t.is_empty())
            .or_else(|| hit.chunk_text.clone())
            .or_else(|| mem.frame_text_by_id(hit.frame_id).ok())
            .unwrap_or_else(|| hit.text.clone());
        let cleaned_text = clean_text_for_display(&full_text);
        let snippet_for_hit = Some(clip_text(&cleaned_text, MAX_HIT_SNIPPET_CHARS));
        let snippet_for_context = Some(clip_text(&cleaned_text, MAX_CONTEXT_SNIPPET_CHARS));
        let hit_obj = make_hit_dict(py, hit, &frame, snippet_for_hit.clone(), Some(cleaned_text))?;
        if primary_source_obj.is_none() {
            primary_source_obj = Some(hit_obj.clone_ref(py));
            primary_context_snippet = snippet_for_context;
            primary_context_title = frame.title.clone();
            primary_context_uri = Some(
                frame
                    .uri
                    .clone()
                    .unwrap_or_else(|| format!("mv2://frames/{}", hit.frame_id)),
            );
        }
        hits_list.append(hit_obj)?;
    }

    let mut by_frame: HashMap<u64, SearchHit> = HashMap::new();
    for hit in &response.retrieval.hits {
        by_frame.insert(hit.frame_id, hit.clone());
    }

    // Build sources list - full detail if return_sources=true, otherwise just one summary
    let sources_list = PyList::empty(py);
    if return_sources {
        // Return all sources with full SourceSpan-style metadata
        for (idx, citation) in response.citations.iter().enumerate() {
            let frame = mem
                .frame_by_id(citation.frame_id)
                .map_err(MemvidCorePy::core_err)?;
            let hit = by_frame.get(&citation.frame_id);

            let source = PyDict::new(py);
            source.set_item("index", idx + 1)?;
            source.set_item("frame_id", citation.frame_id)?;
            source.set_item(
                "uri",
                frame
                    .uri
                    .clone()
                    .unwrap_or_else(|| format!("mv2://frames/{}", citation.frame_id)),
            )?;
            source.set_item("title", frame.title.clone())?;
            if let Some(range) = citation.chunk_range {
                let tuple = PyTuple::new(py, [range.0, range.1]);
                source.set_item("chunk_range", tuple)?;
            } else {
                source.set_item("chunk_range", py.None())?;
            }
            source.set_item("score", citation.score.unwrap_or(0.0))?;
            source.set_item("tags", frame.tags.clone())?;
            source.set_item("labels", frame.labels.clone())?;
            source.set_item("frame_timestamp", frame.timestamp)?;
            source.set_item("content_dates", frame.content_dates.clone())?;

            // Include snippet from hit if available
            if let Some(h) = hit {
                if let Some(snippet) = h.chunk_text.clone().or_else(|| Some(h.text.clone())) {
                    source.set_item("snippet", clip_text(&snippet, MAX_CONTEXT_SNIPPET_CHARS))?;
                } else {
                    source.set_item("snippet", py.None())?;
                }
            } else {
                source.set_item("snippet", py.None())?;
            }

            sources_list.append(source)?;
        }
    } else {
        // Legacy behavior: just one summary source
        for citation in response.citations.iter().take(1) {
            if let Some(hit) = by_frame.get(&citation.frame_id) {
                let frame = mem
                    .frame_by_id(hit.frame_id)
                    .map_err(MemvidCorePy::core_err)?;
                let source = PyDict::new(py);
                source.set_item(
                    "uri",
                    frame
                        .uri
                        .clone()
                        .unwrap_or_else(|| format!("mv2://frames/{}", hit.frame_id)),
                )?;
                source.set_item("frame_id", hit.frame_id)?;
                source.set_item("title", frame.title.clone())?;
                source.set_item("score", citation.score.unwrap_or(0.0))?;
                if let Some(snippet) = hit.chunk_text.clone().or_else(|| Some(hit.text.clone())) {
                    source.set_item("snippet", clip_text(&snippet, MAX_CONTEXT_SNIPPET_CHARS))?;
                } else {
                    source.set_item("snippet", py.None())?;
                }
                sources_list.append(source)?;
            }
        }
    }

    let stats = &response.stats;
    let stats_dict = PyDict::new(py);
    stats_dict.set_item("retrieval_ms", stats.retrieval_ms as u64)?;
    stats_dict.set_item("synthesis_ms", stats.synthesis_ms as u64)?;
    stats_dict.set_item("latency_ms", stats.latency_ms as u64)?;

    let usage_dict = PyDict::new(py);
    usage_dict.set_item("retrieved", response.retrieval.total_hits as u64)?;

    let fragments_list = PyList::empty(py);
    if let Some(fragment) = top_fragment.as_ref() {
        let frag = PyDict::new(py);
        frag.set_item("rank", fragment.rank)?;
        frag.set_item("frame_id", fragment.frame_id)?;
        frag.set_item("uri", &fragment.uri)?;
        frag.set_item("title", &fragment.title)?;
        frag.set_item("score", fragment.score)?;
        frag.set_item("matches", fragment.matches)?;
        if let Some(range) = fragment.range {
            let tuple = PyTuple::new(py, [range.0, range.1]);
            frag.set_item("range", tuple)?;
        } else {
            frag.set_item("range", py.None())?;
        }
        if let Some(range) = fragment.chunk_range {
            let tuple = PyTuple::new(py, [range.0, range.1]);
            frag.set_item("chunk_range", tuple)?;
        } else {
            frag.set_item("chunk_range", py.None())?;
        }
        frag.set_item("text", clip_text(&fragment.text, MAX_CONTEXT_SNIPPET_CHARS))?;
        let kind_value = fragment
            .kind
            .as_ref()
            .map(|kind| format!("{:?}", kind).to_ascii_lowercase());
        frag.set_item("kind", kind_value)?;
        fragments_list.append(frag)?;
    }

    let result = PyDict::new(py);
    result.set_item("question", response.question)?;
    result.set_item("answer", response.answer.clone())?;
    result.set_item("mode", format!("{:?}", response.mode).to_ascii_lowercase())?;
    result.set_item(
        "retriever",
        format!("{:?}", response.retriever).to_ascii_lowercase(),
    )?;
    result.set_item("context_only", response.context_only)?;
    let context_text = if let Some(snippet) = primary_context_snippet.clone() {
        let mut header_parts: Vec<String> = Vec::new();
        if let Some(title) = primary_context_title
            .as_ref()
            .filter(|t| !t.trim().is_empty())
        {
            header_parts.push(title.trim().to_string());
        }
        if let Some(uri) = primary_context_uri.as_ref() {
            header_parts.push(uri.clone());
        }
        if header_parts.is_empty() {
            Some(snippet)
        } else {
            Some(format!("{}\n{}", header_parts.join(" — "), snippet))
        }
    } else if let Some(fragment) = top_fragment.as_ref() {
        Some(clip_text(&fragment.text, MAX_CONTEXT_SNIPPET_CHARS))
    } else if !response.retrieval.context.is_empty() {
        Some(clip_text(
            &response.retrieval.context,
            MAX_CONTEXT_SNIPPET_CHARS,
        ))
    } else {
        None
    };
    if let Some(text) = context_text {
        result.set_item("context", text)?;
    } else {
        result.set_item("context", py.None())?;
    }
    result.set_item("hits", hits_list)?;
    result.set_item("sources", sources_list)?;
    if let Some(primary) = primary_source_obj {
        result.set_item("primary_source", primary)?;
    } else {
        result.set_item("primary_source", py.None())?;
    }
    result.set_item("stats", stats_dict)?;
    result.set_item("usage", usage_dict)?;
    if let Some((requested, used)) = model_info {
        result.set_item("model", &requested)?;
        if requested != used {
            result.set_item("model_used", &used)?;
        }
    }
    result.set_item("context_fragments", fragments_list)?;
    if let Some(answer_text) = response.answer.clone() {
        let lines: Vec<String> = answer_text
            .lines()
            .map(|line| line.trim())
            .filter(|line| !line.is_empty())
            .map(|line| line.to_string())
            .collect();
        if !lines.is_empty() {
            result.set_item("answer_lines", lines)?;
        }
    }

    // Add full chunks when show_chunks=true
    if show_chunks {
        let chunks_list = PyList::empty(py);
        for hit in response.retrieval.hits.iter() {
            let chunk_dict = PyDict::new(py);
            let frame = mem
                .frame_by_id(hit.frame_id)
                .map_err(MemvidCorePy::core_err)?;
            // Get full text content - prefer search_text (where text is stored for put()),
            // fall back to frame_text_by_id, then hit.text
            let full_content = frame
                .search_text
                .clone()
                .filter(|t| !t.is_empty())
                .or_else(|| mem.frame_text_by_id(hit.frame_id).ok())
                .unwrap_or_else(|| hit.text.clone());
            // Clean up the content for display (normalize whitespace)
            let cleaned_content = clean_text_for_display(&full_content);
            chunk_dict.set_item("frame_id", hit.frame_id)?;
            chunk_dict.set_item(
                "uri",
                frame
                    .uri
                    .clone()
                    .unwrap_or_else(|| format!("mv2://frames/{}", hit.frame_id)),
            )?;
            chunk_dict.set_item("title", frame.title.clone())?;
            chunk_dict.set_item("score", hit.score)?;
            chunk_dict.set_item("content", cleaned_content)?;
            if let Some(range) = hit.chunk_range {
                let tuple = PyTuple::new(py, [range.0, range.1]);
                chunk_dict.set_item("chunk_range", tuple)?;
            } else {
                chunk_dict.set_item("chunk_range", py.None())?;
            }
            chunks_list.append(chunk_dict)?;
        }
        result.set_item("chunks", chunks_list)?;
    }

    Ok(result.into())
}

fn build_audit_result(py: Python<'_>, report: AuditReport) -> PyResult<PyObject> {
    let result = PyDict::new(py);
    result.set_item("version", &report.version)?;
    result.set_item("generated_at", report.generated_at)?;
    result.set_item("question", &report.question)?;
    result.set_item("answer", report.answer.clone())?;
    result.set_item("mode", format!("{:?}", report.mode).to_ascii_lowercase())?;
    result.set_item(
        "retriever",
        format!("{:?}", report.retriever).to_ascii_lowercase(),
    )?;

    // Build sources list
    let sources_list = PyList::empty(py);
    for source in &report.sources {
        let source_dict = PyDict::new(py);
        source_dict.set_item("index", source.index)?;
        source_dict.set_item("frame_id", source.frame_id)?;
        source_dict.set_item("uri", &source.uri)?;
        source_dict.set_item("title", source.title.clone())?;
        if let Some(range) = source.chunk_range {
            let tuple = PyTuple::new(py, [range.0, range.1]);
            source_dict.set_item("chunk_range", tuple)?;
        } else {
            source_dict.set_item("chunk_range", py.None())?;
        }
        source_dict.set_item("score", source.score.unwrap_or(0.0))?;
        source_dict.set_item("tags", source.tags.clone())?;
        source_dict.set_item("labels", source.labels.clone())?;
        source_dict.set_item("frame_timestamp", source.frame_timestamp)?;
        source_dict.set_item("content_dates", source.content_dates.clone())?;
        source_dict.set_item("snippet", source.snippet.clone())?;
        sources_list.append(source_dict)?;
    }
    result.set_item("sources", sources_list)?;

    result.set_item("total_hits", report.total_hits)?;

    // Build stats dict
    let stats_dict = PyDict::new(py);
    stats_dict.set_item("retrieval_ms", report.stats.retrieval_ms as u64)?;
    stats_dict.set_item("synthesis_ms", report.stats.synthesis_ms as u64)?;
    stats_dict.set_item("latency_ms", report.stats.latency_ms as u64)?;
    result.set_item("stats", stats_dict)?;

    result.set_item("notes", report.notes.clone())?;

    Ok(result.into())
}

fn build_timeline(
    py: Python<'_>,
    entries: Vec<memvid_core::types::TimelineEntry>,
) -> PyResult<PyObject> {
    let list = PyList::empty(py);
    for entry in entries {
        let dict = PyDict::new(py);
        dict.set_item("frame_id", entry.frame_id)?;
        dict.set_item("timestamp", entry.timestamp)?;
        dict.set_item("preview", entry.preview)?;
        dict.set_item("uri", entry.uri)?;
        dict.set_item("child_frames", entry.child_frames)?;
        list.append(dict)?;
    }
    Ok(list.into())
}

fn make_timeline_query(
    limit: u64,
    since: Option<i64>,
    until: Option<i64>,
    reverse: bool,
) -> TimelineQuery {
    let mut builder = TimelineQueryBuilder::default();
    let capped = if limit == 0 {
        DEFAULT_TIMELINE_LIMIT
    } else {
        limit
    };
    let capped = if capped == 0 { 1 } else { capped };
    builder = builder.limit(NonZeroU64::new(capped).expect("capped is non-zero"));
    if let Some(start) = since {
        builder = builder.since(start);
    }
    if let Some(end) = until {
        builder = builder.until(end);
    }
    builder = builder.reverse(reverse);
    builder.build()
}

#[cfg(feature = "parallel_segments")]
fn parallel_env_default() -> bool {
    env::var("MEMVID_PARALLEL_SEGMENTS")
        .ok()
        .and_then(|value| match value.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Some(true),
            "0" | "false" | "no" | "off" => Some(false),
            _ => None,
        })
        .unwrap_or(true)
}

#[cfg(not(feature = "parallel_segments"))]
fn parallel_env_default() -> bool {
    false
}

#[allow(dead_code)]
fn with_memvid_simple<R, F>(
    path: &str,
    mode: OpenMode,
    enable_lex: bool,
    enable_vec: bool,
    mutating: bool,
    mut f: F,
) -> PyResult<R>
where
    F: FnMut(&mut MemvidCore) -> PyResult<R>,
{
    let mut mem =
        MemvidCorePy::open_inner(Path::new(path), mode).map_err(MemvidCorePy::core_err)?;
    MemvidCorePy::apply_default_lock_settings(&mut mem);
    if mem.is_read_only() {
        MemvidCorePy::ensure_indexes_available(&mem, enable_lex, enable_vec)
            .map_err(MemvidCorePy::core_err)?;
    } else {
        let mut vec_requested = enable_vec;
        loop {
            match MemvidCorePy::configure_indexes(&mut mem, enable_lex, vec_requested) {
                Ok(_) => break,
                Err(MemvidCoreError::FeatureUnavailable { feature })
                    if feature == "vec" && vec_requested =>
                {
                    vec_requested = false;
                    mem = MemvidCorePy::open_inner(Path::new(path), OpenMode::Open)
                        .map_err(MemvidCorePy::core_err)?;
                    MemvidCorePy::apply_default_lock_settings(&mut mem);
                    continue;
                }
                Err(err) => return Err(MemvidCorePy::core_err(err)),
            }
        }
    }
    let result = f(&mut mem)?;
    if mutating && !mem.is_read_only() {
        mem.commit().map_err(MemvidCorePy::core_err)?;
    }
    Ok(result)
}

fn with_memvid_with_py<'py, R, F>(
    py: Python<'py>,
    path: &str,
    mode: OpenMode,
    enable_lex: bool,
    enable_vec: bool,
    mutating: bool,
    mut f: F,
) -> PyResult<R>
where
    F: FnMut(Python<'py>, &mut MemvidCore) -> PyResult<R>,
{
    let mut mem =
        MemvidCorePy::open_inner(Path::new(path), mode).map_err(MemvidCorePy::core_err)?;
    MemvidCorePy::apply_default_lock_settings(&mut mem);
    if mem.is_read_only() {
        MemvidCorePy::ensure_indexes_available(&mem, enable_lex, enable_vec)
            .map_err(MemvidCorePy::core_err)?;
    } else {
        let mut vec_requested = enable_vec;
        loop {
            match MemvidCorePy::configure_indexes(&mut mem, enable_lex, vec_requested) {
                Ok(_) => break,
                Err(MemvidCoreError::FeatureUnavailable { feature })
                    if feature == "vec" && vec_requested =>
                {
                    vec_requested = false;
                    mem = MemvidCorePy::open_inner(Path::new(path), OpenMode::Open)
                        .map_err(MemvidCorePy::core_err)?;
                    MemvidCorePy::apply_default_lock_settings(&mut mem);
                    continue;
                }
                Err(err) => return Err(MemvidCorePy::core_err(err)),
            }
        }
    }
    let result = f(py, &mut mem)?;
    if mutating && !mem.is_read_only() {
        mem.commit().map_err(MemvidCorePy::core_err)?;
    }
    Ok(result)
}

#[pyfunction]
#[pyo3(signature = (path, *, enable_lex = true, enable_vec = true, mode = "auto", read_only = true, lock_timeout_ms = DEFAULT_LOCK_TIMEOUT_MS, force = None, force_writable = false, api_key = None))]
fn open(
    py: Python<'_>,
    path: &str,
    enable_lex: bool,
    enable_vec: bool,
    mode: &str,
    read_only: bool,
    lock_timeout_ms: u64,
    force: Option<&str>,
    force_writable: bool,
    api_key: Option<String>,
) -> PyResult<Py<MemvidCorePy>> {
    let instance = MemvidCorePy::new(
        path.to_string(),
        mode,
        enable_lex,
        enable_vec,
        read_only,
        lock_timeout_ms,
        force,
        force_writable,
        api_key,
    )?;
    Py::new(py, instance)
}

#[pyfunction]
#[pyo3(signature = (path, *, enable_lex = true, enable_vec = true, lock_timeout_ms = DEFAULT_LOCK_TIMEOUT_MS, force = None, api_key = None))]
fn create(
    py: Python<'_>,
    path: &str,
    enable_lex: bool,
    enable_vec: bool,
    lock_timeout_ms: u64,
    force: Option<&str>,
    api_key: Option<String>,
) -> PyResult<Py<MemvidCorePy>> {
    let force_stale = match force {
        Some("stale_only") => true,
        Some(value) => {
            return Err(PyValueError::new_err(format!(
                "unsupported force option '{value}'; expected 'stale_only'"
            )));
        }
        None => false,
    };
    let instance = MemvidCorePy::from_path(
        PathBuf::from(path),
        OpenMode::Create,
        enable_lex,
        enable_vec,
        false,
        false,
        lock_timeout_ms,
        force_stale,
        api_key,
    )?;
    Py::new(py, instance)
}

#[pyfunction]
#[pyo3(signature = (path, *, text = None, file = None, uri = None, title = None, kind = None, track = None, tags = None, labels = None, metadata = None, search_text = None, enable_embedding = true, auto_tag = true, extract_dates = true, parallel = None, no_raw = true, source_path = None))]
fn put(
    path: &str,
    text: Option<&str>,
    file: Option<&str>,
    uri: Option<&str>,
    title: Option<&str>,
    kind: Option<&str>,
    track: Option<&str>,
    tags: Option<Vec<String>>,
    labels: Option<Vec<String>>,
    metadata: Option<HashMap<String, String>>,
    search_text: Option<&str>,
    enable_embedding: bool,
    auto_tag: bool,
    extract_dates: bool,
    parallel: Option<bool>,
    no_raw: bool,
    source_path: Option<&str>,
) -> PyResult<u64> {
    let mut mem = MemvidCorePy::open_inner(Path::new(path), OpenMode::Auto)
        .map_err(MemvidCorePy::core_err)?;
    MemvidCorePy::apply_default_lock_settings(&mut mem);
    MemvidCorePy::configure_indexes(&mut mem, true, true).map_err(MemvidCorePy::core_err)?;
    let title = title
        .map(|value| value.to_string())
        .ok_or_else(|| PyValueError::new_err("legacy put requires a 'title'"))?;
    let label_values = labels.clone().unwrap_or_default();
    if label_values.is_empty() {
        return Err(PyValueError::new_err(
            "legacy put requires at least one label",
        ));
    }
    let legacy_metadata_value = metadata.clone().map(|entries| {
        let mut map = serde_json::Map::new();
        for (k, v) in entries {
            map.insert(k, Value::String(v));
        }
        Value::Object(map)
    });
    let (doc_metadata, extra_metadata) = split_metadata(legacy_metadata_value)?;

    // Check if OPENAI_API_KEY is available for auto-model selection
    let has_openai_key = env::var("OPENAI_API_KEY")
        .ok()
        .map(|k| !k.trim().is_empty())
        .unwrap_or(false);

    let payload = PutPayload {
        text: text.map(|value| value.to_string()),
        file: file.map(|value| value.to_string()),
        uri: uri.map(|value| value.to_string()),
        title,
        kind: kind.map(|value| value.to_string()),
        track: track.map(|value| value.to_string()),
        tags: tags.clone().unwrap_or_default(),
        labels: label_values,
        doc_metadata,
        extra_metadata,
        search_text: search_text.map(|value| value.to_string()),
        enable_embedding,
        // Use OpenAI for model selection if key is available and embedding is enabled
        auto_enabled_via_openai: enable_embedding && has_openai_key,
        embedding_model: None,
        auto_tag,
        extract_dates,
        parallel: Some(parallel.unwrap_or_else(parallel_env_default)),
        no_raw,
        source_path: source_path.map(|value| value.to_string()),
        timestamp: None,
        enable_enrichment: true, // Default to enabled for standalone function
    };
    let seq = perform_put(&mut mem, &payload)?;
    MemvidCorePy::commit_after_put(&mut mem, payload.parallel)?;

    // Run rules-based enrichment
    if let Ok(frame) = mem.frame_by_id(seq) {
        let text = frame.search_text.as_deref().unwrap_or("");
        if !text.is_empty() {
            let mut rules_engine = RulesEngine::new();
            let _ = rules_engine.init();
            let ctx = EnrichmentContext::new(
                seq,
                frame.uri.clone().unwrap_or_default(),
                text.to_string(),
                frame.title.clone(),
                frame.timestamp,
                None,
            );
            let result = rules_engine.enrich(&ctx);
            let has_cards = !result.cards.is_empty();
            for card in result.cards {
                let _ = mem.put_memory_card(card);
            }
            if has_cards {
                let _ = mem.commit();
            }
        }
    }

    Ok(seq)
}

#[pyfunction]
#[pyo3(signature = (path, query, *, k = DEFAULT_FIND_K, snippet_chars = DEFAULT_FIND_SNIPPET, scope = None, cursor = None, as_of_frame = None, as_of_ts = None, acl_context = None, acl_enforcement_mode = "audit"))]
fn find(
    py: Python<'_>,
    path: &str,
    query: &str,
    k: usize,
    snippet_chars: usize,
    scope: Option<&str>,
    cursor: Option<&str>,
    as_of_frame: Option<u64>,
    as_of_ts: Option<i64>,
    acl_context: Option<&PyDict>,
    acl_enforcement_mode: &str,
) -> PyResult<PyObject> {
    let acl_context = parse_acl_context_py(acl_context)?;
    let acl_enforcement_mode = parse_acl_enforcement_mode(acl_enforcement_mode)?;
    with_memvid_with_py(
        py,
        path,
        OpenMode::ReadOnly,
        true,
        true,
        false,
        |py, mem| {
            let request = SearchRequest {
                query: query.to_string(),
                top_k: k,
                snippet_chars,
                uri: None,
                scope: scope.map(|s| s.to_string()),
                cursor: cursor.map(|c| c.to_string()),
                temporal: None,
                as_of_frame,
                as_of_ts,
                no_sketch: false,
                acl_context: acl_context.clone(),
                acl_enforcement_mode,
            };
            let response = mem.search(request).map_err(MemvidCorePy::core_err)?;
            let limit = snippet_chars.min(MAX_HIT_SNIPPET_CHARS);
            build_find_result(py, mem, response, limit)
        },
    )
}

#[pyfunction]
#[pyo3(signature = (path, question, *, k = DEFAULT_ASK_K, mode = "auto", snippet_chars = DEFAULT_ASK_SNIPPET, scope = None, since = None, until = None, context_only = false, model = None, llm_context_chars = None, api_key = None, return_sources = false, show_chunks = false, acl_context = None, acl_enforcement_mode = "audit"))]
fn ask(
    py: Python<'_>,
    path: &str,
    question: &str,
    k: usize,
    mode: &str,
    snippet_chars: usize,
    scope: Option<&str>,
    since: Option<i64>,
    until: Option<i64>,
    context_only: bool,
    model: Option<&str>,
    llm_context_chars: Option<usize>,
    api_key: Option<&str>,
    return_sources: bool,
    show_chunks: bool,
    acl_context: Option<&PyDict>,
    acl_enforcement_mode: &str,
) -> PyResult<PyObject> {
    let acl_context = parse_acl_context_py(acl_context)?;
    let acl_enforcement_mode = parse_acl_enforcement_mode(acl_enforcement_mode)?;
    with_memvid_with_py(
        py,
        path,
        OpenMode::ReadOnly,
        true,
        true,
        false,
        |py, mem| {
            let ask_mode = parse_ask_mode(mode)?;
            let request = AskRequest {
                question: question.to_string(),
                top_k: k,
                snippet_chars,
                uri: None,
                scope: scope.map(|s| s.to_string()),
                cursor: None,
                start: since,
                end: until,
                temporal: None,
                context_only,
                mode: ask_mode,
                as_of_frame: None,
                as_of_ts: None,
                adaptive: None,
                acl_context: acl_context.clone(),
                acl_enforcement_mode,
            };
            let mut response = mem
                .ask::<dyn VecEmbedder>(request, None::<&dyn VecEmbedder>)
                .map_err(MemvidCorePy::core_err)?;
            let mut model_info = None;
            if let Some(model_name) = model {
                match run_model_inference(
                    model_name,
                    question,
                    &response.retrieval.context,
                    response.retrieval.hits.as_slice(),
                    llm_context_chars,
                    api_key,
                    None, // system_prompt_override
                ) {
                    Ok(inference) => {
                        model_info = Some((
                            inference.answer.requested.clone(),
                            inference.answer.model.clone(),
                        ));
                        response.answer = Some(inference.answer.answer.clone());
                        response.retrieval.context = inference.context_body;
                        apply_model_context_fragments(&mut response, inference.context_fragments);
                    }
                    Err(err) => return Err(model_error_to_py(err)),
                }
            }
            build_ask_result(py, mem, response, model_info, return_sources, show_chunks)
        },
    )
}

/// Generate an audit report for a question (standalone function).
#[pyfunction]
#[pyo3(signature = (path, question, *, k = 10, mode = "auto", snippet_chars = 500, scope = None, since = None, until = None, include_snippets = true, out = None, format = "json"))]
fn audit(
    py: Python<'_>,
    path: &str,
    question: &str,
    k: usize,
    mode: &str,
    snippet_chars: usize,
    scope: Option<&str>,
    since: Option<i64>,
    until: Option<i64>,
    include_snippets: bool,
    out: Option<&str>,
    format: &str,
) -> PyResult<PyObject> {
    with_memvid_with_py(
        py,
        path,
        OpenMode::ReadOnly,
        true,
        true,
        false,
        |py, mem| {
            let ask_mode = parse_ask_mode(mode)?;

            let options = AuditOptions {
                top_k: Some(k),
                snippet_chars: Some(snippet_chars),
                mode: Some(ask_mode),
                scope: scope.map(|s| s.to_string()),
                start: since,
                end: until,
                include_snippets,
            };

            let report = mem
                .audit::<dyn VecEmbedder>(question, Some(options), None::<&dyn VecEmbedder>)
                .map_err(MemvidCorePy::core_err)?;

            // Handle output file if specified
            if let Some(out_path) = out {
                let content = match format {
                    "text" => report.to_text(),
                    "markdown" | "md" => report.to_markdown(),
                    "json" => serde_json::to_string_pretty(&report).map_err(|e| {
                        PyValueError::new_err(format!("JSON serialization error: {}", e))
                    })?,
                    _ => {
                        return Err(PyValueError::new_err(format!(
                            "Unknown format: {}. Expected 'text', 'markdown', or 'json'",
                            format
                        )));
                    }
                };
                fs::write(out_path, content).map_err(|e| {
                    PyIOError::new_err(format!("Failed to write output file: {}", e))
                })?;
            }

            build_audit_result(py, report)
        },
    )
}

#[pyfunction]
#[pyo3(signature = (path, *, deep = false))]
fn verify(py: Python<'_>, path: &str, deep: bool) -> PyResult<PyObject> {
    let report = MemvidCore::verify(path, deep).map_err(MemvidCorePy::core_err)?;
    let status = report.overall_status;
    let value = serde_json::to_value(&report)
        .map_err(|err| PyValueError::new_err(format!("failed to encode verify report: {err}")))?;
    if matches!(status, VerificationStatus::Failed) {
        let py_report = value_to_py(py, value.clone())?;
        let err = build_exception(
            py,
            py.get_type::<VerifyFailedError>(),
            "MV006",
            "verification failed".to_string(),
        );
        let _ = err.value(py).setattr("report", py_report);
        Err(err)
    } else {
        value_to_py(py, value)
    }
}

#[pyfunction]
#[pyo3(signature = (path, *, rebuild_time_index = false, rebuild_lex_index = false, rebuild_vec_index = false, vacuum = false, dry_run = false, quiet = false))]
fn doctor(
    py: Python<'_>,
    path: &str,
    rebuild_time_index: bool,
    rebuild_lex_index: bool,
    rebuild_vec_index: bool,
    vacuum: bool,
    dry_run: bool,
    quiet: bool,
) -> PyResult<PyObject> {
    let mut options = DoctorOptions::default();
    options.rebuild_time_index = rebuild_time_index;
    options.rebuild_lex_index = rebuild_lex_index;
    options.rebuild_vec_index = rebuild_vec_index;
    options.vacuum = vacuum;
    options.dry_run = dry_run;
    options.quiet = quiet;
    let report = MemvidCore::doctor(path, options).map_err(MemvidCorePy::core_err)?;
    let value = serde_json::to_value(report)
        .map_err(|err| PyValueError::new_err(format!("failed to encode doctor report: {err}")))?;
    value_to_py(py, value)
}

fn lock_owner_to_py(py: Python<'_>, owner: PyLockOwner) -> PyResult<PyObject> {
    let value = serde_json::to_value(owner)
        .map_err(|err| PyValueError::new_err(format!("failed to encode lock owner: {err}")))?;
    value_to_py(py, value)
}

#[pyfunction]
#[pyo3(signature = (path))]
fn lock_who(py: Python<'_>, path: &str) -> PyResult<Option<PyObject>> {
    let owner = lock_current_owner(Path::new(path)).map_err(MemvidCorePy::core_err)?;
    match owner {
        Some(owner) => Ok(Some(lock_owner_to_py(py, owner)?)),
        None => Ok(None),
    }
}

#[pyfunction]
#[pyo3(signature = (path))]
fn lock_nudge(path: &str) -> PyResult<bool> {
    let owner = lock_current_owner(Path::new(path)).map_err(MemvidCorePy::core_err)?;
    let Some(owner) = owner else {
        return Ok(false);
    };
    let Some(pid) = owner.pid else {
        return Ok(false);
    };
    #[cfg(unix)]
    unsafe {
        let result = libc::kill(pid as libc::pid_t, libc::SIGUSR1);
        if result == 0 {
            Ok(true)
        } else {
            Err(PyIOError::new_err(
                std::io::Error::last_os_error().to_string(),
            ))
        }
    }
    #[cfg(not(unix))]
    {
        Err(PyRuntimeError::new_err(
            "lock_nudge is only supported on Unix platforms",
        ))
    }
}

// ============================================================================
// Encryption capsules (.mv2e)
// ============================================================================

/// Lock (encrypt) an `.mv2` memory file into a `.mv2e` capsule.
///
/// Args:
///     path (str): Path to the input `.mv2` file
///     password (str): Encryption password (required)
///     output (str | None): Optional output path (default: `<path>.mv2e`)
///     force (bool): Overwrite output file if it exists
///
/// Returns:
///     str: Output path of the encrypted capsule
#[cfg(feature = "encryption")]
#[pyfunction(name = "lock")]
#[pyo3(signature = (path, *, password, output = None, force = false))]
fn lock_capsule(
    py: Python<'_>,
    path: &str,
    password: &str,
    output: Option<&str>,
    force: bool,
) -> PyResult<String> {
    if password.trim().is_empty() {
        return Err(build_exception(
            py,
            py.get_type::<EncryptionError>(),
            "MV016",
            "Password cannot be empty".to_string(),
        ));
    }

    let input = PathBuf::from(path);
    let output_path = output
        .map(PathBuf::from)
        .unwrap_or_else(|| input.with_extension("mv2e"));

    if output_path.exists() {
        if !force {
            return Err(build_exception(
                py,
                py.get_type::<EncryptionError>(),
                "MV016",
                format!(
                    "Output file already exists: {} (use force=True to overwrite)",
                    output_path.display()
                ),
            ));
        }
        let _ = fs::remove_file(&output_path);
    }

    let mut password_bytes = password.as_bytes().to_vec();
    let result =
        memvid_core::encryption::lock_file(&input, Some(output_path.as_path()), &password_bytes)
            .map_err(|err| map_capsule_error(py, err))?;
    password_bytes.fill(0);

    Ok(result.display().to_string())
}

/// Unlock (decrypt) an `.mv2e` capsule into an `.mv2` memory file.
///
/// Args:
///     path (str): Path to the input `.mv2e` capsule
///     password (str): Decryption password (required)
///     output (str | None): Optional output path (default: `<path>.mv2`)
///     force (bool): Overwrite output file if it exists
///
/// Returns:
///     str: Output path of the decrypted memory file
#[cfg(feature = "encryption")]
#[pyfunction(name = "unlock")]
#[pyo3(signature = (path, *, password, output = None, force = false))]
fn unlock_capsule(
    py: Python<'_>,
    path: &str,
    password: &str,
    output: Option<&str>,
    force: bool,
) -> PyResult<String> {
    if password.trim().is_empty() {
        return Err(build_exception(
            py,
            py.get_type::<EncryptionError>(),
            "MV016",
            "Password cannot be empty".to_string(),
        ));
    }

    let input = PathBuf::from(path);
    let output_path = output
        .map(PathBuf::from)
        .unwrap_or_else(|| input.with_extension("mv2"));

    if output_path.exists() {
        if !force {
            return Err(build_exception(
                py,
                py.get_type::<EncryptionError>(),
                "MV016",
                format!(
                    "Output file already exists: {} (use force=True to overwrite)",
                    output_path.display()
                ),
            ));
        }
        let _ = fs::remove_file(&output_path);
    }

    let mut password_bytes = password.as_bytes().to_vec();
    let result =
        memvid_core::encryption::unlock_file(&input, Some(output_path.as_path()), &password_bytes)
            .map_err(|err| map_capsule_error(py, err))?;
    password_bytes.fill(0);

    Ok(result.display().to_string())
}

// ============================================================================
// Local model helpers (CLIP / NER)
// ============================================================================

#[cfg(feature = "clip")]
#[pyclass(name = "ClipModel", module = "memvid_sdk", unsendable)]
struct ClipModelPy {
    inner: memvid_core::ClipModel,
}

#[cfg(feature = "clip")]
#[pymethods]
impl ClipModelPy {
    #[new]
    fn new() -> PyResult<Self> {
        let inner = memvid_core::ClipModel::default_model().map_err(MemvidCorePy::core_err)?;
        Ok(Self { inner })
    }

    fn model(&self) -> String {
        self.inner.model_info().name.to_string()
    }

    fn dims(&self) -> u32 {
        self.inner.dims()
    }

    fn embed_text(&self, text: &str) -> PyResult<Vec<f32>> {
        self.inner.encode_text(text).map_err(MemvidCorePy::core_err)
    }

    fn embed_image(&self, image_path: &str) -> PyResult<Vec<f32>> {
        self.inner
            .encode_image_file(Path::new(image_path))
            .map_err(MemvidCorePy::core_err)
    }

    fn embed_images(&self, image_paths: Vec<String>) -> PyResult<Vec<Vec<f32>>> {
        use memvid_core::ClipEmbeddingProvider;

        let buffers: Vec<PathBuf> = image_paths.iter().map(PathBuf::from).collect();
        let refs: Vec<&Path> = buffers.iter().map(|p| p.as_path()).collect();
        self.inner
            .embed_image_batch(&refs)
            .map_err(MemvidCorePy::core_err)
    }
}

#[cfg(feature = "logic_mesh")]
#[pyclass(name = "NerModel", module = "memvid_sdk", unsendable)]
struct NerModelPy {
    inner: Mutex<Option<memvid_core::NerModel>>,
    models_dir: PathBuf,
}

#[cfg(feature = "logic_mesh")]
#[pymethods]
impl NerModelPy {
    #[new]
    fn new() -> PyResult<Self> {
        let models_dir = resolve_models_dir().map_err(MemvidCorePy::core_err)?;
        Ok(Self {
            inner: Mutex::new(None),
            models_dir,
        })
    }

    fn model(&self) -> String {
        memvid_core::NER_MODEL_NAME.to_string()
    }

    fn entity_types(&self) -> Vec<String> {
        vec![
            "PERSON".to_string(),
            "ORG".to_string(),
            "LOCATION".to_string(),
            "MISC".to_string(),
        ]
    }

    #[pyo3(signature = (text, *, min_confidence = 0.0))]
    fn extract(&self, py: Python<'_>, text: &str, min_confidence: f32) -> PyResult<PyObject> {
        if text.trim().is_empty() {
            return Ok(PyList::empty(py).into());
        }

        let mut guard = self.inner.lock().map_err(|_| {
            MemvidCorePy::core_err(MemvidCoreError::NerModelNotAvailable {
                reason: "NER model cache poisoned".into(),
            })
        })?;

        if guard.is_none() {
            let model_path = memvid_core::ner_model_path(&self.models_dir);
            let tokenizer_path = memvid_core::ner_tokenizer_path(&self.models_dir);

            if !model_path.exists() || !tokenizer_path.exists() {
                return Err(MemvidCorePy::core_err(
                    MemvidCoreError::NerModelNotAvailable {
                        reason: format!(
                            "NER model files not found. Expected:\n- {}\n- {}",
                            model_path.display(),
                            tokenizer_path.display()
                        )
                        .into(),
                    },
                ));
            }

            let model = memvid_core::NerModel::load(model_path, tokenizer_path, Some(0.0))
                .map_err(MemvidCorePy::core_err)?;
            *guard = Some(model);
        }

        let extracted = guard
            .as_mut()
            .expect("loaded above")
            .extract(text)
            .map_err(MemvidCorePy::core_err)?;

        let list = PyList::empty(py);
        for entity in extracted {
            if entity.confidence < min_confidence {
                continue;
            }

            let upper = entity.entity_type.to_ascii_uppercase();
            let kind = match upper.as_str() {
                "PER" | "PERSON" => "PERSON",
                "ORG" => "ORG",
                "LOC" | "LOCATION" => "LOCATION",
                "MISC" => "MISC",
                other => other,
            };

            let dict = PyDict::new(py);
            dict.set_item("name", entity.text)?;
            dict.set_item("type", kind)?;
            dict.set_item("confidence", entity.confidence)?;
            dict.set_item("byte_start", entity.byte_start)?;
            dict.set_item("byte_end", entity.byte_end)?;
            list.append(dict)?;
        }
        Ok(list.into())
    }
}

/// Mask PII (Personally Identifiable Information) in text.
///
/// Detects and replaces common PII patterns with placeholder tokens:
/// - Email addresses → [EMAIL]
/// - US Social Security Numbers → [SSN]
/// - Phone numbers → [PHONE]
/// - Credit card numbers → [CREDIT_CARD]
/// - IPv4 addresses → [IP_ADDRESS]
/// - API keys/tokens → [API_KEY]
///
/// Args:
///     text (str): The text to mask
///
/// Returns:
///     str: Text with PII replaced by placeholder tokens
///
/// Example:
///     >>> from memvid_sdk._lib import mask_pii
///     >>> text = "Contact john@example.com at 555-123-4567"
///     >>> masked = mask_pii(text)
///     >>> print(masked)
///     "Contact [EMAIL] at [PHONE]"
#[pyfunction]
fn mask_pii(text: &str) -> String {
    memvid_core::pii::mask_pii(text)
}

#[pyfunction]
fn version_info(py: Python<'_>) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("native_version", env!("CARGO_PKG_VERSION"))?;
    dict.set_item("memvid_core_version", memvid_core::MEMVID_CORE_VERSION)?;

    let features = PyDict::new(py);
    features.set_item("lex", true)?;
    features.set_item("vec", true)?;
    features.set_item("parallel_segments", cfg!(feature = "parallel_segments"))?;
    features.set_item("temporal_track", cfg!(feature = "temporal_track"))?;
    features.set_item("clip", cfg!(feature = "clip"))?;
    features.set_item("logic_mesh", cfg!(feature = "logic_mesh"))?;
    features.set_item("encryption", cfg!(feature = "encryption"))?;
    features.set_item("whisper", false)?;
    dict.set_item("features", features)?;

    Ok(dict.into())
}

/// Parse an XLSX file using the Rust structured extraction pipeline.
///
/// Returns structured table data with header-value pairing, merged cell
/// propagation, and number format detection.
///
/// Args:
///     file_path: Path to the XLSX file
///     max_chars: Target chunk size in characters (default: 1200)
///     max_chunks: Maximum chunks to produce (default: 500)
///
/// Returns:
///     dict with keys: text, chunks, tables, diagnostics, timing_ms
#[pyfunction]
#[pyo3(signature = (file_path, *, max_chars = 1200, max_chunks = 500))]
fn parse_xlsx_structured(
    py: Python<'_>,
    file_path: &str,
    max_chars: usize,
    max_chunks: usize,
) -> PyResult<PyObject> {
    use memvid_core::{XlsxChunkingOptions, XlsxReader};

    let bytes = std::fs::read(file_path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to read file: {e}")))?;

    let opts = XlsxChunkingOptions {
        max_chars,
        max_chunks,
    };

    let start = std::time::Instant::now();
    let result = XlsxReader::extract_structured_with_options(&bytes, opts)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("XLSX extraction failed: {e}")))?;
    let elapsed_ms = start.elapsed().as_millis() as u64;

    let dict = PyDict::new(py);
    dict.set_item("text", &result.text)?;
    dict.set_item("timing_ms", elapsed_ms)?;

    // Chunks
    let chunks = PyList::empty(py);
    for c in &result.chunks.chunks {
        let chunk_dict = PyDict::new(py);
        chunk_dict.set_item("text", &c.text)?;
        chunk_dict.set_item("chunk_type", format!("{:?}", c.chunk_type))?;
        chunk_dict.set_item("index", c.index)?;
        chunk_dict.set_item("element_id", &c.element_id)?;
        chunk_dict.set_item("context", &c.context)?;
        chunks.append(chunk_dict)?;
    }
    dict.set_item("chunks", chunks)?;

    // Tables
    let tables = PyList::empty(py);
    for t in &result.tables {
        let table_dict = PyDict::new(py);
        table_dict.set_item("name", &t.name)?;
        table_dict.set_item("sheet_name", &t.sheet_name)?;
        table_dict.set_item("headers", &t.headers)?;
        table_dict.set_item("header_row", t.header_row)?;
        table_dict.set_item("first_data_row", t.first_data_row)?;
        table_dict.set_item("last_data_row", t.last_data_row)?;
        table_dict.set_item("first_col", t.first_col)?;
        table_dict.set_item("last_col", t.last_col)?;
        table_dict.set_item("confidence", t.confidence)?;
        let col_types: Vec<String> = t.column_types.iter().map(|ct| format!("{ct:?}")).collect();
        table_dict.set_item("column_types", col_types)?;
        tables.append(table_dict)?;
    }
    dict.set_item("tables", tables)?;

    // Diagnostics
    let diag = PyDict::new(py);
    diag.set_item("warnings", &result.diagnostics.warnings)?;
    diag.set_item("tables_processed", result.chunks.tables_processed)?;
    diag.set_item("tables_split", result.chunks.tables_split)?;
    dict.set_item("diagnostics", diag)?;

    Ok(dict.into())
}

#[pymodule]
fn _lib(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    #[cfg(feature = "parallel_segments")]
    {
        m.add_class::<PyBuildOpts>()?;
    }
    m.add_class::<MemvidCorePy>()?;
    #[cfg(feature = "clip")]
    {
        m.add_class::<ClipModelPy>()?;
    }
    #[cfg(feature = "logic_mesh")]
    {
        m.add_class::<NerModelPy>()?;
    }
    m.add("MemvidError", py.get_type::<MemvidError>())?;
    m.add(
        "CapacityExceededError",
        py.get_type::<CapacityExceededError>(),
    )?;
    m.add("LockedError", py.get_type::<LockedError>())?;
    m.add("TicketInvalidError", py.get_type::<TicketInvalidError>())?;
    m.add("TicketReplayError", py.get_type::<TicketReplayError>())?;
    m.add(
        "LexIndexDisabledError",
        py.get_type::<LexIndexDisabledError>(),
    )?;
    m.add(
        "TimeIndexMissingError",
        py.get_type::<TimeIndexMissingError>(),
    )?;
    m.add("VerifyFailedError", py.get_type::<VerifyFailedError>())?;
    m.add("ApiKeyRequiredError", py.get_type::<ApiKeyRequiredError>())?;
    m.add(
        "MemoryAlreadyBoundError",
        py.get_type::<MemoryAlreadyBoundError>(),
    )?;
    m.add("FrameNotFoundError", py.get_type::<FrameNotFoundError>())?;
    m.add(
        "VecIndexDisabledError",
        py.get_type::<VecIndexDisabledError>(),
    )?;
    m.add("CorruptFileError", py.get_type::<CorruptFileError>())?;
    m.add("FileNotFoundError", py.get_type::<FileNotFoundError>())?;
    m.add(
        "VecDimensionMismatchError",
        py.get_type::<VecDimensionMismatchError>(),
    )?;
    m.add(
        "EmbeddingFailedError",
        py.get_type::<EmbeddingFailedError>(),
    )?;
    m.add("EncryptionError", py.get_type::<EncryptionError>())?;
    m.add(
        "ClipIndexDisabledError",
        py.get_type::<ClipIndexDisabledError>(),
    )?;
    m.add(
        "NerModelNotAvailableError",
        py.get_type::<NerModelNotAvailableError>(),
    )?;
    m.add_function(wrap_pyfunction!(open, m)?)?;
    m.add_function(wrap_pyfunction!(create, m)?)?;
    m.add_function(wrap_pyfunction!(put, m)?)?;
    m.add_function(wrap_pyfunction!(find, m)?)?;
    m.add_function(wrap_pyfunction!(ask, m)?)?;
    m.add_function(wrap_pyfunction!(audit, m)?)?;
    m.add_function(wrap_pyfunction!(verify, m)?)?;
    m.add_function(wrap_pyfunction!(doctor, m)?)?;
    m.add_function(wrap_pyfunction!(lock_who, m)?)?;
    m.add_function(wrap_pyfunction!(lock_nudge, m)?)?;
    #[cfg(feature = "encryption")]
    {
        m.add_function(wrap_pyfunction!(lock_capsule, m)?)?;
        m.add_function(wrap_pyfunction!(unlock_capsule, m)?)?;
    }
    m.add_function(wrap_pyfunction!(mask_pii, m)?)?;
    m.add_function(wrap_pyfunction!(version_info, m)?)?;
    m.add_function(wrap_pyfunction!(parse_xlsx_structured, m)?)?;
    Ok(())
}
