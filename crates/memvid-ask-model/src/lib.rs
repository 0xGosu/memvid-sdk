use std::fmt;
use std::io::{IsTerminal, Write, stderr};
use std::path::PathBuf;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::thread;
use std::time::Duration;

use memvid_core::types::SearchHit;

#[derive(Debug, Clone)]
pub struct ModelAnswer {
    pub requested: String,
    pub model: String,
    pub answer: String,
}

#[derive(Debug, Clone)]
pub struct ModelInference {
    pub answer: ModelAnswer,
    pub context_body: String,
    pub context_fragments: Vec<ModelContextFragment>,
    pub usage: Option<TokenUsage>,
    pub grounding: Option<GroundingResult>,
    /// True if this result came from the cache
    pub cached: bool,
}

/// Token usage and cost information from LLM inference
#[derive(Debug, Clone, Default)]
pub struct TokenUsage {
    /// Input/prompt tokens
    pub input_tokens: u32,
    /// Output/completion tokens
    pub output_tokens: u32,
    /// Total tokens (input + output)
    pub total_tokens: u32,
    /// Estimated cost in USD (based on model pricing)
    pub cost_usd: f64,
}

/// Cache for LLM answers to avoid redundant API calls
/// Uses Blake3 hash of (query + context) as the key
pub mod cache {
    use std::collections::HashMap;
    use std::sync::Mutex;

    /// Cached answer entry
    #[derive(Debug, Clone)]
    pub struct CacheEntry {
        pub answer: String,
        pub model: String,
        pub input_tokens: u32,
        pub output_tokens: u32,
        pub cost_usd: f64,
        pub grounding_score: f32,
        pub created_at: std::time::SystemTime,
    }

    /// In-memory LRU cache for answers
    /// Thread-safe with a simple mutex
    pub struct AnswerCache {
        entries: Mutex<HashMap<[u8; 32], CacheEntry>>,
        max_size: usize,
        hits: std::sync::atomic::AtomicU64,
        misses: std::sync::atomic::AtomicU64,
    }

    impl AnswerCache {
        /// Create a new cache with the specified maximum size
        pub fn new(max_size: usize) -> Self {
            Self {
                entries: Mutex::new(HashMap::new()),
                max_size,
                hits: std::sync::atomic::AtomicU64::new(0),
                misses: std::sync::atomic::AtomicU64::new(0),
            }
        }

        /// Generate a cache key from query and context
        pub fn make_key(query: &str, context: &str, model: &str) -> [u8; 32] {
            use std::io::Write;
            let mut hasher = blake3::Hasher::new();
            let _ = write!(hasher, "{}|{}|{}", model, query, context);
            *hasher.finalize().as_bytes()
        }

        /// Look up an entry in the cache
        pub fn get(&self, key: &[u8; 32]) -> Option<CacheEntry> {
            let entries = self.entries.lock().ok()?;
            let result = entries.get(key).cloned();
            if result.is_some() {
                self.hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            } else {
                self.misses
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            result
        }

        /// Insert an entry into the cache
        pub fn insert(&self, key: [u8; 32], entry: CacheEntry) {
            if let Ok(mut entries) = self.entries.lock() {
                // Simple LRU: if at capacity, remove oldest entry
                if entries.len() >= self.max_size {
                    let oldest_key = entries
                        .iter()
                        .min_by_key(|(_, v)| v.created_at)
                        .map(|(k, _)| *k);
                    if let Some(k) = oldest_key {
                        entries.remove(&k);
                    }
                }
                entries.insert(key, entry);
            }
        }

        /// Clear the cache
        pub fn clear(&self) {
            if let Ok(mut entries) = self.entries.lock() {
                entries.clear();
            }
        }

        /// Get cache statistics
        pub fn stats(&self) -> CacheStats {
            let entries = self.entries.lock().map(|e| e.len()).unwrap_or(0);
            let hits = self.hits.load(std::sync::atomic::Ordering::Relaxed);
            let misses = self.misses.load(std::sync::atomic::Ordering::Relaxed);
            CacheStats {
                entries,
                hits,
                misses,
                hit_rate: if hits + misses > 0 {
                    hits as f64 / (hits + misses) as f64
                } else {
                    0.0
                },
            }
        }

        /// Estimated cost savings from cache hits
        pub fn estimated_savings(&self) -> f64 {
            if let Ok(entries) = self.entries.lock() {
                let hits = self.hits.load(std::sync::atomic::Ordering::Relaxed);
                let avg_cost =
                    entries.values().map(|e| e.cost_usd).sum::<f64>() / entries.len().max(1) as f64;
                hits as f64 * avg_cost
            } else {
                0.0
            }
        }
    }

    impl Default for AnswerCache {
        fn default() -> Self {
            Self::new(100) // Default to 100 entries
        }
    }

    #[derive(Debug, Clone)]
    pub struct CacheStats {
        pub entries: usize,
        pub hits: u64,
        pub misses: u64,
        pub hit_rate: f64,
    }

    // Global cache instance
    lazy_static::lazy_static! {
        pub static ref GLOBAL_CACHE: AnswerCache = AnswerCache::new(500);
    }

    /// Check cache and return cached result if available
    pub fn check_cache(query: &str, context: &str, model: &str) -> Option<CacheEntry> {
        let key = AnswerCache::make_key(query, context, model);
        GLOBAL_CACHE.get(&key)
    }

    /// Store result in cache
    pub fn store_in_cache(query: &str, context: &str, model: &str, entry: CacheEntry) {
        let key = AnswerCache::make_key(query, context, model);
        GLOBAL_CACHE.insert(key, entry);
    }

    /// Get global cache statistics
    pub fn global_stats() -> CacheStats {
        GLOBAL_CACHE.stats()
    }

    /// Clear the global cache
    pub fn clear_global_cache() {
        GLOBAL_CACHE.clear();
    }
}

/// Result of grounding/hallucination verification
#[derive(Debug, Clone, Default)]
pub struct GroundingResult {
    /// Overall grounding score (0.0 to 1.0)
    /// Higher = more grounded in context, less likely to hallucinate
    pub score: f32,
    /// Number of sentences in the answer
    pub sentence_count: usize,
    /// Number of sentences with at least one grounded claim
    pub grounded_sentences: usize,
    /// Individual sentence scores
    pub sentence_scores: Vec<f32>,
    /// Warning flag: true if potential hallucination detected
    pub has_warning: bool,
    /// Explanation of the warning (if any)
    pub warning_reason: Option<String>,
}

impl GroundingResult {
    /// Returns a human-readable grade based on grounding score
    pub fn grade(&self) -> &'static str {
        match self.score {
            s if s >= 0.8 => "A",
            s if s >= 0.6 => "B",
            s if s >= 0.4 => "C",
            s if s >= 0.2 => "D",
            _ => "F",
        }
    }

    /// Returns a label like "HIGH", "MEDIUM", "LOW"
    pub fn label(&self) -> &'static str {
        match self.score {
            s if s >= 0.7 => "HIGH",
            s if s >= 0.4 => "MEDIUM",
            _ => "LOW",
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelContextFragment {
    pub rank: usize,
    pub uri: String,
    pub title: Option<String>,
    pub score: Option<f32>,
    pub matches: usize,
    pub frame_id: u64,
    pub range: (usize, usize),
    pub chunk_range: Option<(usize, usize)>,
    pub text: String,
    pub kind: ModelContextFragmentKind,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ModelContextFragmentKind {
    Full,
    Summary,
}

impl ModelContextFragment {
    fn from_record(record: context::ContextRecord) -> Self {
        let kind = match record.mode {
            context::ContextMode::Full => ModelContextFragmentKind::Full,
            context::ContextMode::Summary => ModelContextFragmentKind::Summary,
        };
        Self {
            rank: record.rank,
            uri: record.uri,
            title: record.title,
            score: record.score,
            matches: record.matches,
            frame_id: record.frame_id,
            range: record.range,
            chunk_range: record.chunk_range,
            text: record.text,
            kind,
        }
    }
}

#[derive(Debug)]
pub enum ModelRunError {
    UnsupportedModel(String),
    AssetsMissing {
        model: String,
        missing: Vec<PathBuf>,
    },
    Runtime(anyhow::Error),
}

impl fmt::Display for ModelRunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedModel(model) => write!(f, "unsupported model '{model}'"),
            Self::AssetsMissing { model, missing } => {
                let paths: Vec<_> = missing
                    .iter()
                    .map(|path| path.display().to_string())
                    .collect();
                write!(
                    f,
                    "model '{model}' missing required assets: {}",
                    paths.join(", ")
                )
            }
            Self::Runtime(err) => write!(f, "model runtime error: {err}"),
        }
    }
}

impl std::error::Error for ModelRunError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Runtime(err) => Some(err.root_cause()),
            _ => None,
        }
    }
}

const LOCAL_CONTEXT_CHARS: usize = 32_768;
const MAX_QUESTION_CHARS: usize = 512;
const LOCAL_MAX_OUTPUT_TOKENS: usize = 256;
const REMOTE_MAX_OUTPUT_TOKENS: usize = 768;
const SYSTEM_PROMPT: &str = r#"You are a precise, intelligent assistant that answers questions using ONLY the provided retrieval context.

## Core Principles
1. GROUND EVERY CLAIM in the context. If asked for a number, quote it exactly.
2. NEVER hallucinate or use external knowledge. If unsure, say "Based on the context..."
3. BE CONCISE but complete. One clear answer is better than verbose hedging.

## CRITICAL: Correction Handling (MANDATORY)
**STOP AND READ THIS FIRST.** Before answering ANY question:
1. Scan ALL sources for "Correction:" in the title or "mv2://correction/" in the URI
2. If ANY correction exists that relates to the question, USE THAT ANSWER ONLY
3. IGNORE ALL OTHER SOURCES that contradict the correction - they are OUTDATED
4. If multiple corrections exist, use the FIRST one listed (most recent)

**VIOLATION OF THIS RULE IS A CRITICAL ERROR.** Example:
- Question: "Where does Ben live?"
- Correction says: "Ben lives in Kenya"
- Other doc says: "Ben lives in Germany"
- CORRECT ANSWER: "Kenya" (from correction)
- WRONG ANSWER: "Germany" (ignores correction = FAIL)

## Answer Strategy
- For NUMERIC questions: Extract the exact value. If multiple values exist, identify which is most relevant (usually the most recent or most specific match).
- For YES/NO questions: Answer directly, then briefly explain why.
- For COMPARISON questions: Present both sides with their values.
- For LIST questions: Use bullet points or numbered lists.
- For TEMPORAL questions: Note that later timestamps = more current information. State WHEN data is from.
- For CALCULATION questions: Show your work step-by-step.
- For ANALYTICAL/PATTERN questions (e.g., "reverted", "changed back", "any differences over time"):
  1. TRACE each attribute's value across ALL time periods in the context
  2. Look for A→B→A patterns where a value changes then returns to its original state
  3. Terms like "consolidated", "same as", "unified", or "aligned" often indicate returning to a prior arrangement
  4. Compare explicit state changes: if Period 1 says "X was same as Y", Period 2 says "X different from Y", and Period 3 says "X consolidated/same as Y again", that IS a reversion
  5. Create a timeline table if helpful to track changes

## Handling Ambiguity
- If the question is ambiguous, interpret it reasonably and state your interpretation.
- If multiple valid answers exist, present the most likely one first, then mention alternatives.
- If context is insufficient, say what IS known, then note what's missing.

## Quality Standards
- PREFER specific answers over vague ones ("$1,234.56" not "around a thousand")
- CITE context when helpful ("[Source: ...]")
- CORRECT obvious typos in your interpretation (e.g., "teh" → "the")
- For percentages/ratios, include the actual numbers when available"#;
const TINYLLAMA_LABEL: &str = "tinyllama-1.1b";
const LOCAL_PROMPT_MARGIN_CHARS: usize = 2_048;
const REMOTE_PROMPT_MARGIN_CHARS: usize = 4_096;
const OLLAMA_PROMPT_CHARS: usize = 110_000;
const OPENAI_PROMPT_CHARS: usize = 240_000;
const NVIDIA_PROMPT_CHARS: usize = 240_000;
const GEMINI_PROMPT_CHARS: usize = 320_000;
const CLAUDE_PROMPT_CHARS: usize = 360_000;
const XAI_PROMPT_CHARS: usize = 260_000; // Grok models: ~131K tokens
const GROQ_PROMPT_CHARS: usize = 260_000; // LLaMA 3.3 70B: 128K tokens
const MISTRAL_PROMPT_CHARS: usize = 260_000; // Mistral Large: 128K tokens

#[derive(Debug, Clone, Copy)]
struct ModelContextBudget {
    total_chars: usize,
    reserved_chars: usize,
}

impl ModelContextBudget {
    const fn new(total_chars: usize, reserved_chars: usize) -> Self {
        Self {
            total_chars,
            reserved_chars,
        }
    }

    fn context_chars(&self) -> usize {
        self.total_chars.saturating_sub(self.reserved_chars)
    }

    fn question_limit(&self) -> usize {
        MAX_QUESTION_CHARS
            .min(self.reserved_chars.max(1))
            .min(self.total_chars.max(1))
    }

    fn apply_override(self, override_context_chars: usize) -> Self {
        let total = override_context_chars.saturating_add(self.reserved_chars);
        Self {
            total_chars: total.max(self.reserved_chars + 1),
            reserved_chars: self.reserved_chars,
        }
    }

    fn prompt_ceiling(&self) -> usize {
        self.total_chars
    }
}

pub struct PromptParts {
    completion_prompt: String,
    user_message: String,
    max_output_tokens: usize,
}

impl PromptParts {
    pub fn completion_prompt(&self) -> &str {
        &self.completion_prompt
    }

    pub fn user_message(&self) -> &str {
        &self.user_message
    }

    pub fn max_output_tokens(&self) -> usize {
        self.max_output_tokens
    }
}

/// Normalize and enhance a question for optimal LLM interpretation.
///
/// This function:
/// 1. Ensures questions end with `?` for consistent interpretation
/// 2. Fixes common typos and abbreviations
/// 3. Clarifies ambiguous phrasing
/// 4. Expands common abbreviations for better matching
fn normalize_question(question: &str) -> String {
    let trimmed = question.trim();
    if trimmed.is_empty() {
        return trimmed.to_string();
    }

    // Step 1: Fix common typos and normalize spacing
    let mut normalized = fix_common_typos(trimmed);

    // Step 2: Expand common abbreviations for clarity
    normalized = expand_abbreviations(&normalized);

    // Step 3: Ensure proper punctuation
    normalized = ensure_question_punctuation(&normalized);

    normalized
}

/// Fix common typos that affect query interpretation
fn fix_common_typos(text: &str) -> String {
    let mut result = text.to_string();

    // Common typo patterns (case-insensitive replacements)
    let typos: &[(&str, &str)] = &[
        // Common misspellings
        ("teh ", "the "),
        ("hte ", "the "),
        ("adn ", "and "),
        ("taht ", "that "),
        ("wiht ", "with "),
        ("thier ", "their "),
        ("recieve", "receive"),
        ("occured", "occurred"),
        ("seperate", "separate"),
        // Question word typos
        ("waht ", "what "),
        ("hwat ", "what "),
        ("wehn ", "when "),
        ("whre ", "where "),
        ("wher ", "where "),
        ("howm ", "how "),
        ("hwo ", "who "),
        // Common finger slips
        ("amoutn", "amount"),
        ("totla", "total"),
        ("nubmer", "number"),
        ("vlaue", "value"),
        ("prive", "price"),
        ("proce", "price"),
        ("revneue", "revenue"),
        ("reveneu", "revenue"),
    ];

    for (typo, correction) in typos {
        // Case-insensitive replacement
        let lower = result.to_lowercase();
        if lower.contains(*typo) {
            let start = lower.find(*typo).unwrap();
            let end = start + typo.len();
            result = format!("{}{}{}", &result[..start], correction, &result[end..]);
        }
    }

    // Normalize multiple spaces to single
    let mut prev_space = false;
    result = result
        .chars()
        .filter(|c| {
            if c.is_whitespace() {
                if prev_space {
                    false
                } else {
                    prev_space = true;
                    true
                }
            } else {
                prev_space = false;
                true
            }
        })
        .collect();

    result
}

/// Generate optimized search keywords from a question using LLM
/// Returns the original question plus extracted search terms for better retrieval
pub fn generate_search_query(
    question: &str,
    model: &str,
    api_key: &str,
) -> Result<String, ModelRunError> {
    // For lexical search, we need to be careful - adding too many terms can hurt
    // because Tantivy uses AND logic. We need a very short, focused query.
    // IMPORTANT: Keep abbreviations as-is since documents often use the abbreviation form.
    let prompt = format!(
        r#"Extract 2 key search terms from this question.
KEEP abbreviations exactly as written (QPS, API, SDK, etc.) - don't expand them.
Output only the main topic and one key term.

Question: {}

Examples:
- "What is the QPS for memvid?" → "memvid QPS"
- "How many queries per second?" → "QPS throughput"
- "What's the API rate limit?" → "API rate"
- "How much does it cost?" → "cost pricing"

Output exactly 2 words, nothing else."#,
        question
    );

    // Use a fast model for keyword extraction
    // The model passed in is already the fast variant (gpt-4o-mini or claude-haiku)
    let extraction_model =
        if model.starts_with("gpt") || model.starts_with("o1") || model.contains("openai") {
            "gpt-4o-mini"
        } else if model.starts_with("claude") || model.contains("anthropic") {
            "claude-haiku-4-5"
        } else if model.contains("llama") || model.contains("groq") || model.contains("mixtral") {
            "llama-3.1-8b-instant" // Fast Groq model for keyword extraction
        } else if model.contains("grok") || model.contains("xai") {
            "grok-4-fast"
        } else if model.contains("mistral") {
            "mistral-small-latest" // Fast Mistral model
        } else {
            // For other models, just return the original question
            return Ok(question.to_string());
        };

    // Make a quick API call for query rewriting
    let rewritten = call_llm_for_keywords(&prompt, extraction_model, api_key)?;

    // If we got a good rewritten query, use it; otherwise fall back to original
    let rewritten = rewritten.trim();
    if rewritten.is_empty() || rewritten.len() > 100 {
        // LLM returned empty or too long - use original
        Ok(question.to_string())
    } else {
        // Use the short, focused rewritten query for search
        Ok(rewritten.to_string())
    }
}

/// Quick LLM call for keyword extraction (lightweight, fast model)
fn call_llm_for_keywords(
    prompt: &str,
    model: &str,
    api_key: &str,
) -> Result<String, ModelRunError> {
    use reqwest::blocking::Client;
    use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};

    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(10)) // Fast timeout for keyword extraction
        .build()
        .map_err(|e| ModelRunError::Runtime(anyhow::anyhow!("HTTP client error: {e}")))?;

    // Determine API endpoint and headers based on model
    let (url, is_anthropic) = if model.starts_with("gpt") || model.starts_with("o1") {
        ("https://api.openai.com/v1/chat/completions", false)
    } else if model.starts_with("claude") {
        ("https://api.anthropic.com/v1/messages", true)
    } else if model.contains("llama") || model.contains("mixtral") {
        ("https://api.groq.com/openai/v1/chat/completions", false)
    } else if model.contains("grok") {
        ("https://api.x.ai/v1/chat/completions", false)
    } else if model.contains("mistral") {
        ("https://api.mistral.ai/v1/chat/completions", false)
    } else {
        return Err(ModelRunError::UnsupportedModel(model.to_string()));
    };

    let response = if is_anthropic {
        let mut headers = HeaderMap::new();
        headers.insert(
            reqwest::header::HeaderName::from_static("x-api-key"),
            HeaderValue::from_str(api_key)
                .map_err(|e| ModelRunError::Runtime(anyhow::anyhow!("Invalid API key: {e}")))?,
        );
        headers.insert(
            reqwest::header::HeaderName::from_static("anthropic-version"),
            HeaderValue::from_static("2023-06-01"),
        );
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        client
            .post(url)
            .headers(headers)
            .json(&serde_json::json!({
                "model": model,
                "max_tokens": 100,
                "messages": [{"role": "user", "content": prompt}]
            }))
            .send()
    } else {
        // OpenAI-compatible API (OpenAI, Groq, XAI, Mistral)
        let mut headers = HeaderMap::new();
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", api_key))
                .map_err(|e| ModelRunError::Runtime(anyhow::anyhow!("Invalid API key: {e}")))?,
        );
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        client
            .post(url)
            .headers(headers)
            .json(&serde_json::json!({
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 100,
                "temperature": 0.0
            }))
            .send()
    };

    match response {
        Ok(resp) => {
            let json: serde_json::Value = resp
                .json()
                .map_err(|e| ModelRunError::Runtime(anyhow::anyhow!("JSON parse error: {e}")))?;

            // Extract text from response
            let text = if model.starts_with("claude") {
                json["content"][0]["text"].as_str().unwrap_or("")
            } else {
                json["choices"][0]["message"]["content"]
                    .as_str()
                    .unwrap_or("")
            };

            Ok(text.to_string())
        }
        Err(_) => {
            // On error, just return empty (fall back to original query)
            // This is not a fatal error - we just use the original query
            Ok(String::new())
        }
    }
}

/// Expand common abbreviations for better context matching (fallback when LLM not available)
fn expand_abbreviations(text: &str) -> String {
    // This is now a simple fallback - the main expansion happens via LLM
    text.to_string()
}

/// Ensure the question ends with proper punctuation
fn ensure_question_punctuation(text: &str) -> String {
    let trimmed = text.trim();

    // Already ends with punctuation - don't modify
    if trimmed.ends_with('?') || trimmed.ends_with('.') || trimmed.ends_with('!') {
        return trimmed.to_string();
    }

    // Check if it looks like a question (starts with question word or auxiliary verb)
    let lower = trimmed.to_lowercase();
    let question_starters = [
        "how", "what", "where", "when", "why", "which", "who", "whom", "whose", "is", "are", "was",
        "were", "will", "would", "can", "could", "should", "do", "does", "did", "have", "has",
        "had", "may", "might", "shall", "tell me", "show me", "find", "list", "give me", "explain",
    ];

    let is_question = question_starters.iter().any(|starter| {
        lower.starts_with(starter)
            && (lower.len() == starter.len()
                || !lower[starter.len()..].starts_with(|c: char| c.is_alphanumeric()))
    });

    if is_question {
        format!("{}?", trimmed)
    } else {
        trimmed.to_string()
    }
}

fn build_prompt_parts(
    question: &str,
    context: &str,
    budget: &ModelContextBudget,
    max_output_tokens: usize,
) -> PromptParts {
    let mut context_section = context.to_string();
    let normalized_question = normalize_question(question);
    let trimmed_question = trim_to(&normalized_question, budget.question_limit());

    // Detect question type for better prompting
    let question_type = detect_question_type(&trimmed_question);
    let type_hint = question_type.hint();

    let system_section = format!("### System\n{SYSTEM_PROMPT}");
    let question_section = format!("### Question\n{trimmed_question}");
    let answer_stub = "### Answer\n";

    let overhead = system_section.len() + 2 + question_section.len() + 2 + answer_stub.len();
    if budget.prompt_ceiling() > overhead {
        let max_context_len = budget
            .prompt_ceiling()
            .saturating_sub(overhead)
            .min(budget.context_chars());
        if context_section.len() > max_context_len {
            context_section = clamp_to(&context_section, max_context_len);
        }
    } else {
        context_section = String::new();
    }

    // Handle empty context gracefully
    let context_instruction = if context_section.trim().is_empty() {
        "Note: No relevant context was found. Answer based on what you know, but clearly state this limitation."
    } else {
        ""
    };

    let completion_prompt =
        format!("{system_section}\n\n{context_section}\n\n{question_section}\n\n### Answer\n");

    // Enhanced user message with type-specific guidance
    let user_message = if context_instruction.is_empty() {
        format!(
            "{context_section}\n\n---\nQuestion: {trimmed_question}\n{type_hint}\nProvide a direct, accurate answer using only the context above."
        )
    } else {
        format!("{context_instruction}\n\nQuestion: {trimmed_question}\n{type_hint}")
    };

    PromptParts {
        completion_prompt,
        user_message,
        max_output_tokens,
    }
}

/// Question type classification for better prompting
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QuestionType {
    Numeric,     // "how much", "how many", "what is the value"
    YesNo,       // "is", "are", "does", "can", "will"
    List,        // "list", "what are the", "show all"
    Comparison,  // "compare", "difference", "vs", "versus"
    Temporal,    // "when", "what date", "how long"
    Explanation, // "why", "explain", "how does"
    Factual,     // "what", "who", "where"
    Other,
}

impl QuestionType {
    fn hint(&self) -> &'static str {
        match self {
            Self::Numeric => "(Expected: a specific number or value)",
            Self::YesNo => "(Expected: yes/no with brief explanation)",
            Self::List => "(Expected: a list of items)",
            Self::Comparison => "(Expected: comparison of two or more items)",
            Self::Temporal => "(Expected: a date, time, or duration)",
            Self::Explanation => "(Expected: reasoning or explanation)",
            Self::Factual => "(Expected: a factual answer)",
            Self::Other => "",
        }
    }
}

fn detect_question_type(question: &str) -> QuestionType {
    let lower = question.to_lowercase();

    // Numeric patterns
    if lower.contains("how much")
        || lower.contains("how many")
        || lower.contains("what is the value")
        || lower.contains("what's the value")
        || lower.contains("total")
        || lower.contains("sum")
        || lower.contains("average")
        || lower.contains("percentage")
        || lower.contains("rate")
        || lower.contains("amount")
        || lower.contains("price")
        || lower.contains("cost")
        || lower.contains("revenue")
        || lower.contains("profit")
    {
        return QuestionType::Numeric;
    }

    // Yes/No patterns
    let yes_no_starters = [
        "is ", "are ", "does ", "do ", "can ", "will ", "has ", "have ", "was ", "were ",
    ];
    if yes_no_starters.iter().any(|s| lower.starts_with(s)) {
        return QuestionType::YesNo;
    }

    // List patterns
    if lower.contains("list")
        || lower.contains("show all")
        || lower.contains("what are the")
        || lower.contains("name all")
        || lower.contains("enumerate")
    {
        return QuestionType::List;
    }

    // Comparison patterns
    if lower.contains("compare")
        || lower.contains("difference between")
        || lower.contains(" vs ")
        || lower.contains("versus")
        || lower.contains("better than")
        || lower.contains("worse than")
    {
        return QuestionType::Comparison;
    }

    // Temporal patterns
    if lower.starts_with("when")
        || lower.contains("what date")
        || lower.contains("how long")
        || lower.contains("how old")
        || lower.contains("since when")
    {
        return QuestionType::Temporal;
    }

    // Explanation patterns
    if lower.starts_with("why")
        || lower.starts_with("explain")
        || lower.contains("how does")
        || lower.contains("reason for")
        || lower.contains("cause of")
    {
        return QuestionType::Explanation;
    }

    // Factual patterns
    if lower.starts_with("what") || lower.starts_with("who") || lower.starts_with("where") {
        return QuestionType::Factual;
    }

    QuestionType::Other
}

/// Post-process the LLM answer for quality
pub fn postprocess_answer(answer: &str) -> String {
    let mut result = answer.trim().to_string();

    // Remove common LLM artifacts
    let artifacts = [
        "Based on the provided context,",
        "According to the context,",
        "From the context provided,",
        "The context shows that",
        "Based on the information provided,",
    ];
    for artifact in artifacts {
        if result.starts_with(artifact) {
            result = result[artifact.len()..].trim_start().to_string();
            // Capitalize first letter
            if let Some(first) = result.chars().next() {
                result = first.to_uppercase().chain(result.chars().skip(1)).collect();
            }
        }
    }

    // Normalize whitespace
    result = result.split_whitespace().collect::<Vec<_>>().join(" ");

    // Ensure the answer doesn't start with lowercase
    if let Some(first) = result.chars().next() {
        if first.is_lowercase() && !result.starts_with("i ") {
            result = first.to_uppercase().chain(result.chars().skip(1)).collect();
        }
    }

    result
}

fn trim_to(text: &str, limit: usize) -> String {
    if text.len() <= limit {
        text.to_string()
    } else {
        let mut truncated = text[..limit].to_string();
        truncated.push_str("...");
        truncated
    }
}

fn clamp_to(text: &str, limit: usize) -> String {
    if text.len() <= limit {
        text.to_string()
    } else if limit <= 3 {
        "...".chars().take(limit).collect()
    } else {
        let mut end = limit.saturating_sub(3);
        // Find valid UTF-8 char boundary (curly quotes, emojis, etc. are multi-byte)
        while end > 0 && !text.is_char_boundary(end) {
            end -= 1;
        }
        if end == 0 {
            return "...".to_string();
        }
        let mut truncated = text[..end].to_string();
        truncated.push_str("...");
        truncated
    }
}

struct ThinkingSpinner {
    flag: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
}

impl ThinkingSpinner {
    fn start() -> Self {
        let flag = Arc::new(AtomicBool::new(true));
        let thread_flag = flag.clone();

        // Only show spinner if stderr is a TTY (interactive terminal).
        // This prevents control characters from polluting output when
        // stderr is redirected or combined with stdout (e.g., `2>&1`).
        let is_tty = stderr().is_terminal();

        let handle = thread::spawn(move || {
            if !is_tty {
                // Not a TTY, don't show spinner - just wait for stop signal
                while thread_flag.load(Ordering::Relaxed) {
                    thread::sleep(Duration::from_millis(200));
                }
                return;
            }

            let frames = [
                "Thinking    ",
                "Thinking.   ",
                "Thinking..  ",
                "Thinking... ",
                "Thinking .. ",
                "Thinking  . ",
            ];
            let mut idx = 0;
            let mut err = stderr();
            while thread_flag.load(Ordering::Relaxed) {
                let frame = frames[idx % frames.len()];
                let _ = write!(err, "\r{frame}");
                let _ = err.flush();
                idx = idx.wrapping_add(1);
                thread::sleep(Duration::from_millis(200));
            }
            let _ = write!(err, "\r             \r");
            let _ = err.flush();
        });

        Self {
            flag,
            handle: Some(handle),
        }
    }

    fn stop(&mut self) {
        if let Some(handle) = self.handle.take() {
            self.flag.store(false, Ordering::Relaxed);
            let _ = handle.join();
        }
    }
}

impl Drop for ThinkingSpinner {
    fn drop(&mut self) {
        self.stop();
    }
}

#[derive(Debug, Clone)]
enum ModelKind {
    TinyLlama,
    Ghost { pack_path: PathBuf },
    Ollama { model: String },
    OpenAi { model: String },
    Nvidia { model: String },
    Gemini { model: String },
    Claude { model: String },
    Xai { model: String },
    Groq { model: String },
    Mistral { model: String },
}

impl ModelKind {
    fn parse(raw: &str) -> Option<Self> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return None;
        }

        let (provider, explicit_model) = if let Some((p, rest)) = trimmed.split_once(':') {
            let value = rest.trim();
            let explicit = if value.is_empty() {
                None
            } else {
                Some(value.to_string())
            };
            (p.trim().to_ascii_lowercase(), explicit)
        } else {
            (trimmed.to_ascii_lowercase(), None)
        };

        match provider.as_str() {
            "tinyllama" | "tiny-llama" | "tinyllama-1.1b" => Some(Self::TinyLlama),
            "ghost" => explicit_model.map(|value| Self::Ghost {
                pack_path: PathBuf::from(value),
            }),
            "ollama" => Some(Self::Ollama {
                model: explicit_model.unwrap_or_else(|| "ollama1.5".to_string()),
            }),
            "ollama1.5" | "ollama1-5" => Some(Self::Ollama {
                model: "ollama1.5".to_string(),
            }),
            "openai" => Some(Self::OpenAi {
                model: normalize_openai_model(explicit_model),
            }),
            "nvidia" | "nv" => Some(Self::Nvidia {
                model: normalize_nvidia_model(explicit_model),
            }),
            "gemini" | "google" => Some(Self::Gemini {
                model: normalize_gemini_model(explicit_model),
            }),
            "claude" | "anthropic" => Some(Self::Claude {
                model: normalize_claude_model(explicit_model),
            }),
            "xai" | "grok" => Some(Self::Xai {
                model: normalize_xai_model(explicit_model),
            }),
            "groq" => Some(Self::Groq {
                model: normalize_groq_model(explicit_model),
            }),
            "mistral" => Some(Self::Mistral {
                model: normalize_mistral_model(explicit_model),
            }),
            // Auto-detect provider from model name prefix
            // For Ollama models with colons in the name (e.g., qwen2.5:1.5b),
            // we need to use the full original name, not just the provider prefix
            _ => Self::infer_from_model_name_full(trimmed, &provider),
        }
    }

    /// Infer the provider from a model name, using the full original name for Ollama models.
    /// This handles model names with colons like "qwen2.5:1.5b" by using the full name.
    fn infer_from_model_name_full(full_name: &str, prefix: &str) -> Option<Self> {
        let lowered = prefix.to_ascii_lowercase();

        // Gemini models: gemini-*, models/gemini-*
        if lowered.starts_with("gemini") || lowered.starts_with("models/gemini") {
            return Some(Self::Gemini {
                model: full_name.to_string(),
            });
        }

        // OpenAI models: gpt-*, o1-*, chatgpt-*, text-davinci-*, etc.
        if lowered.starts_with("gpt-")
            || lowered.starts_with("o1-")
            || lowered.starts_with("o3-")
            || lowered.starts_with("chatgpt-")
            || lowered.starts_with("text-")
        {
            return Some(Self::OpenAi {
                model: full_name.to_string(),
            });
        }

        // Claude/Anthropic models: claude-*
        if lowered.starts_with("claude-") {
            return Some(Self::Claude {
                model: full_name.to_string(),
            });
        }

        // xAI Grok models: grok-*
        if lowered.starts_with("grok-") {
            return Some(Self::Xai {
                model: full_name.to_string(),
            });
        }

        // Mistral API models: mistral-*
        if lowered.starts_with("mistral-") {
            return Some(Self::Mistral {
                model: full_name.to_string(),
            });
        }

        // Groq models: llama-* (via Groq), mixtral-*
        if lowered.starts_with("llama-") || lowered.starts_with("mixtral-") {
            return Some(Self::Groq {
                model: full_name.to_string(),
            });
        }

        // Ollama models: llama*, phi*, qwen*, gemma*, etc.
        // Use the full name to preserve version tags like ":1.5b"
        if lowered.starts_with("llama")
            || lowered.starts_with("phi")
            || lowered.starts_with("codellama")
            || lowered.starts_with("deepseek")
            || lowered.starts_with("qwen")
            || lowered.starts_with("gemma")
        {
            return Some(Self::Ollama {
                model: full_name.to_string(),
            });
        }

        None
    }

    fn label(&self) -> String {
        match self {
            Self::TinyLlama => TINYLLAMA_LABEL.to_string(),
            Self::Ghost { pack_path } => format!("ghost:{}", pack_path.display()),
            Self::Ollama { model } => format!("ollama:{model}"),
            Self::OpenAi { model } => format!("openai:{model}"),
            Self::Nvidia { model } => format!("nvidia:{model}"),
            Self::Gemini { model } => format!("gemini:{model}"),
            Self::Claude { model } => format!("claude:{model}"),
            Self::Xai { model } => format!("xai:{model}"),
            Self::Groq { model } => format!("groq:{model}"),
            Self::Mistral { model } => format!("mistral:{model}"),
        }
    }

    fn context_budget(&self) -> ModelContextBudget {
        match self {
            Self::TinyLlama => {
                ModelContextBudget::new(LOCAL_CONTEXT_CHARS, LOCAL_PROMPT_MARGIN_CHARS)
            }
            Self::Ghost { .. } => {
                ModelContextBudget::new(LOCAL_CONTEXT_CHARS, LOCAL_PROMPT_MARGIN_CHARS)
            }
            Self::Ollama { .. } => {
                ModelContextBudget::new(OLLAMA_PROMPT_CHARS, REMOTE_PROMPT_MARGIN_CHARS)
            }
            Self::OpenAi { .. } => {
                ModelContextBudget::new(OPENAI_PROMPT_CHARS, REMOTE_PROMPT_MARGIN_CHARS)
            }
            Self::Nvidia { .. } => {
                ModelContextBudget::new(NVIDIA_PROMPT_CHARS, REMOTE_PROMPT_MARGIN_CHARS)
            }
            Self::Gemini { .. } => {
                ModelContextBudget::new(GEMINI_PROMPT_CHARS, REMOTE_PROMPT_MARGIN_CHARS)
            }
            Self::Claude { .. } => {
                ModelContextBudget::new(CLAUDE_PROMPT_CHARS, REMOTE_PROMPT_MARGIN_CHARS)
            }
            Self::Xai { .. } => {
                ModelContextBudget::new(XAI_PROMPT_CHARS, REMOTE_PROMPT_MARGIN_CHARS)
            }
            Self::Groq { .. } => {
                ModelContextBudget::new(GROQ_PROMPT_CHARS, REMOTE_PROMPT_MARGIN_CHARS)
            }
            Self::Mistral { .. } => {
                ModelContextBudget::new(MISTRAL_PROMPT_CHARS, REMOTE_PROMPT_MARGIN_CHARS)
            }
        }
    }

    fn max_output_tokens(&self) -> usize {
        match self {
            Self::TinyLlama => LOCAL_MAX_OUTPUT_TOKENS,
            Self::Ghost { .. } => LOCAL_MAX_OUTPUT_TOKENS,
            Self::Ollama { .. }
            | Self::OpenAi { .. }
            | Self::Nvidia { .. }
            | Self::Gemini { .. }
            | Self::Claude { .. }
            | Self::Xai { .. }
            | Self::Groq { .. }
            | Self::Mistral { .. } => REMOTE_MAX_OUTPUT_TOKENS,
        }
    }
}

fn normalize_openai_model(explicit: Option<String>) -> String {
    match explicit {
        Some(raw) if !raw.trim().is_empty() => raw,
        _ => "gpt-4o-mini".to_string(),
    }
}

fn normalize_nvidia_model(explicit: Option<String>) -> String {
    match explicit {
        Some(raw) if !raw.trim().is_empty() => raw,
        _ => std::env::var("NVIDIA_LLM_MODEL")
            .or_else(|_| std::env::var("NVIDIA_MODEL"))
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
            .unwrap_or_default(),
    }
}

fn normalize_gemini_model(explicit: Option<String>) -> String {
    let default_model = "gemini-2.5-flash".to_string();
    let Some(raw) = explicit else {
        return default_model;
    };

    let lowered = raw.to_ascii_lowercase();
    match lowered.as_str() {
        "gemini-pro" | "gemini-1.5-pro" | "gemini-1.5-flash" | "gemini-2.0-pro-exp" => raw,
        _ => raw,
    }
}

fn normalize_claude_model(explicit: Option<String>) -> String {
    let default_model = "claude-sonnet-4-5".to_string();
    let Some(raw) = explicit else {
        return default_model;
    };

    // Map old model names to new ones
    match raw.as_str() {
        "claude-3-5-sonnet-20241022" | "claude-3.5-sonnet" | "sonnet" => {
            "claude-sonnet-4-5".to_string()
        }
        "claude-3-haiku-20240307" | "claude-3-haiku" | "haiku" => "claude-haiku-4-5".to_string(),
        "claude-3-opus-20240229" | "claude-3-opus" | "opus" => "claude-opus-4".to_string(),
        _ => raw,
    }
}

fn normalize_xai_model(explicit: Option<String>) -> String {
    let default_model = "grok-4-fast".to_string();
    let Some(raw) = explicit else {
        return default_model;
    };

    // Map common aliases to actual model names
    match raw.to_lowercase().as_str() {
        "grok" | "grok-fast" => "grok-4-fast".to_string(),
        "grok-4" | "grok-3" | "grok-4-fast" => raw, // Keep explicit versions
        _ => raw,
    }
}

fn normalize_groq_model(explicit: Option<String>) -> String {
    let default_model = "llama-3.3-70b-versatile".to_string();
    let Some(raw) = explicit else {
        return default_model;
    };

    // Map common aliases to actual model names
    match raw.to_lowercase().as_str() {
        "llama" | "llama3" | "llama-3" => "llama-3.3-70b-versatile".to_string(),
        "llama-70b" | "llama3-70b" => "llama-3.3-70b-versatile".to_string(),
        "llama-8b" | "llama3-8b" => "llama-3.1-8b-instant".to_string(),
        "mixtral" => "mixtral-8x7b-32768".to_string(),
        _ => raw,
    }
}

fn normalize_mistral_model(explicit: Option<String>) -> String {
    let default_model = "mistral-large-latest".to_string();
    let Some(raw) = explicit else {
        return default_model;
    };

    // Map common aliases to actual model names
    match raw.to_lowercase().as_str() {
        "mistral" | "large" | "mistral-large" => "mistral-large-latest".to_string(),
        "medium" | "mistral-medium" => "mistral-medium-latest".to_string(),
        "small" | "mistral-small" => "mistral-small-latest".to_string(),
        _ => raw,
    }
}

/// Calculate cost for a given model based on token usage.
/// Prices are per 1M tokens in USD (December 2025 pricing).
pub fn calculate_cost(model: &str, input_tokens: u32, output_tokens: u32) -> f64 {
    let (input_price, output_price) = match model.to_lowercase().as_str() {
        // OpenAI pricing (per 1M tokens) - Dec 2025
        m if m.contains("gpt-4o-mini") => (0.15, 0.60),
        m if m.contains("gpt-4o") => (2.50, 10.00),
        m if m.contains("gpt-4.5") => (75.00, 150.00),
        m if m.contains("gpt-4.1-mini") => (0.40, 1.60),
        m if m.contains("gpt-4.1") => (2.00, 8.00),
        m if m.contains("gpt-5.2") => (1.75, 14.00),
        m if m.contains("gpt-5") => (1.75, 14.00),
        m if m.contains("gpt-4-turbo") => (10.00, 30.00),
        m if m.contains("gpt-4") => (30.00, 60.00),
        m if m.contains("gpt-3.5") => (0.50, 1.50),
        m if m.contains("o1") || m.contains("o3") => (15.00, 60.00),

        // Claude/Anthropic pricing (per 1M tokens) - Dec 2025
        m if m.contains("claude-4-opus") || m.contains("claude-opus-4") => (15.00, 75.00),
        m if m.contains("claude-4-sonnet") || m.contains("claude-sonnet-4") => (3.00, 15.00),
        m if m.contains("claude-4-haiku") || m.contains("claude-haiku-4") => (0.25, 1.25),
        m if m.contains("claude-3-5-sonnet") || m.contains("claude-3.5-sonnet") => (3.00, 15.00),
        m if m.contains("claude-3-opus") => (15.00, 75.00),
        m if m.contains("claude-3-sonnet") => (3.00, 15.00),
        m if m.contains("claude-3-haiku") => (0.25, 1.25),
        m if m.contains("claude") => (3.00, 15.00), // Default to Sonnet pricing

        // Gemini/Google pricing (per 1M tokens) - Dec 2025
        m if m.contains("gemini-2.5-flash") => (0.15, 3.50),
        m if m.contains("gemini-2.5-pro") => (1.25, 10.00),
        m if m.contains("gemini-2.0") => (0.10, 0.40),
        m if m.contains("gemini-1.5-pro") => (1.25, 5.00),
        m if m.contains("gemini-1.5-flash") => (0.075, 0.30),
        m if m.contains("gemini") => (0.15, 3.50), // Default to 2.5 Flash

        // xAI Grok pricing (per 1M tokens) - Dec 2025
        m if m.contains("grok-4-fast") => (0.20, 0.50),
        m if m.contains("grok-4") => (3.00, 15.00),
        m if m.contains("grok-3") => (3.00, 15.00),
        m if m.contains("grok") => (3.00, 15.00),

        // Groq pricing (per 1M tokens) - Dec 2025
        m if m.contains("llama-3.3-70b") => (0.59, 0.79),
        m if m.contains("llama-3.1-70b") => (0.59, 0.79),
        m if m.contains("llama-3.1-8b") => (0.05, 0.08),
        m if m.contains("mixtral-8x7b") => (0.24, 0.24),

        // Mistral pricing (per 1M tokens) - Dec 2025
        m if m.contains("mistral-large-3") || m.contains("mistral-large-latest") => (0.50, 1.50),
        m if m.contains("mistral-large") => (2.00, 6.00),
        m if m.contains("mistral-medium") => (0.40, 1.20),
        m if m.contains("mistral-small") => (0.10, 0.30),
        m if m.contains("mistral") => (0.50, 1.50),

        // DeepSeek pricing (per 1M tokens) - Dec 2025
        m if m.contains("deepseek-v3") || m.contains("deepseek") => (0.27, 1.10),

        // NVIDIA NIM pricing (per 1M tokens)
        m if m.contains("nvidia") => (1.00, 3.00),

        // Local/free models
        m if m.contains("ollama") || m.contains("tinyllama") => (0.0, 0.0),

        // Default pricing (conservative estimate)
        _ => (1.00, 3.00),
    };

    let input_cost = (input_tokens as f64 / 1_000_000.0) * input_price;
    let output_cost = (output_tokens as f64 / 1_000_000.0) * output_price;
    input_cost + output_cost
}

/// Internal result type for provider runs
struct ProviderResult {
    answer: String,
    usage: Option<TokenUsage>,
}

/// Minimum score threshold for relevance. Below this, we say "no relevant info found".
/// Set to 0.0 to only block clearly irrelevant queries (negative scores).
const RELEVANCE_THRESHOLD: f32 = 0.0;

pub fn run_model_inference(
    requested_model: &str,
    question: &str,
    fallback_context: &str,
    hits: &[SearchHit],
    context_override: Option<usize>,
    api_key: Option<&str>,
    system_prompt_override: Option<&str>,
) -> Result<ModelInference, ModelRunError> {
    let Some(model_kind) = ModelKind::parse(requested_model) else {
        return Err(ModelRunError::UnsupportedModel(requested_model.to_string()));
    };

    // Check if top hit score is below relevance threshold
    let top_score = hits.first().and_then(|h| h.score).unwrap_or(0.0);
    if hits.is_empty() || top_score < RELEVANCE_THRESHOLD {
        // Extract unique topics from available hits for suggestions
        let mut topics: Vec<String> = hits
            .iter()
            .take(5)
            .filter_map(|h| h.title.clone())
            .collect();
        topics.dedup();

        let suggestions = if topics.is_empty() {
            "Try asking about the topics in your memory file.".to_string()
        } else {
            format!(
                "Your memory contains information about: {}. Try asking about these topics.",
                topics.join(", ")
            )
        };

        let no_match_answer = format!(
            "No relevant information found for your question.\n\n{}\n\nRelevance score: {:.2} (threshold: {:.2})",
            suggestions, top_score, RELEVANCE_THRESHOLD
        );

        return Ok(ModelInference {
            answer: ModelAnswer {
                requested: requested_model.to_string(),
                model: "none".to_string(),
                answer: no_match_answer,
            },
            context_body: String::new(),
            context_fragments: Vec::new(),
            usage: Some(TokenUsage {
                input_tokens: 0,
                output_tokens: 0,
                total_tokens: 0,
                cost_usd: 0.0,
            }),
            grounding: Some(GroundingResult {
                score: 0.0,
                sentence_count: 0,
                grounded_sentences: 0,
                sentence_scores: Vec::new(),
                has_warning: true,
                warning_reason: Some(
                    "No relevant information found - retrieval score below threshold".to_string(),
                ),
            }),
            cached: false,
        });
    }

    let mut budget = model_kind.context_budget();
    if let Some(override_chars) = context_override {
        budget = budget.apply_override(override_chars);
    }

    let context_plan = context::assemble_context(hits, fallback_context, &budget);

    // Check cache first
    if let Some(cached) = cache::check_cache(question, &context_plan.body, &model_kind.label()) {
        let grounding = Some(GroundingResult {
            score: cached.grounding_score,
            sentence_count: 0,
            grounded_sentences: 0,
            sentence_scores: Vec::new(),
            has_warning: cached.grounding_score < 0.4,
            warning_reason: if cached.grounding_score < 0.4 {
                Some("Cached answer - original grounding was low".to_string())
            } else {
                None
            },
        });

        let context_fragments = context_plan
            .records
            .into_iter()
            .map(ModelContextFragment::from_record)
            .collect();

        return Ok(ModelInference {
            answer: ModelAnswer {
                requested: requested_model.to_string(),
                model: cached.model.clone(),
                answer: cached.answer.clone(),
            },
            context_body: context_plan.body,
            context_fragments,
            usage: Some(TokenUsage {
                input_tokens: cached.input_tokens,
                output_tokens: cached.output_tokens,
                total_tokens: cached.input_tokens + cached.output_tokens,
                cost_usd: 0.0, // Cached = no cost
            }),
            grounding,
            cached: true,
        });
    }

    let prompt = build_prompt_parts(
        question,
        &context_plan.body,
        &budget,
        model_kind.max_output_tokens(),
    );

    let result = match &model_kind {
        ModelKind::TinyLlama => {
            #[cfg(feature = "llama-cpp")]
            {
                ProviderResult {
                    answer: tinyllama::run(&prompt)?,
                    usage: None, // Local models don't track tokens
                }
            }
            #[cfg(not(feature = "llama-cpp"))]
            {
                return Err(ModelRunError::UnsupportedModel(
                    "tinyllama (llama-cpp feature not enabled)".to_string(),
                ));
            }
        }
        ModelKind::Ghost { pack_path } => {
            return Err(ModelRunError::UnsupportedModel(
                format!("ghost model '{}' (ghost runtime not yet available)", pack_path.display()),
            ));
        }
        ModelKind::Ollama { model } => ProviderResult {
            answer: ollama::run(model, &prompt)?,
            usage: None, // Ollama doesn't always return usage
        },
        ModelKind::OpenAi { model } => {
            openai::run(model, &prompt, api_key, system_prompt_override)?
        }
        ModelKind::Nvidia { model } => ProviderResult {
            answer: nvidia::run(model, &prompt, api_key, system_prompt_override)?,
            usage: None, // NVIDIA NIM doesn't consistently return usage
        },
        ModelKind::Gemini { model } => {
            gemini::run(model, &prompt, api_key, system_prompt_override)?
        }
        ModelKind::Claude { model } => {
            claude::run(model, &prompt, api_key, system_prompt_override)?
        }
        ModelKind::Xai { model } => xai::run(model, &prompt, api_key, system_prompt_override)?,
        ModelKind::Groq { model } => groq::run(model, &prompt, api_key, system_prompt_override)?,
        ModelKind::Mistral { model } => {
            mistral::run(model, &prompt, api_key, system_prompt_override)?
        }
    };

    let context::ContextAggregation {
        body: context_body,
        records,
    } = context_plan;
    let context_fragments = records
        .into_iter()
        .map(ModelContextFragment::from_record)
        .collect();

    // Verify grounding of the answer against the context
    let grounding = Some(verify_grounding(&result.answer, &context_body));

    // Store in cache for future use
    let grounding_score = grounding.as_ref().map(|g| g.score).unwrap_or(0.5);
    let (input_tokens, output_tokens, cost_usd) = result
        .usage
        .as_ref()
        .map(|u| (u.input_tokens, u.output_tokens, u.cost_usd))
        .unwrap_or((0, 0, 0.0));

    cache::store_in_cache(
        question,
        &context_body,
        &model_kind.label(),
        cache::CacheEntry {
            answer: result.answer.clone(),
            model: model_kind.label(),
            input_tokens,
            output_tokens,
            cost_usd,
            grounding_score,
            created_at: std::time::SystemTime::now(),
        },
    );

    // Apply post-processing to clean up the answer
    let processed_answer = postprocess_answer(&result.answer);

    Ok(ModelInference {
        answer: ModelAnswer {
            requested: requested_model.to_string(),
            model: model_kind.label(),
            answer: processed_answer,
        },
        context_body,
        context_fragments,
        usage: result.usage,
        grounding,
        cached: false,
    })
}

/// Verify how well the answer is grounded in the provided context.
/// Returns a GroundingResult with a score (0.0 to 1.0) indicating
/// how well the answer is supported by the context.
pub fn verify_grounding(answer: &str, context: &str) -> GroundingResult {
    use std::collections::HashSet;

    if answer.is_empty() {
        return GroundingResult {
            score: 1.0, // Empty answer = no hallucination
            sentence_count: 0,
            grounded_sentences: 0,
            sentence_scores: Vec::new(),
            has_warning: false,
            warning_reason: None,
        };
    }

    if context.is_empty() {
        return GroundingResult {
            score: 0.0,
            sentence_count: 1,
            grounded_sentences: 0,
            sentence_scores: vec![0.0],
            has_warning: true,
            warning_reason: Some("No context provided - answer may be hallucinated".to_string()),
        };
    }

    // Normalize context for comparison
    let context_lower = context.to_lowercase();
    let context_words: HashSet<&str> = context_lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() > 2)
        .collect();

    // Split answer into sentences
    let sentences: Vec<&str> = answer
        .split(|c| c == '.' || c == '!' || c == '?')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty() && s.len() > 10)
        .collect();

    if sentences.is_empty() {
        return GroundingResult {
            score: 0.5, // Can't verify
            sentence_count: 0,
            grounded_sentences: 0,
            sentence_scores: Vec::new(),
            has_warning: false,
            warning_reason: None,
        };
    }

    let mut sentence_scores = Vec::with_capacity(sentences.len());
    let mut grounded_count = 0;

    for sentence in &sentences {
        let sentence_lower = sentence.to_lowercase();
        let sentence_words: HashSet<&str> = sentence_lower
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() > 2)
            .collect();

        if sentence_words.is_empty() {
            sentence_scores.push(0.5);
            continue;
        }

        // Calculate word overlap
        let overlap: usize = sentence_words.intersection(&context_words).count();
        let score = (overlap as f32) / (sentence_words.len() as f32).max(1.0);

        // Also check for exact phrase matches (stronger signal)
        let phrase_bonus = if context_lower.contains(&sentence_lower) {
            0.3
        } else {
            // Check for significant substring matches
            let words: Vec<&str> = sentence_lower.split_whitespace().collect();
            if words.len() >= 3 {
                let phrase = words[..3.min(words.len())].join(" ");
                if context_lower.contains(&phrase) {
                    0.15
                } else {
                    0.0
                }
            } else {
                0.0
            }
        };

        let final_score = (score + phrase_bonus).min(1.0);
        sentence_scores.push(final_score);

        if final_score >= 0.3 {
            grounded_count += 1;
        }
    }

    let overall_score = if sentence_scores.is_empty() {
        0.5
    } else {
        sentence_scores.iter().sum::<f32>() / sentence_scores.len() as f32
    };

    // Determine warning
    let (has_warning, warning_reason) = if overall_score < 0.2 {
        (
            true,
            Some("Answer appears to be poorly grounded in context".to_string()),
        )
    } else if overall_score < 0.4 && grounded_count < sentences.len() / 2 {
        (
            true,
            Some("Some statements may not be supported by context".to_string()),
        )
    } else {
        (false, None)
    };

    GroundingResult {
        score: overall_score,
        sentence_count: sentences.len(),
        grounded_sentences: grounded_count,
        sentence_scores,
        has_warning,
        warning_reason,
    }
}

mod context {
    use super::{ModelContextBudget, clamp_to};
    use memvid_core::types::SearchHit;

    const CONTEXT_HEADER: &str = "## Retrieval Context\n";
    const PRIMARY_HEADER: &str = "### Primary Hit\n";
    const CORRECTION_WARNING: &str = r#"
╔══════════════════════════════════════════════════════════════════╗
║  🔴 USER CORRECTION - THIS IS THE AUTHORITATIVE ANSWER          ║
║  Any contradicting information below is OUTDATED and WRONG.     ║
║  YOU MUST USE THE ANSWER FROM THIS CORRECTION.                  ║
╚══════════════════════════════════════════════════════════════════╝
"#;
    const SUPPORT_HEADER: &str = "### Supporting Hits\n";
    const SUMMARY_HEADER: &str = "### Overflow Summaries\n";
    const SUMMARY_HIGHLIGHT_CHARS: usize = 240;
    /// Minimum chars for a micro-summary when budget is tight (title + rank info)
    #[allow(dead_code)]
    const MICRO_SUMMARY_CHARS: usize = 80;

    #[derive(Debug, Clone)]
    pub(super) struct ContextAggregation {
        pub body: String,
        pub records: Vec<ContextRecord>,
    }

    impl ContextAggregation {
        fn from_fallback(fallback: &str, limit: usize) -> Self {
            let body = if limit == 0 || fallback.is_empty() {
                String::new()
            } else if fallback.len() <= limit {
                fallback.to_string()
            } else {
                clamp_to(fallback, limit)
            };
            Self {
                body,
                records: Vec::new(),
            }
        }
    }

    #[derive(Debug, Clone)]
    pub(super) struct ContextRecord {
        pub rank: usize,
        pub uri: String,
        pub title: Option<String>,
        pub score: Option<f32>,
        pub matches: usize,
        pub frame_id: u64,
        pub range: (usize, usize),
        pub chunk_range: Option<(usize, usize)>,
        pub text: String,
        pub mode: ContextMode,
    }

    #[derive(Debug, Clone, Copy, Eq, PartialEq)]
    pub(super) enum ContextMode {
        Full,
        Summary,
    }

    #[derive(Debug, Clone)]
    pub(super) struct ContextAssemblyPlan {
        primary: Option<ContextRecord>,
        supporting: Vec<ContextRecord>,
        summaries: Vec<ContextRecord>,
    }

    pub(super) fn assemble_context(
        hits: &[SearchHit],
        fallback: &str,
        budget: &ModelContextBudget,
    ) -> ContextAggregation {
        if hits.is_empty() {
            return ContextAggregation::from_fallback(fallback, budget.context_chars());
        }

        let plan = assemble_plan(hits, budget.context_chars());
        let mut body = String::new();
        let mut records = Vec::new();

        body.push_str(CONTEXT_HEADER);
        // Check if primary is a correction BEFORE moving it
        let primary_is_correction = plan
            .primary
            .as_ref()
            .map(|p| p.uri.contains("mv2://correction/"))
            .unwrap_or(false);
        if let Some(primary) = plan.primary {
            body.push_str(PRIMARY_HEADER);
            // Add correction warning if primary hit is a correction
            if primary_is_correction {
                body.push_str(CORRECTION_WARNING);
            }
            body.push_str(&primary.text);
            body.push_str("\n\n");
            records.push(primary);
        }

        if !plan.supporting.is_empty() {
            body.push_str(SUPPORT_HEADER);
            if primary_is_correction {
                body.push_str("⚠️ **WARNING: The following sources may contain OUTDATED information. Use the correction above.**\n\n");
            }
            for record in plan.supporting {
                // If primary is a correction, skip older corrections in supporting hits
                // to avoid confusing the LLM with conflicting correction data
                if primary_is_correction && record.uri.contains("mv2://correction/") {
                    continue;
                }
                body.push_str(&record.text);
                body.push_str("\n\n");
                records.push(record);
            }
        }

        if !plan.summaries.is_empty() {
            body.push_str(SUMMARY_HEADER);
            for record in plan.summaries {
                body.push_str(&record.text);
                body.push_str("\n\n");
                records.push(record);
            }
        }

        ContextAggregation { body, records }
    }

    fn assemble_plan(hits: &[SearchHit], mut remaining_chars: usize) -> ContextAssemblyPlan {
        let mut records = Vec::new();
        for hit in hits.iter().take(32) {
            let full_record = build_record(hit, render_full(hit), ContextMode::Full);
            let summary_record = build_record(hit, render_summary(hit), ContextMode::Summary);
            let micro_record = build_record(hit, render_micro_summary(hit), ContextMode::Summary);
            records.push((full_record, summary_record, micro_record));
        }

        let mut plan = ContextAssemblyPlan {
            primary: None,
            supporting: Vec::new(),
            summaries: Vec::new(),
        };

        // Handle primary hit (rank #1) - always include at least a summary
        if let Some((primary_full, primary_summary, primary_micro)) = records.first() {
            if primary_full.text.len() <= remaining_chars {
                remaining_chars = remaining_chars.saturating_sub(primary_full.text.len());
                plan.primary = Some(primary_full.clone());
            } else if primary_summary.text.len() <= remaining_chars {
                // Primary as summary (unusual but possible with very tight budget)
                remaining_chars = remaining_chars.saturating_sub(primary_summary.text.len());
                plan.primary = Some(primary_summary.clone());
            } else if primary_micro.text.len() <= remaining_chars {
                // At minimum, include micro-summary for primary
                remaining_chars = remaining_chars.saturating_sub(primary_micro.text.len());
                plan.primary = Some(primary_micro.clone());
            }
        }

        // Process remaining hits with fallback to micro-summaries
        for (idx, (full, summary, micro)) in records.iter().enumerate() {
            if idx == 0 {
                continue;
            }

            if full.text.len() <= remaining_chars {
                remaining_chars = remaining_chars.saturating_sub(full.text.len());
                plan.supporting.push(full.clone());
            } else if summary.text.len() <= remaining_chars {
                remaining_chars = remaining_chars.saturating_sub(summary.text.len());
                plan.summaries.push(summary.clone());
            } else if micro.text.len() <= remaining_chars {
                // Fallback: include at least a micro-summary to preserve ranking info
                // This ensures high-ranked hits are never completely dropped
                remaining_chars = remaining_chars.saturating_sub(micro.text.len());
                plan.summaries.push(micro.clone());
            }
            // If even micro-summary doesn't fit, the budget is truly exhausted
        }

        plan
    }

    fn render_full(hit: &SearchHit) -> String {
        let content = hit
            .chunk_text
            .clone()
            .or_else(|| Some(hit.text.clone()))
            .unwrap_or_default();

        // Clean up the content for better LLM comprehension
        let clean_content = clean_text_for_llm(&content);

        // Use a cleaner format that LLMs parse better
        let title = hit.title.clone().unwrap_or_default();
        let source_info = if title.is_empty() {
            format!("[Source #{}]", hit.rank)
        } else {
            format!("[Source #{}: {}]", hit.rank, title)
        };

        // Include relevance indicator for context
        let relevance = match hit.score {
            Some(s) if s > 0.8 => "⬤ High relevance",
            Some(s) if s > 0.5 => "◐ Medium relevance",
            _ => "",
        };

        if relevance.is_empty() {
            format!("{}\n{}", source_info, clean_content)
        } else {
            format!("{} ({})\n{}", source_info, relevance, clean_content)
        }
    }

    fn render_summary(hit: &SearchHit) -> String {
        let snippet = hit
            .chunk_text
            .clone()
            .or_else(|| Some(hit.text.clone()))
            .unwrap_or_default();
        let snippet = trim_highlight(&snippet, SUMMARY_HIGHLIGHT_CHARS);
        let clean_snippet = clean_text_for_llm(&snippet);
        format!("[Source #{}] {}", hit.rank, clean_snippet)
    }

    /// Create a minimal summary when budget is very tight.
    /// Always fits within MICRO_SUMMARY_CHARS to ensure important hits are never dropped.
    fn render_micro_summary(hit: &SearchHit) -> String {
        let title = hit.title.clone().unwrap_or_else(|| "untitled".to_string());
        let title_truncated = clamp_to(&title, 40);
        // Format: "[#2: document.pdf] ..." - ~50-60 chars max
        format!("[#{}: {}] ...", hit.rank, title_truncated)
    }

    /// Clean text to improve LLM comprehension
    fn clean_text_for_llm(text: &str) -> String {
        let mut result = text.to_string();

        // Remove excessive whitespace while preserving paragraph structure
        result = result
            .lines()
            .map(|line| line.trim())
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join("\n");

        // Normalize unicode quotes and dashes (using string patterns for multi-byte chars)
        result = result
            .replace("\u{2018}", "'") // Left single quote '
            .replace("\u{2019}", "'") // Right single quote '
            .replace("\u{201C}", "\"") // Left double quote "
            .replace("\u{201D}", "\"") // Right double quote "
            .replace("\u{2013}", "-") // En dash –
            .replace("\u{2014}", "-"); // Em dash —

        // Remove null bytes and other control characters
        result = result
            .chars()
            .filter(|c| !c.is_control() || *c == '\n' || *c == '\t')
            .collect();

        result
    }

    fn trim_highlight(text: &str, limit: usize) -> String {
        let clean = text.replace('\n', " ");
        clamp_to(&clean, limit)
    }

    fn build_record(hit: &SearchHit, text: String, mode: ContextMode) -> ContextRecord {
        ContextRecord {
            rank: hit.rank,
            uri: hit.uri.clone(),
            title: hit.title.clone(),
            score: hit.score,
            matches: hit.matches,
            frame_id: hit.frame_id,
            range: hit.range,
            chunk_range: hit.chunk_range,
            text,
            mode,
        }
    }
}

#[cfg(feature = "llama-cpp")]
mod tinyllama {
    use super::{ModelRunError, PromptParts, TINYLLAMA_LABEL, ThinkingSpinner};
    use anyhow::anyhow;
    use llama_cpp::standard_sampler::StandardSampler;
    use llama_cpp::{LlamaModel, LlamaParams, SessionParams};
    use tokio::runtime::Builder;

    use std::path::{Path, PathBuf};

    const MODEL_DIR: &str = "models/tinyllama";
    const GGUF_HINT: &str = "*.gguf";

    pub(super) fn run(prompt: &PromptParts) -> Result<String, ModelRunError> {
        let base_dir = Path::new(MODEL_DIR);
        let assets = RequiredAssets::new(base_dir);

        if let Some(missing) = assets.missing_paths() {
            return Err(ModelRunError::AssetsMissing {
                model: TINYLLAMA_LABEL.to_string(),
                missing,
            });
        }

        let gguf_path = assets.gguf_path.clone().ok_or_else(|| {
            ModelRunError::Runtime(anyhow!(
                "no GGUF model file found in {}",
                base_dir.display()
            ))
        })?;

        unsafe {
            std::env::set_var("GGML_LOG_LEVEL", "ERROR");
            std::env::set_var("LLAMA_LOG_LEVEL", "ERROR");
        }

        let model =
            LlamaModel::load_from_file(&gguf_path, LlamaParams::default()).map_err(|err| {
                ModelRunError::Runtime(anyhow!(
                    "failed to load TinyLlama weights from {}: {err}",
                    gguf_path.display()
                ))
            })?;

        let mut session_params = SessionParams::default();
        if session_params.n_ctx == 0 {
            session_params.n_ctx = 2048;
        }
        session_params.n_batch = session_params.n_ctx.min(512);
        if session_params.n_ubatch == 0 {
            session_params.n_ubatch = 512;
        }
        let max_tokens = session_params.n_ctx as usize;
        let mut session = model.create_session(session_params).map_err(|err| {
            ModelRunError::Runtime(anyhow!("failed to create TinyLlama session: {err}"))
        })?;

        let mut priming_tokens = model
            .tokenize_bytes(prompt.completion_prompt().as_bytes(), true, true)
            .map_err(|err| {
                ModelRunError::Runtime(anyhow!("failed to tokenize TinyLlama prompt: {err}"))
            })?;

        let requested_tokens = prompt.max_output_tokens();
        if max_tokens > 0 {
            let reserved = requested_tokens + 64;
            if priming_tokens.len() >= max_tokens.saturating_sub(reserved) {
                let target = max_tokens.saturating_sub(reserved).max(1);
                let tail_start = priming_tokens.len().saturating_sub(target);
                priming_tokens = priming_tokens.split_off(tail_start);
            }
        }

        session
            .advance_context_with_tokens(&priming_tokens)
            .map_err(|err| {
                ModelRunError::Runtime(anyhow!("failed to prime TinyLlama context: {err}"))
            })?;

        let handle = session
            .start_completing_with(StandardSampler::default(), requested_tokens)
            .map_err(|err| ModelRunError::Runtime(anyhow!("completion failed to start: {err}")))?;

        let runtime = Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|err| {
                ModelRunError::Runtime(anyhow!("failed to build tokio runtime: {err}"))
            })?;

        let mut spinner = ThinkingSpinner::start();
        let generated = runtime.block_on(async { handle.into_string_async().await });
        spinner.stop();

        let answer = generated.trim().to_string();

        if answer.is_empty() {
            Ok("No answer generated by TinyLlama.".to_string())
        } else {
            Ok(answer)
        }
    }

    struct RequiredAssets {
        gguf_path: Option<PathBuf>,
        base_dir: PathBuf,
    }

    impl RequiredAssets {
        fn new(base_dir: &Path) -> Self {
            let gguf_path = find_first_gguf(base_dir);
            Self {
                gguf_path,
                base_dir: base_dir.to_path_buf(),
            }
        }

        fn missing_paths(&self) -> Option<Vec<PathBuf>> {
            if self.gguf_path.is_some() {
                None
            } else {
                Some(vec![self.base_dir.join(GGUF_HINT)])
            }
        }
    }

    fn find_first_gguf(base_dir: &Path) -> Option<PathBuf> {
        let mut entries: Vec<PathBuf> = std::fs::read_dir(base_dir)
            .ok()?
            .filter_map(|entry| entry.ok().map(|e| e.path()))
            .filter(|path| path.is_file() && path.extension().map_or(false, |ext| ext == "gguf"))
            .collect();
        entries.sort();
        entries.into_iter().next()
    }
}

mod ollama {
    use super::{ModelRunError, PromptParts, ThinkingSpinner};
    use anyhow::anyhow;
    use reqwest::blocking::Client;
    use serde::Deserialize;
    use serde_json::json;

    const ENDPOINT: &str = "http://127.0.0.1:11434/api/generate";

    pub(super) fn run(model: &str, prompt: &PromptParts) -> Result<String, ModelRunError> {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()
            .map_err(|err| ModelRunError::Runtime(anyhow!("failed to build HTTP client: {err}")))?;

        let mut spinner = ThinkingSpinner::start();
        let response = client
            .post(ENDPOINT)
            .json(&json!({
                "model": model,
                "prompt": prompt.completion_prompt(),
                "stream": false
            }))
            .send()
            .map_err(|err| ModelRunError::Runtime(anyhow!("ollama request failed: {err}")))?
            .error_for_status()
            .map_err(|err| {
                ModelRunError::Runtime(anyhow!("ollama returned error status: {err}"))
            })?;

        let body: GenerateResponse = response.json().map_err(|err| {
            ModelRunError::Runtime(anyhow!("failed to decode ollama response: {err}"))
        })?;
        spinner.stop();

        let text = body.response.trim().to_string();
        if text.is_empty() {
            Ok("No answer returned by Ollama.".to_string())
        } else {
            Ok(text)
        }
    }

    #[derive(Debug, Deserialize)]
    struct GenerateResponse {
        #[serde(default)]
        response: String,
    }
}

mod openai {
    use super::{
        ModelRunError, PromptParts, ProviderResult, SYSTEM_PROMPT, ThinkingSpinner, TokenUsage,
        calculate_cost,
    };
    use anyhow::anyhow;
    use reqwest::blocking::Client;
    use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};
    use serde::Deserialize;
    use serde_json::json;

    const CHAT_ENDPOINT: &str = "https://api.openai.com/v1/chat/completions";
    const RESPONSES_ENDPOINT: &str = "https://api.openai.com/v1/responses";

    pub(super) fn run(
        model: &str,
        prompt: &PromptParts,
        override_key: Option<&str>,
        system_prompt_override: Option<&str>,
    ) -> Result<ProviderResult, ModelRunError> {
        let system_prompt = system_prompt_override.unwrap_or(SYSTEM_PROMPT);
        let key = override_key
            .map(|value| value.to_string())
            .or_else(|| std::env::var("OPENAI_API_KEY").ok())
            .ok_or_else(|| {
                ModelRunError::Runtime(anyhow!(
                    "OPENAI_API_KEY environment variable is required for OpenAI models"
                ))
            })?;

        let mut headers = HeaderMap::new();
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {key}")).map_err(|err| {
                ModelRunError::Runtime(anyhow!("invalid OPENAI_API_KEY header value: {err}"))
            })?,
        );
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        let client = Client::builder()
            .no_proxy()
            .timeout(std::time::Duration::from_secs(60))
            .default_headers(headers)
            .build()
            .map_err(|err| ModelRunError::Runtime(anyhow!("failed to build HTTP client: {err}")))?;

        let mut spinner = ThinkingSpinner::start();
        let (text, usage) = if requires_responses_api(model) {
            let combined_prompt = format!(
                "System instructions:\n{}\n\nUser query:\n{}",
                system_prompt,
                prompt.user_message()
            );
            let payload = json!({
                "model": model,
                "input": combined_prompt,
                "max_output_tokens": prompt.max_output_tokens() as u32,
                "reasoning": {
                    "effort": "low"
                }
            });

            let response = client
                .post(RESPONSES_ENDPOINT)
                .json(&payload)
                .send()
                .map_err(|err| ModelRunError::Runtime(anyhow!("OpenAI request failed: {err}")))?;

            let status = response.status();
            if !status.is_success() {
                let body = response
                    .text()
                    .unwrap_or_else(|_| "<failed to read body>".to_string());
                return Err(ModelRunError::Runtime(anyhow!(
                    "OpenAI returned error status {status}: {body}"
                )));
            }

            let body: ResponsesResponse = response.json().map_err(|err| {
                ModelRunError::Runtime(anyhow!("failed to decode OpenAI response: {err}"))
            })?;

            let usage = body.usage.as_ref().map(|u| {
                let input = u.input_tokens.unwrap_or(0);
                let output = u.output_tokens.unwrap_or(0);
                TokenUsage {
                    input_tokens: input,
                    output_tokens: output,
                    total_tokens: input + output,
                    cost_usd: calculate_cost(model, input, output),
                }
            });
            (extract_responses_text(&body), usage)
        } else {
            let payload = json!({
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt.user_message()}
                ],
                "temperature": 0.2,
                "max_tokens": prompt.max_output_tokens() as u32
            });

            let response = client
                .post(CHAT_ENDPOINT)
                .json(&payload)
                .send()
                .map_err(|err| ModelRunError::Runtime(anyhow!("OpenAI request failed: {err}")))?;

            let status = response.status();
            if !status.is_success() {
                let body = response
                    .text()
                    .unwrap_or_else(|_| "<failed to read body>".to_string());
                return Err(ModelRunError::Runtime(anyhow!(
                    "OpenAI returned error status {status}: {body}"
                )));
            }

            let body: ChatResponse = response.json().map_err(|err| {
                ModelRunError::Runtime(anyhow!("failed to decode OpenAI response: {err}"))
            })?;

            let usage = body.usage.as_ref().map(|u| TokenUsage {
                input_tokens: u.prompt_tokens,
                output_tokens: u.completion_tokens,
                total_tokens: u.total_tokens,
                cost_usd: calculate_cost(model, u.prompt_tokens, u.completion_tokens),
            });
            (extract_chat_text(&body), usage)
        };
        spinner.stop();
        Ok(ProviderResult {
            answer: text,
            usage,
        })
    }

    #[derive(Debug, Deserialize)]
    struct ChatResponse {
        choices: Vec<Choice>,
        #[serde(default)]
        usage: Option<ChatUsage>,
    }

    #[derive(Debug, Deserialize)]
    struct ChatUsage {
        prompt_tokens: u32,
        completion_tokens: u32,
        total_tokens: u32,
    }

    #[derive(Debug, Deserialize)]
    struct Choice {
        message: ChatMessage,
    }

    #[derive(Debug, Deserialize)]
    struct ChatMessage {
        #[serde(default)]
        content: Option<String>,
    }

    #[derive(Debug, Deserialize)]
    struct ResponsesResponse {
        #[serde(default)]
        output: Vec<ResponseItem>,
        #[serde(default)]
        output_text: Vec<String>,
        #[serde(default)]
        usage: Option<ResponsesUsage>,
    }

    #[derive(Debug, Deserialize)]
    struct ResponsesUsage {
        #[serde(default)]
        input_tokens: Option<u32>,
        #[serde(default)]
        output_tokens: Option<u32>,
    }

    #[derive(Debug, Deserialize)]
    struct ResponseItem {
        #[serde(default)]
        content: Vec<ResponseContent>,
    }

    #[derive(Debug, Deserialize)]
    struct ResponseContent {
        #[serde(rename = "type")]
        kind: String,
        #[serde(default)]
        text: Option<String>,
    }

    fn extract_chat_text(body: &ChatResponse) -> String {
        body.choices
            .iter()
            .find_map(|choice| choice.message.content.clone())
            .map(|value| value.trim().to_string())
            .unwrap_or_else(|| "No answer returned by OpenAI.".to_string())
    }

    fn extract_responses_text(body: &ResponsesResponse) -> String {
        if !body.output_text.is_empty() {
            let text = body
                .output_text
                .iter()
                .find(|value| !value.trim().is_empty());
            if let Some(text) = text {
                return text.trim().to_string();
            }
        }
        for item in &body.output {
            for segment in &item.content {
                match segment.kind.as_str() {
                    "output_text" | "text" => {
                        if let Some(text) = &segment.text {
                            let trimmed = text.trim();
                            if !trimmed.is_empty() {
                                return trimmed.to_string();
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        "No answer returned by OpenAI.".to_string()
    }

    fn requires_responses_api(model: &str) -> bool {
        let lowered = model.to_ascii_lowercase();
        lowered.starts_with("gpt-5") || lowered.contains("gpt-4.1")
    }
}

mod nvidia {
    use super::{ModelRunError, PromptParts, SYSTEM_PROMPT, ThinkingSpinner};
    use anyhow::anyhow;
    use reqwest::blocking::Client;
    use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};
    use serde::Deserialize;
    use serde_json::json;

    pub(super) fn run(
        model: &str,
        prompt: &PromptParts,
        override_key: Option<&str>,
        system_prompt_override: Option<&str>,
    ) -> Result<String, ModelRunError> {
        let system_prompt = system_prompt_override.unwrap_or(SYSTEM_PROMPT);
        let key = override_key
            .map(|value| value.to_string())
            .or_else(|| std::env::var("NVIDIA_API_KEY").ok())
            .ok_or_else(|| {
                ModelRunError::Runtime(anyhow!(
                    "NVIDIA_API_KEY environment variable is required for NVIDIA models"
                ))
            })?;

        let model = model.trim();
        if model.is_empty() {
            return Err(ModelRunError::Runtime(anyhow!(
                "NVIDIA model name required. Use `nvidia:<model>` or set NVIDIA_LLM_MODEL."
            )));
        }

        let base_url = std::env::var("NVIDIA_BASE_URL")
            .ok()
            .map(|value| value.trim().trim_end_matches('/').to_string())
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| "https://integrate.api.nvidia.com".to_string());
        let endpoint = format!("{base_url}/v1/chat/completions");

        let mut headers = HeaderMap::new();
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {key}")).map_err(|err| {
                ModelRunError::Runtime(anyhow!("invalid NVIDIA_API_KEY header value: {err}"))
            })?,
        );
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .default_headers(headers)
            .build()
            .map_err(|err| ModelRunError::Runtime(anyhow!("failed to build HTTP client: {err}")))?;

        let payload = json!({
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt.user_message()}
            ],
            "temperature": 0.2,
            "max_tokens": prompt.max_output_tokens() as u32
        });

        let mut spinner = ThinkingSpinner::start();
        let response = client
            .post(endpoint)
            .json(&payload)
            .send()
            .map_err(|err| ModelRunError::Runtime(anyhow!("NVIDIA request failed: {err}")))?;

        let status = response.status();
        if !status.is_success() {
            let body = response
                .text()
                .unwrap_or_else(|_| "<failed to read body>".to_string());
            spinner.stop();
            return Err(ModelRunError::Runtime(anyhow!(
                "NVIDIA returned error status {status}: {body}"
            )));
        }

        let body: ChatResponse = response.json().map_err(|err| {
            ModelRunError::Runtime(anyhow!("failed to decode NVIDIA response: {err}"))
        })?;
        spinner.stop();

        let text = body
            .choices
            .into_iter()
            .find_map(|choice| choice.message.content)
            .map(|value| value.trim().to_string())
            .unwrap_or_else(|| "No answer returned by NVIDIA.".to_string());

        Ok(text)
    }

    #[derive(Debug, Deserialize)]
    struct ChatResponse {
        choices: Vec<Choice>,
    }

    #[derive(Debug, Deserialize)]
    struct Choice {
        message: ChatMessage,
    }

    #[derive(Debug, Deserialize)]
    struct ChatMessage {
        #[serde(default)]
        content: Option<String>,
    }
}

mod gemini {
    use super::{
        ModelRunError, PromptParts, ProviderResult, SYSTEM_PROMPT, ThinkingSpinner, TokenUsage,
        calculate_cost,
    };
    use anyhow::anyhow;
    use reqwest::blocking::Client;
    use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};
    use serde::Deserialize;
    use serde_json::json;

    pub(super) fn run(
        model: &str,
        prompt: &PromptParts,
        override_key: Option<&str>,
        system_prompt_override: Option<&str>,
    ) -> Result<ProviderResult, ModelRunError> {
        let system_prompt = system_prompt_override.unwrap_or(SYSTEM_PROMPT);
        let key = override_key
            .map(|value| value.to_string())
            .or_else(|| std::env::var("GEMINI_API_KEY").ok())
            .ok_or_else(|| {
                ModelRunError::Runtime(anyhow!(
                    "GEMINI_API_KEY environment variable is required for Gemini models"
                ))
            })?;

        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent",
            model
        );

        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            HeaderName::from_static("x-goog-api-key"),
            HeaderValue::from_str(&key).map_err(|err| {
                ModelRunError::Runtime(anyhow!("invalid GEMINI_API_KEY header value: {err}"))
            })?,
        );

        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .default_headers(headers)
            .build()
            .map_err(|err| ModelRunError::Runtime(anyhow!("failed to build HTTP client: {err}")))?;

        let payload = json!({
            "contents": [{
                "parts": [
                    { "text": system_prompt },
                    { "text": prompt.user_message() }
                ]
            }],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": prompt.max_output_tokens() as u32,
                "topK": 40,
                "topP": 0.95
            }
        });

        let mut spinner = ThinkingSpinner::start();
        let response = client
            .post(url)
            .json(&payload)
            .send()
            .map_err(|err| ModelRunError::Runtime(anyhow!("Gemini request failed: {err}")))?
            .error_for_status()
            .map_err(|err| {
                ModelRunError::Runtime(anyhow!("Gemini returned error status: {err}"))
            })?;

        let body: GenerateResponse = response.json().map_err(|err| {
            ModelRunError::Runtime(anyhow!("failed to decode Gemini response: {err}"))
        })?;
        spinner.stop();

        let text = body
            .candidates
            .iter()
            .flat_map(|candidate| candidate.content.parts.iter())
            .find_map(|part| part.text.clone())
            .map(|value| value.trim().to_string())
            .unwrap_or_else(|| "No answer returned by Gemini.".to_string());

        let usage = body.usage_metadata.as_ref().map(|u| {
            let input = u.prompt_token_count.unwrap_or(0);
            let output = u.candidates_token_count.unwrap_or(0);
            TokenUsage {
                input_tokens: input,
                output_tokens: output,
                total_tokens: input + output,
                cost_usd: calculate_cost(model, input, output),
            }
        });

        Ok(ProviderResult {
            answer: text,
            usage,
        })
    }

    #[derive(Debug, Deserialize)]
    struct GenerateResponse {
        candidates: Vec<Candidate>,
        #[serde(default, rename = "usageMetadata")]
        usage_metadata: Option<GeminiUsage>,
    }

    #[derive(Debug, Deserialize)]
    struct GeminiUsage {
        #[serde(default, rename = "promptTokenCount")]
        prompt_token_count: Option<u32>,
        #[serde(default, rename = "candidatesTokenCount")]
        candidates_token_count: Option<u32>,
    }

    #[derive(Debug, Deserialize)]
    struct Candidate {
        content: CandidateContent,
    }

    #[derive(Debug, Deserialize)]
    struct CandidateContent {
        parts: Vec<CandidatePart>,
    }

    #[derive(Debug, Deserialize)]
    struct CandidatePart {
        #[serde(default)]
        text: Option<String>,
    }
}

mod claude {
    use super::{
        ModelRunError, PromptParts, ProviderResult, SYSTEM_PROMPT, ThinkingSpinner, TokenUsage,
        calculate_cost,
    };
    use anyhow::anyhow;
    use reqwest::blocking::Client;
    use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};
    use serde::Deserialize;
    use serde_json::json;

    const ENDPOINT: &str = "https://api.anthropic.com/v1/messages";
    const API_VERSION: &str = "2023-06-01";

    pub(super) fn run(
        model: &str,
        prompt: &PromptParts,
        override_key: Option<&str>,
        system_prompt_override: Option<&str>,
    ) -> Result<ProviderResult, ModelRunError> {
        let system_prompt = system_prompt_override.unwrap_or(SYSTEM_PROMPT);
        let key = override_key
            .map(|value| value.to_string())
            .or_else(|| std::env::var("ANTHROPIC_API_KEY").ok())
            .or_else(|| std::env::var("CLAUDE_API_KEY").ok())
            .ok_or_else(|| {
                ModelRunError::Runtime(anyhow!(
                    "ANTHROPIC_API_KEY environment variable is required for Claude models"
                ))
            })?;

        let mut headers = HeaderMap::new();
        headers.insert(
            HeaderName::from_static("x-api-key"),
            HeaderValue::from_str(&key).map_err(|err| {
                ModelRunError::Runtime(anyhow!("invalid ANTHROPIC_API_KEY header value: {err}"))
            })?,
        );
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            HeaderName::from_static("anthropic-version"),
            HeaderValue::from_static(API_VERSION),
        );

        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .default_headers(headers)
            .build()
            .map_err(|err| ModelRunError::Runtime(anyhow!("failed to build HTTP client: {err}")))?;

        let payload = json!({
            "model": model,
            "max_tokens": prompt.max_output_tokens() as u32,
            "temperature": 0.2,
            "system": system_prompt,
            "messages": [{
                "role": "user",
                "content": [{"type": "text", "text": prompt.user_message()}]
            }]
        });

        let mut spinner = ThinkingSpinner::start();
        let response = client
            .post(ENDPOINT)
            .json(&payload)
            .send()
            .map_err(|err| ModelRunError::Runtime(anyhow!("Claude request failed: {err}")))?
            .error_for_status()
            .map_err(|err| {
                ModelRunError::Runtime(anyhow!("Claude returned error status: {err}"))
            })?;

        let body: ClaudeResponse = response.json().map_err(|err| {
            ModelRunError::Runtime(anyhow!("failed to decode Claude response: {err}"))
        })?;
        spinner.stop();

        let text = body
            .content
            .iter()
            .find_map(|part| match part {
                ContentBlock::Text { text } if !text.trim().is_empty() => {
                    Some(text.trim().to_string())
                }
                _ => None,
            })
            .unwrap_or_else(|| "No answer returned by Claude.".to_string());

        let usage = body.usage.as_ref().map(|u| TokenUsage {
            input_tokens: u.input_tokens,
            output_tokens: u.output_tokens,
            total_tokens: u.input_tokens + u.output_tokens,
            cost_usd: calculate_cost(model, u.input_tokens, u.output_tokens),
        });

        Ok(ProviderResult {
            answer: text,
            usage,
        })
    }

    #[derive(Debug, Deserialize)]
    struct ClaudeResponse {
        #[serde(default)]
        content: Vec<ContentBlock>,
        #[serde(default)]
        usage: Option<ClaudeUsage>,
    }

    #[derive(Debug, Deserialize)]
    struct ClaudeUsage {
        input_tokens: u32,
        output_tokens: u32,
    }

    #[derive(Debug, Deserialize)]
    #[serde(tag = "type", rename_all = "lowercase")]
    enum ContentBlock {
        Text {
            text: String,
        },
        #[serde(other)]
        Other,
    }
}

mod xai {
    use super::{
        ModelRunError, PromptParts, ProviderResult, SYSTEM_PROMPT, ThinkingSpinner, TokenUsage,
        calculate_cost,
    };
    use anyhow::anyhow;
    use reqwest::blocking::Client;
    use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};
    use serde::Deserialize;
    use serde_json::json;

    const ENDPOINT: &str = "https://api.x.ai/v1/chat/completions";

    pub(super) fn run(
        model: &str,
        prompt: &PromptParts,
        override_key: Option<&str>,
        system_prompt_override: Option<&str>,
    ) -> Result<ProviderResult, ModelRunError> {
        let system_prompt = system_prompt_override.unwrap_or(SYSTEM_PROMPT);
        let key = override_key
            .map(|value| value.to_string())
            .or_else(|| std::env::var("XAI_API_KEY").ok())
            .or_else(|| std::env::var("GROK_API_KEY").ok())
            .ok_or_else(|| {
                ModelRunError::Runtime(anyhow!(
                    "XAI_API_KEY environment variable is required for Grok models"
                ))
            })?;

        let mut headers = HeaderMap::new();
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {key}")).map_err(|err| {
                ModelRunError::Runtime(anyhow!("invalid XAI_API_KEY header value: {err}"))
            })?,
        );
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .default_headers(headers)
            .build()
            .map_err(|err| ModelRunError::Runtime(anyhow!("failed to build HTTP client: {err}")))?;

        let payload = json!({
            "model": model,
            "max_tokens": prompt.max_output_tokens() as u32,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt.user_message()}
            ]
        });

        let mut spinner = ThinkingSpinner::start();
        let response = client
            .post(ENDPOINT)
            .json(&payload)
            .send()
            .map_err(|err| ModelRunError::Runtime(anyhow!("xAI request failed: {err}")))?
            .error_for_status()
            .map_err(|err| ModelRunError::Runtime(anyhow!("xAI returned error status: {err}")))?;

        let body: XaiResponse = response.json().map_err(|err| {
            ModelRunError::Runtime(anyhow!("failed to decode xAI response: {err}"))
        })?;
        spinner.stop();

        let text = body
            .choices
            .first()
            .and_then(|c| c.message.content.as_ref())
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| "No answer returned by Grok.".to_string());

        let usage = body.usage.as_ref().map(|u| TokenUsage {
            input_tokens: u.prompt_tokens,
            output_tokens: u.completion_tokens,
            total_tokens: u
                .total_tokens
                .unwrap_or(u.prompt_tokens + u.completion_tokens),
            cost_usd: calculate_cost(model, u.prompt_tokens, u.completion_tokens),
        });

        Ok(ProviderResult {
            answer: text,
            usage,
        })
    }

    #[derive(Debug, Deserialize)]
    struct XaiResponse {
        #[serde(default)]
        choices: Vec<XaiChoice>,
        #[serde(default)]
        usage: Option<XaiUsage>,
    }

    #[derive(Debug, Deserialize)]
    struct XaiChoice {
        message: XaiMessage,
    }

    #[derive(Debug, Deserialize)]
    struct XaiMessage {
        content: Option<String>,
    }

    #[derive(Debug, Deserialize)]
    struct XaiUsage {
        prompt_tokens: u32,
        completion_tokens: u32,
        total_tokens: Option<u32>,
    }
}

mod groq {
    use super::{
        ModelRunError, PromptParts, ProviderResult, SYSTEM_PROMPT, ThinkingSpinner, TokenUsage,
        calculate_cost,
    };
    use anyhow::anyhow;
    use reqwest::blocking::Client;
    use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};
    use serde::Deserialize;
    use serde_json::json;

    const ENDPOINT: &str = "https://api.groq.com/openai/v1/chat/completions";

    pub(super) fn run(
        model: &str,
        prompt: &PromptParts,
        override_key: Option<&str>,
        system_prompt_override: Option<&str>,
    ) -> Result<ProviderResult, ModelRunError> {
        let system_prompt = system_prompt_override.unwrap_or(SYSTEM_PROMPT);
        let key = override_key
            .map(|value| value.to_string())
            .or_else(|| std::env::var("GROQ_API_KEY").ok())
            .ok_or_else(|| {
                ModelRunError::Runtime(anyhow!(
                    "GROQ_API_KEY environment variable is required for Groq models"
                ))
            })?;

        let mut headers = HeaderMap::new();
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {key}")).map_err(|err| {
                ModelRunError::Runtime(anyhow!("invalid GROQ_API_KEY header value: {err}"))
            })?,
        );
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .default_headers(headers)
            .build()
            .map_err(|err| ModelRunError::Runtime(anyhow!("failed to build HTTP client: {err}")))?;

        let payload = json!({
            "model": model,
            "max_tokens": prompt.max_output_tokens() as u32,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt.user_message()}
            ]
        });

        let mut spinner = ThinkingSpinner::start();
        let response = client
            .post(ENDPOINT)
            .json(&payload)
            .send()
            .map_err(|err| ModelRunError::Runtime(anyhow!("Groq request failed: {err}")))?
            .error_for_status()
            .map_err(|err| ModelRunError::Runtime(anyhow!("Groq returned error status: {err}")))?;

        let body: GroqResponse = response.json().map_err(|err| {
            ModelRunError::Runtime(anyhow!("failed to decode Groq response: {err}"))
        })?;
        spinner.stop();

        let text = body
            .choices
            .first()
            .and_then(|c| c.message.content.as_ref())
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| "No answer returned by Groq.".to_string());

        let usage = body.usage.as_ref().map(|u| TokenUsage {
            input_tokens: u.prompt_tokens,
            output_tokens: u.completion_tokens,
            total_tokens: u
                .total_tokens
                .unwrap_or(u.prompt_tokens + u.completion_tokens),
            cost_usd: calculate_cost(model, u.prompt_tokens, u.completion_tokens),
        });

        Ok(ProviderResult {
            answer: text,
            usage,
        })
    }

    #[derive(Debug, Deserialize)]
    struct GroqResponse {
        #[serde(default)]
        choices: Vec<GroqChoice>,
        #[serde(default)]
        usage: Option<GroqUsage>,
    }

    #[derive(Debug, Deserialize)]
    struct GroqChoice {
        message: GroqMessage,
    }

    #[derive(Debug, Deserialize)]
    struct GroqMessage {
        content: Option<String>,
    }

    #[derive(Debug, Deserialize)]
    struct GroqUsage {
        prompt_tokens: u32,
        completion_tokens: u32,
        total_tokens: Option<u32>,
    }
}

mod mistral {
    use super::{
        ModelRunError, PromptParts, ProviderResult, SYSTEM_PROMPT, ThinkingSpinner, TokenUsage,
        calculate_cost,
    };
    use anyhow::anyhow;
    use reqwest::blocking::Client;
    use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};
    use serde::Deserialize;
    use serde_json::json;

    const ENDPOINT: &str = "https://api.mistral.ai/v1/chat/completions";

    pub(super) fn run(
        model: &str,
        prompt: &PromptParts,
        override_key: Option<&str>,
        system_prompt_override: Option<&str>,
    ) -> Result<ProviderResult, ModelRunError> {
        let system_prompt = system_prompt_override.unwrap_or(SYSTEM_PROMPT);
        let key = override_key
            .map(|value| value.to_string())
            .or_else(|| std::env::var("MISTRAL_API_KEY").ok())
            .ok_or_else(|| {
                ModelRunError::Runtime(anyhow!(
                    "MISTRAL_API_KEY environment variable is required for Mistral models"
                ))
            })?;

        let mut headers = HeaderMap::new();
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {key}")).map_err(|err| {
                ModelRunError::Runtime(anyhow!("invalid MISTRAL_API_KEY header value: {err}"))
            })?,
        );
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .default_headers(headers)
            .build()
            .map_err(|err| ModelRunError::Runtime(anyhow!("failed to build HTTP client: {err}")))?;

        let payload = json!({
            "model": model,
            "max_tokens": prompt.max_output_tokens() as u32,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt.user_message()}
            ]
        });

        let mut spinner = ThinkingSpinner::start();
        let response = client
            .post(ENDPOINT)
            .json(&payload)
            .send()
            .map_err(|err| ModelRunError::Runtime(anyhow!("Mistral request failed: {err}")))?
            .error_for_status()
            .map_err(|err| {
                ModelRunError::Runtime(anyhow!("Mistral returned error status: {err}"))
            })?;

        let body: MistralResponse = response.json().map_err(|err| {
            ModelRunError::Runtime(anyhow!("failed to decode Mistral response: {err}"))
        })?;
        spinner.stop();

        let text = body
            .choices
            .first()
            .and_then(|c| c.message.content.as_ref())
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| "No answer returned by Mistral.".to_string());

        let usage = body.usage.as_ref().map(|u| TokenUsage {
            input_tokens: u.prompt_tokens,
            output_tokens: u.completion_tokens,
            total_tokens: u
                .total_tokens
                .unwrap_or(u.prompt_tokens + u.completion_tokens),
            cost_usd: calculate_cost(model, u.prompt_tokens, u.completion_tokens),
        });

        Ok(ProviderResult {
            answer: text,
            usage,
        })
    }

    #[derive(Debug, Deserialize)]
    struct MistralResponse {
        #[serde(default)]
        choices: Vec<MistralChoice>,
        #[serde(default)]
        usage: Option<MistralUsage>,
    }

    #[derive(Debug, Deserialize)]
    struct MistralChoice {
        message: MistralMessage,
    }

    #[derive(Debug, Deserialize)]
    struct MistralMessage {
        content: Option<String>,
    }

    #[derive(Debug, Deserialize)]
    struct MistralUsage {
        prompt_tokens: u32,
        completion_tokens: u32,
        total_tokens: Option<u32>,
    }
}

// ============================================================================
// Entity Extraction API
// ============================================================================

/// Default system prompt for entity extraction
pub const ENTITY_EXTRACTION_PROMPT: &str = r#"Extract named entities from the provided text. Return a JSON object with an "entities" array.

Each entity should have:
- "name": The entity name as it appears in the text
- "type": One of "PERSON", "ORG", "LOCATION", "DATE", "PRODUCT", "EVENT", or "OTHER"
- "confidence": A number between 0.0 and 1.0 indicating your confidence

Guidelines:
1. Only include entities you're confident about (confidence >= 0.7)
2. Preserve the original capitalization of entity names
3. For organizations, include full names (e.g., "S&P Global" not just "S&P")
4. For people, include full names when available
5. Deduplicate: if an entity appears multiple times, include it only once

Return format:
{"entities": [{"name": "...", "type": "...", "confidence": 0.9}, ...]}"#;

/// Extracted entity from text
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExtractedEntity {
    pub name: String,
    #[serde(rename = "type")]
    pub entity_type: String,
    pub confidence: f32,
}

/// Response from entity extraction
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EntityExtractionResponse {
    pub entities: Vec<ExtractedEntity>,
    pub model: String,
    pub text_chars: usize,
}

/// Extract entities from text using an LLM
///
/// # Arguments
/// * `model` - Model identifier (e.g., "openai:gpt-4o-mini", "claude:claude-3-5-sonnet")
/// * `text` - The text to extract entities from
/// * `system_prompt` - Optional custom system prompt (uses default if None)
/// * `api_key` - Optional API key (uses environment variable if None)
///
/// # Returns
/// An `EntityExtractionResponse` with the extracted entities
///
/// # Example
/// ```ignore
/// let response = extract_entities(
///     "openai:gpt-4o-mini",
///     "John Smith met with Microsoft CEO Satya Nadella in Seattle.",
///     None,  // use default prompt
///     None,  // use OPENAI_API_KEY env var
/// )?;
/// for entity in response.entities {
///     println!("{}: {} ({:.0}%)", entity.name, entity.entity_type, entity.confidence * 100.0);
/// }
/// ```
pub fn extract_entities(
    model: &str,
    text: &str,
    system_prompt: Option<&str>,
    api_key: Option<&str>,
) -> Result<EntityExtractionResponse, ModelRunError> {
    let prompt = system_prompt.unwrap_or(ENTITY_EXTRACTION_PROMPT);
    let text_chars = text.len();

    // Determine model provider and make API call
    let (provider, model_name) = parse_model_spec(model);

    let json_response = match provider.as_str() {
        "openai" => extract_entities_openai(&model_name, text, prompt, api_key)?,
        "claude" | "anthropic" => extract_entities_claude(&model_name, text, prompt, api_key)?,
        "gemini" | "google" => extract_entities_gemini(&model_name, text, prompt, api_key)?,
        _ => {
            return Err(ModelRunError::UnsupportedModel(format!(
                "Entity extraction not supported for provider '{}'. Use openai:, claude:, or gemini:",
                provider
            )));
        }
    };

    // Parse the JSON response
    let entities = parse_entity_response(&json_response)?;

    Ok(EntityExtractionResponse {
        entities,
        model: model.to_string(),
        text_chars,
    })
}

fn parse_model_spec(model: &str) -> (String, String) {
    if let Some((provider, name)) = model.split_once(':') {
        (provider.to_lowercase(), name.to_string())
    } else {
        // Default to OpenAI if no provider specified
        ("openai".to_string(), model.to_string())
    }
}

fn parse_entity_response(json_str: &str) -> Result<Vec<ExtractedEntity>, ModelRunError> {
    // Try to parse the response, handling various formats
    let trimmed = json_str.trim();

    // Handle markdown code blocks
    let clean_json = if trimmed.starts_with("```json") {
        trimmed
            .strip_prefix("```json")
            .and_then(|s| s.strip_suffix("```"))
            .unwrap_or(trimmed)
            .trim()
    } else if trimmed.starts_with("```") {
        trimmed
            .strip_prefix("```")
            .and_then(|s| s.strip_suffix("```"))
            .unwrap_or(trimmed)
            .trim()
    } else {
        trimmed
    };

    // Try parsing as {"entities": [...]}
    #[derive(serde::Deserialize)]
    struct EntityResponse {
        entities: Vec<ExtractedEntity>,
    }

    if let Ok(response) = serde_json::from_str::<EntityResponse>(clean_json) {
        return Ok(response.entities);
    }

    // Try parsing as a direct array [...]
    if let Ok(entities) = serde_json::from_str::<Vec<ExtractedEntity>>(clean_json) {
        return Ok(entities);
    }

    Err(ModelRunError::Runtime(anyhow::anyhow!(
        "Failed to parse entity extraction response as JSON: {}",
        &clean_json[..clean_json.len().min(200)]
    )))
}

fn extract_entities_openai(
    model: &str,
    text: &str,
    system_prompt: &str,
    api_key: Option<&str>,
) -> Result<String, ModelRunError> {
    use serde_json::json;

    let api_key = api_key
        .map(|s| s.to_string())
        .or_else(|| std::env::var("OPENAI_API_KEY").ok())
        .ok_or_else(|| {
            ModelRunError::Runtime(anyhow::anyhow!(
                "OpenAI API key required. Set OPENAI_API_KEY or pass api_key parameter."
            ))
        })?;

    let model_name = if model.is_empty() {
        "gpt-4o-mini"
    } else {
        model
    };

    let client = reqwest::blocking::Client::builder()
        .no_proxy()
        .build()
        .map_err(|err| {
            ModelRunError::Runtime(anyhow::anyhow!("failed to build HTTP client: {err}"))
        })?;
    let payload = json!({
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.1
    });

    let response = client
        .post("https://api.openai.com/v1/chat/completions")
        .header("Authorization", format!("Bearer {}", api_key))
        .json(&payload)
        .send()
        .map_err(|e| ModelRunError::Runtime(anyhow::anyhow!("OpenAI request failed: {}", e)))?
        .error_for_status()
        .map_err(|e| ModelRunError::Runtime(anyhow::anyhow!("OpenAI returned error: {}", e)))?;

    #[derive(serde::Deserialize)]
    struct OpenAIResponse {
        choices: Vec<OpenAIChoice>,
    }
    #[derive(serde::Deserialize)]
    struct OpenAIChoice {
        message: OpenAIMessage,
    }
    #[derive(serde::Deserialize)]
    struct OpenAIMessage {
        content: String,
    }

    let body: OpenAIResponse = response.json().map_err(|e| {
        ModelRunError::Runtime(anyhow::anyhow!("Failed to parse OpenAI response: {}", e))
    })?;

    body.choices
        .into_iter()
        .next()
        .map(|c| c.message.content)
        .ok_or_else(|| ModelRunError::Runtime(anyhow::anyhow!("No response from OpenAI")))
}

fn extract_entities_claude(
    model: &str,
    text: &str,
    system_prompt: &str,
    api_key: Option<&str>,
) -> Result<String, ModelRunError> {
    use serde_json::json;

    let api_key = api_key
        .map(|s| s.to_string())
        .or_else(|| std::env::var("ANTHROPIC_API_KEY").ok())
        .ok_or_else(|| {
            ModelRunError::Runtime(anyhow::anyhow!(
                "Anthropic API key required. Set ANTHROPIC_API_KEY or pass api_key parameter."
            ))
        })?;

    let model_name = if model.is_empty() {
        "claude-3-5-sonnet-20241022"
    } else {
        model
    };

    let client = reqwest::blocking::Client::builder()
        .no_proxy()
        .build()
        .map_err(|err| {
            ModelRunError::Runtime(anyhow::anyhow!("failed to build HTTP client: {err}"))
        })?;
    let payload = json!({
        "model": model_name,
        "max_tokens": 4096,
        "system": format!("{}\n\nRespond with valid JSON only.", system_prompt),
        "messages": [
            {"role": "user", "content": text}
        ]
    });

    let response = client
        .post("https://api.anthropic.com/v1/messages")
        .header("x-api-key", &api_key)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .json(&payload)
        .send()
        .map_err(|e| ModelRunError::Runtime(anyhow::anyhow!("Claude request failed: {}", e)))?
        .error_for_status()
        .map_err(|e| ModelRunError::Runtime(anyhow::anyhow!("Claude returned error: {}", e)))?;

    #[derive(serde::Deserialize)]
    struct ClaudeResponse {
        content: Vec<ClaudeContent>,
    }
    #[derive(serde::Deserialize)]
    struct ClaudeContent {
        text: Option<String>,
    }

    let body: ClaudeResponse = response.json().map_err(|e| {
        ModelRunError::Runtime(anyhow::anyhow!("Failed to parse Claude response: {}", e))
    })?;

    body.content
        .into_iter()
        .find_map(|c| c.text)
        .ok_or_else(|| ModelRunError::Runtime(anyhow::anyhow!("No text response from Claude")))
}

fn extract_entities_gemini(
    model: &str,
    text: &str,
    system_prompt: &str,
    api_key: Option<&str>,
) -> Result<String, ModelRunError> {
    use serde_json::json;

    let api_key = api_key
        .map(|s| s.to_string())
        .or_else(|| std::env::var("GEMINI_API_KEY").ok())
        .or_else(|| std::env::var("GOOGLE_API_KEY").ok())
        .ok_or_else(|| {
            ModelRunError::Runtime(anyhow::anyhow!(
                "Gemini API key required. Set GEMINI_API_KEY or pass api_key parameter."
            ))
        })?;

    let model_name = if model.is_empty() {
        "gemini-2.0-flash"
    } else {
        model
    };
    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
        model_name, api_key
    );

    let client = reqwest::blocking::Client::builder()
        .no_proxy()
        .build()
        .map_err(|err| {
            ModelRunError::Runtime(anyhow::anyhow!("failed to build HTTP client: {err}"))
        })?;
    let payload = json!({
        "contents": [{
            "parts": [{"text": format!("{}\n\nText to analyze:\n{}", system_prompt, text)}]
        }],
        "generationConfig": {
            "temperature": 0.1,
            "responseMimeType": "application/json"
        }
    });

    let response = client
        .post(&url)
        .json(&payload)
        .send()
        .map_err(|e| ModelRunError::Runtime(anyhow::anyhow!("Gemini request failed: {}", e)))?
        .error_for_status()
        .map_err(|e| ModelRunError::Runtime(anyhow::anyhow!("Gemini returned error: {}", e)))?;

    #[derive(serde::Deserialize)]
    struct GeminiResponse {
        candidates: Vec<GeminiCandidate>,
    }
    #[derive(serde::Deserialize)]
    struct GeminiCandidate {
        content: GeminiContent,
    }
    #[derive(serde::Deserialize)]
    struct GeminiContent {
        parts: Vec<GeminiPart>,
    }
    #[derive(serde::Deserialize)]
    struct GeminiPart {
        text: Option<String>,
    }

    let body: GeminiResponse = response.json().map_err(|e| {
        ModelRunError::Runtime(anyhow::anyhow!("Failed to parse Gemini response: {}", e))
    })?;

    body.candidates
        .into_iter()
        .next()
        .and_then(|c| c.content.parts.into_iter().find_map(|p| p.text))
        .ok_or_else(|| ModelRunError::Runtime(anyhow::anyhow!("No text response from Gemini")))
}

#[cfg(test)]
mod tests {
    use super::*;

    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn normalize_models() {
        assert_eq!(normalize_openai_model(None), "gpt-4o-mini");
        assert_eq!(
            normalize_nvidia_model(Some("meta/llama3-8b-instruct".to_string())),
            "meta/llama3-8b-instruct"
        );
        let _lock = ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::remove_var("NVIDIA_LLM_MODEL");
            std::env::remove_var("NVIDIA_MODEL");
        }
        assert_eq!(normalize_nvidia_model(None), "");
        assert_eq!(normalize_gemini_model(None), "gemini-2.5-flash");
        assert_eq!(normalize_claude_model(None), "claude-sonnet-4-5");
        assert_eq!(normalize_xai_model(None), "grok-4-fast");
        assert_eq!(normalize_groq_model(None), "llama-3.3-70b-versatile");
        assert_eq!(normalize_mistral_model(None), "mistral-large-latest");
    }

    #[test]
    fn parse_entity_json() {
        let json = r#"{"entities": [{"name": "John", "type": "PERSON", "confidence": 0.95}]}"#;
        let entities = parse_entity_response(json).unwrap();
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].name, "John");
    }

    #[test]
    fn parse_entity_json_with_markdown() {
        let json = r#"```json
{"entities": [{"name": "Microsoft", "type": "ORG", "confidence": 0.99}]}
```"#;
        let entities = parse_entity_response(json).unwrap();
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].name, "Microsoft");
    }

    #[test]
    fn parse_model_spec_test() {
        let (provider, model) = parse_model_spec("openai:gpt-4o");
        assert_eq!(provider, "openai");
        assert_eq!(model, "gpt-4o");

        let (provider, model) = parse_model_spec("gpt-4o-mini");
        assert_eq!(provider, "openai");
        assert_eq!(model, "gpt-4o-mini");
    }

    #[test]
    fn modelkind_parses_ghost_spec() {
        let kind = ModelKind::parse("ghost:/tmp/model.ghostpack").unwrap();
        match kind {
            ModelKind::Ghost { pack_path } => {
                assert_eq!(pack_path, PathBuf::from("/tmp/model.ghostpack"));
            }
            other => panic!("expected ghost, got {other:?}"),
        }
    }

    #[test]
    fn run_model_inference_ghost_returns_unsupported() {
        let hit = SearchHit {
            rank: 0,
            frame_id: 0,
            uri: "mv2://test".to_string(),
            title: Some("Test".to_string()),
            range: (0, 3),
            text: "ctx".to_string(),
            matches: 1,
            chunk_range: None,
            chunk_text: None,
            score: Some(1.0),
            metadata: None,
        };

        let err = run_model_inference("ghost:/tmp/fake.ghostpack", "hello?", "", &[hit], None, None, None)
            .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("ghost") || msg.contains("Ghost"), "error should mention ghost: {msg}");
    }

    #[test]
    fn normalize_question_adds_question_mark() {
        // Should add ? to questions without punctuation
        // Note: abbreviation expansion may also occur (IRR -> IRR (internal rate of return))
        let result = normalize_question("how much is the LP rate");
        assert!(result.ends_with('?'), "should end with ?");

        assert_eq!(
            normalize_question("what is the total revenue"),
            "what is the total revenue?"
        );
        assert_eq!(
            normalize_question("where does John live"),
            "where does John live?"
        );
        assert_eq!(normalize_question("is this correct"), "is this correct?");
        assert_eq!(normalize_question("can you help me"), "can you help me?");
    }

    #[test]
    fn normalize_question_preserves_existing_punctuation() {
        // Should NOT modify queries that already have punctuation
        assert_eq!(normalize_question("how much is X?"), "how much is X?");
        assert_eq!(
            normalize_question("Tell me about the project."),
            "Tell me about the project."
        );
        assert_eq!(normalize_question("Do it now!"), "Do it now!");
    }

    #[test]
    fn normalize_question_ignores_non_questions() {
        // Should NOT add ? to non-question statements
        assert_eq!(
            normalize_question("revenue for Q1 2024"),
            "revenue for Q1 2024"
        );
        assert_eq!(normalize_question("total sales"), "total sales");
        // Should not match partial words
        assert_eq!(
            normalize_question("howitzer specifications"),
            "howitzer specifications"
        );
    }

    #[test]
    fn normalize_question_handles_edge_cases() {
        assert_eq!(normalize_question(""), "");
        assert_eq!(normalize_question("  "), "");
        // Note: typo correction and expansion happen, so result may differ
        let result = normalize_question("  how much  ");
        assert!(result.ends_with('?'), "should end with ?");
    }

    #[test]
    fn fix_typos_corrects_common_errors() {
        assert!(fix_common_typos("teh quick brown fox").contains("the"));
        assert!(fix_common_typos("waht is this").contains("what"));
        assert!(fix_common_typos("totla revenue").contains("total"));
    }

    #[test]
    fn expand_abbreviations_works() {
        // Test that abbreviations are expanded
        let result = expand_abbreviations("what is the irr");
        assert!(result.contains("internal rate of return") || result.contains("irr"));
    }

    #[test]
    fn question_type_detection() {
        assert_eq!(
            detect_question_type("how much is X?"),
            QuestionType::Numeric
        );
        assert_eq!(
            detect_question_type("is this correct?"),
            QuestionType::YesNo
        );
        assert_eq!(detect_question_type("list all items"), QuestionType::List);
        assert_eq!(
            detect_question_type("when was it created?"),
            QuestionType::Temporal
        );
        assert_eq!(
            detect_question_type("why did this happen?"),
            QuestionType::Explanation
        );
        assert_eq!(
            detect_question_type("what is the name?"),
            QuestionType::Factual
        );
    }

    #[test]
    fn postprocess_removes_artifacts() {
        let answer = "Based on the provided context, the value is 42.";
        let processed = postprocess_answer(answer);
        assert!(!processed.starts_with("Based on"));
        assert!(processed.contains("42"));
    }

    #[test]
    fn postprocess_capitalizes() {
        let answer = "the answer is yes";
        let processed = postprocess_answer(answer);
        assert!(processed.starts_with('T'), "should start with capital T");
    }

    #[test]
    fn postprocess_normalizes_whitespace() {
        let answer = "too    many     spaces    here";
        let processed = postprocess_answer(answer);
        assert!(!processed.contains("  "), "should not have double spaces");
    }
}
