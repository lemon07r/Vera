//! Query-aware ranking heuristics layered on top of dense + lexical retrieval.
//!
//! These heuristics intentionally stay simple and deterministic. They target
//! recurring benchmark failures that single-vector retrieval struggles with:
//! config files at repo root, test/docs noise, symbol-type disambiguation, and
//! same-file crowding for multi-file questions.

use crate::chunk_text::file_name;
use crate::corpus::{ContentClass, classify_content, classify_path, content_class_label};
use crate::retrieval::query_classifier::{QueryType, classify_query};
use crate::retrieval::query_utils::{
    looks_like_compound_identifier, looks_like_filename, path_depth, trim_query_token,
};
use crate::types::{Language, SearchFilters, SearchResult, SymbolType};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RankingStage {
    Initial,
    PostRerank,
}

#[derive(Debug, Clone)]
struct QueryFeatures {
    query_word_count: usize,
    path_fragment: Option<String>,
    exact_filename: Option<String>,
    exact_identifier_case: Option<String>,
    exact_identifier: Option<String>,
    keywords: Vec<String>,
    requested_symbol_types: Vec<SymbolType>,
    query_type: QueryType,
    wants_test_paths: bool,
    wants_docs_paths: bool,
    wants_example_paths: bool,
    wants_config_paths: bool,
    wants_runtime_paths: bool,
    wants_archive_paths: bool,
    wants_compat_paths: bool,
    wants_type_declarations: bool,
    requested_versions: Vec<String>,
    wants_multi_file_diversity: bool,
    mentions_implementation: bool,
    mentions_definition: bool,
}

impl QueryFeatures {
    fn from_query(query: &str) -> Self {
        let lower = query.trim().to_ascii_lowercase();
        let query_type = classify_query(query);
        let raw_tokens: Vec<&str> = query.split_whitespace().collect();
        let cleaned_tokens: Vec<String> = query
            .split_whitespace()
            .map(clean_query_token)
            .filter(|token| !token.is_empty())
            .collect();
        let path_fragment = cleaned_tokens
            .iter()
            .find(|token| looks_like_path_fragment(token))
            .cloned();
        let exact_filename = cleaned_tokens
            .iter()
            .find(|token| looks_like_filename(token))
            .map(|token| file_name(token).to_string());
        let exact_identifier = raw_tokens
            .iter()
            .map(|token| trim_query_token(token))
            .find(|token| {
                !token.is_empty()
                    && !looks_like_filename(&token.to_ascii_lowercase())
                    && looks_like_compound_identifier(token)
            })
            .map(|token| token.to_ascii_lowercase())
            .or_else(|| {
                if query_type == QueryType::Identifier && cleaned_tokens.len() == 1 {
                    cleaned_tokens
                        .first()
                        .filter(|token| !looks_like_filename(token))
                        .cloned()
                } else {
                    None
                }
            });
        let exact_identifier_case = exact_identifier.as_ref().and_then(|_| {
            raw_tokens
                .iter()
                .map(|token| trim_query_token(token))
                .find(|token| {
                    !token.is_empty()
                        && !looks_like_filename(&token.to_ascii_lowercase())
                        && looks_like_compound_identifier(token)
                })
                .map(ToString::to_string)
        });
        let keywords = cleaned_tokens
            .iter()
            .filter(|token| !looks_like_filename(token) && !is_query_stopword(token))
            .map(|token| normalize_token(token))
            .filter(|token| !token.is_empty())
            .collect();
        let requested_symbol_types = requested_symbol_types(&lower);

        Self {
            query_word_count: raw_tokens.len(),
            path_fragment,
            exact_identifier_case,
            wants_test_paths: mentions_any(&lower, &["test", "tests", "spec", "__tests__"]),
            wants_docs_paths: mentions_any(&lower, &["docs", "documentation", "readme"]),
            wants_example_paths: mentions_any(&lower, &["example", "examples", "demo", "sample"]),
            wants_config_paths: is_path_weighted_query(query)
                || mentions_any(
                    &lower,
                    &["configuration", "config", "workspace", "settings"],
                ),
            wants_runtime_paths: mentions_any(
                &lower,
                &[
                    "runtime",
                    "bundle",
                    "bundles",
                    "minified",
                    "minify",
                    "extract",
                    "extracted",
                    "asar",
                    "dist",
                ],
            ),
            wants_archive_paths: mentions_any(
                &lower,
                &["archive", "archived", "legacy", "snapshot", "deprecated"],
            ),
            wants_compat_paths: mentions_any(
                &lower,
                &[
                    "compat",
                    "compatibility",
                    "legacy",
                    "shim",
                    "polyfill",
                    "adapter",
                ],
            ),
            wants_type_declarations: mentions_any(
                &lower,
                &["declaration", "declarations", ".d.ts", "types", "typings"],
            ),
            requested_versions: requested_versions(&cleaned_tokens),
            wants_multi_file_diversity: !is_path_weighted_query(query)
                && (query_type == QueryType::NaturalLanguage
                    || (exact_identifier.is_some() && raw_tokens.len() <= 2)),
            mentions_implementation: mentions_any(
                &lower,
                &[
                    "implementation",
                    "implementations",
                    "impl",
                    "mounted",
                    "registration",
                ],
            ),
            mentions_definition: mentions_any(
                &lower,
                &[
                    "definition",
                    "definitions",
                    "define",
                    "declared",
                    "declaration",
                ],
            ),
            exact_filename,
            exact_identifier,
            keywords,
            requested_symbol_types,
            query_type,
        }
    }
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn apply_query_ranking(
    query: &str,
    results: Vec<SearchResult>,
    stage: RankingStage,
) -> Vec<SearchResult> {
    apply_query_ranking_with_filters(query, results, stage, &SearchFilters::default())
}

pub(crate) fn apply_query_ranking_with_filters(
    query: &str,
    results: Vec<SearchResult>,
    stage: RankingStage,
    filters: &SearchFilters,
) -> Vec<SearchResult> {
    if results.len() <= 1 {
        return results;
    }

    let features = QueryFeatures::from_query(query);
    let file_relevance = file_relevance_counts(&features, &results);
    let len = results.len() as f64;
    let mut scored: Vec<(f64, usize, SearchResult)> = results
        .into_iter()
        .enumerate()
        .map(|(idx, mut result)| {
            let base_rank = 1.0 - (idx as f64 / len);
            let same_file_hits = file_relevance
                .get(&result.file_path)
                .copied()
                .unwrap_or_default();
            let prior = score_prior(&features, &result, stage, filters, same_file_hits);
            let combined = base_rank + prior;
            result.score = combined;
            (combined, idx, result)
        })
        .collect();

    scored.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });

    let reranked = scored.into_iter().map(|(_, _, result)| result).collect();
    let reranked = if features.wants_multi_file_diversity {
        diversify_by_file(reranked)
    } else {
        reranked
    };
    stamp_rank_scores(reranked)
}

pub(crate) fn classify_file_role(file_path: &str, language: Language) -> ContentClass {
    classify_path(file_path, language)
}

pub(crate) fn file_role_label(file_path: &str, language: Language) -> &'static str {
    content_class_label(classify_file_role(file_path, language))
}

pub(crate) fn is_path_weighted_query(query: &str) -> bool {
    let lower = query.trim().to_ascii_lowercase();
    lower.contains('/')
        || lower.contains('\\')
        || lower.contains(".toml")
        || lower.contains(".json")
        || lower.contains(".yaml")
        || lower.contains(".yml")
        || lower.contains(".ini")
        || lower.contains(".conf")
        || lower.contains("dockerfile")
        || lower.contains("makefile")
        || lower.contains("cmakelists.txt")
}

fn score_prior(
    features: &QueryFeatures,
    result: &SearchResult,
    stage: RankingStage,
    filters: &SearchFilters,
    same_file_hits: usize,
) -> f64 {
    let stage_weight = match stage {
        RankingStage::Initial => 1.0,
        RankingStage::PostRerank => 0.55,
    };
    let depth = path_depth(&result.file_path) as f64;
    let role = classify_content(&result.file_path, result.language, &result.content);
    let mut bonus = 0.0;
    let file_path = result.file_path.to_ascii_lowercase();
    let result_filename = file_name(&result.file_path).to_ascii_lowercase();
    let allow_filename_semantic_bonus = matches!(
        role,
        ContentClass::Source | ContentClass::Config | ContentClass::Unknown
    );
    let path_fragment_match = features
        .path_fragment
        .as_deref()
        .is_some_and(|fragment| path_matches_fragment(&file_path, fragment));
    let filename_boost_allowed = features.path_fragment.is_none() || path_fragment_match;

    if path_fragment_match {
        bonus += stage_weight * 1.2;
    }

    if let Some(filename) = features.exact_filename.as_deref() {
        if filename_boost_allowed && result_filename == filename {
            let filename_bonus = if features.wants_config_paths {
                if depth == 0.0 {
                    1.15
                } else {
                    (0.45 - depth.min(5.0) * 0.08).max(0.08)
                }
            } else if depth == 0.0 {
                0.9
            } else {
                (0.6 - depth.min(5.0) * 0.06).max(0.12)
            };
            bonus += stage_weight * filename_bonus;
        } else if filename_boost_allowed && file_path.ends_with(filename) {
            bonus += stage_weight * 0.15;
        }
    }

    if features.wants_config_paths && matches!(role, ContentClass::Config) {
        bonus += stage_weight
            * if depth == 0.0 {
                0.35
            } else {
                (0.2 - depth.min(5.0) * 0.03).max(0.05)
            };
    }

    if let Some(identifier) = features.exact_identifier.as_deref() {
        if result
            .symbol_name
            .as_deref()
            .is_some_and(|name| name.eq_ignore_ascii_case(identifier))
        {
            bonus += stage_weight
                * if features.query_word_count <= 2 {
                    0.7
                } else {
                    0.55
                };
            bonus += stage_weight * if depth <= 2.0 { 0.18 } else { 0.05 };
            if features.requested_symbol_types.contains(&SymbolType::Class)
                && is_internal_definition_path(&file_path)
            {
                bonus -= stage_weight * 0.35;
            }
            if features
                .exact_identifier_case
                .as_deref()
                .is_some_and(|name| result.symbol_name.as_deref() == Some(name))
            {
                bonus += stage_weight * 0.28;
            }
        } else if file_stem(&result_filename).eq_ignore_ascii_case(identifier) {
            bonus += stage_weight * 0.18;
        } else if identifier_matches_parent_dir(identifier, &file_path) {
            bonus += stage_weight * 0.14;
        }
    }

    let stem_overlap = file_stem(&result_filename);
    if features.query_type == QueryType::NaturalLanguage
        && !features.keywords.is_empty()
        && !features.wants_config_paths
        && allow_filename_semantic_bonus
    {
        let normalized_stem = normalize_token(stem_overlap);
        if features.keywords.contains(&normalized_stem)
            || features
                .keywords
                .iter()
                .any(|keyword| shares_keyword_stem(&normalized_stem, keyword))
        {
            bonus += stage_weight * 0.6;
        }
    }

    if features.query_type == QueryType::NaturalLanguage
        && !features.keywords.is_empty()
        && !features.wants_config_paths
        && allow_filename_semantic_bonus
    {
        if let Some(symbol_name) = result.symbol_name.as_deref() {
            let symbol_bonus = symbol_keyword_bonus(symbol_name, &features.keywords);
            if symbol_bonus > 0.0 {
                bonus += stage_weight * symbol_bonus;
            }
        }
        let parent_bonus = parent_dir_keyword_bonus(&file_path, &features.keywords);
        if parent_bonus > 0.0 {
            bonus += stage_weight * parent_bonus;
        }
    }

    if !features.requested_symbol_types.is_empty()
        && result
            .symbol_type
            .is_some_and(|sym| features.requested_symbol_types.contains(&sym))
    {
        bonus += stage_weight * 0.62;
        if features
            .exact_identifier_case
            .as_deref()
            .is_some_and(|name| result.symbol_name.as_deref() == Some(name))
        {
            bonus += stage_weight * 0.2;
        }
    } else if !features.requested_symbol_types.is_empty() {
        bonus -= stage_weight
            * if features.exact_identifier_case.is_some() {
                0.9
            } else {
                0.55
            };
    }

    if features.mentions_definition && is_definition_symbol(result.symbol_type) {
        bonus += stage_weight
            * if result.symbol_name.is_some() {
                0.34
            } else {
                0.18
            };
    }

    if same_file_hits >= 2 && features.query_type == QueryType::NaturalLanguage {
        bonus += stage_weight * ((same_file_hits.min(4) - 1) as f64 * 0.08);
    }

    if !features.wants_test_paths && matches!(role, ContentClass::Test) {
        bonus -= stage_weight * 0.8;
    }
    if matches!(role, ContentClass::Archive) {
        bonus += if features.wants_archive_paths {
            stage_weight * 0.18
        } else {
            -stage_weight * 0.85
        };
    }
    if matches!(role, ContentClass::Runtime) {
        bonus += if features.wants_runtime_paths {
            stage_weight * 0.95
        } else {
            -stage_weight * 0.72
        };
    } else if features.wants_runtime_paths {
        bonus -= stage_weight * 0.24;
    }
    if !features.wants_docs_paths && matches!(role, ContentClass::Docs) {
        bonus -= stage_weight
            * if prefers_source_over_docs(features) {
                0.95
            } else {
                0.55
            };
    }
    if !features.wants_example_paths && matches!(role, ContentClass::Example | ContentClass::Bench)
    {
        bonus -= stage_weight * 0.38;
    }
    if !features.wants_compat_paths && is_compat_path(&file_path) {
        bonus -= stage_weight * 0.52;
    } else if features.wants_compat_paths && is_compat_path(&file_path) {
        bonus += stage_weight * 0.32;
    }
    if !features.wants_type_declarations && is_typescript_declaration(&file_path) {
        bonus -= stage_weight * 0.62;
    }
    if is_reexport_barrel(result) && !features.mentions_definition {
        bonus -= stage_weight * 0.85;
    }
    bonus += stage_weight * version_path_bonus(features, &file_path);
    if matches!(role, ContentClass::Generated) {
        bonus -= stage_weight
            * if features.wants_runtime_paths {
                0.18
            } else {
                0.95
            };
        if filters.include_generated == Some(false) {
            bonus -= stage_weight * 0.8;
        }
    }
    if matches!(role, ContentClass::Source | ContentClass::Config) {
        bonus += stage_weight
            * if features.query_type == QueryType::Identifier || features.path_fragment.is_some() {
                if depth <= 2.0 { 0.24 } else { 0.12 }
            } else if depth <= 2.0 {
                0.12
            } else {
                0.05
            };
    }
    if let Some(scope) = filters.scope {
        if crate::corpus::matches_scope(role, scope, filters.include_generated.unwrap_or(true)) {
            bonus += stage_weight * 0.18;
        } else {
            bonus -= stage_weight * 1.1;
        }
    }

    if features.mentions_implementation && looks_like_impl_block(result) {
        bonus += stage_weight * 0.18;
    }

    if features.query_type == QueryType::NaturalLanguage && is_public_symbol(result) {
        bonus += stage_weight * 0.05;
    }

    if prefers_structural_chunks(features) {
        bonus += stage_weight * structural_chunk_bias(result);
    }

    bonus
}

fn prefers_source_over_docs(features: &QueryFeatures) -> bool {
    features.query_type == QueryType::NaturalLanguage
        && features.query_word_count >= 4
        && !features.wants_config_paths
        && !features.wants_runtime_paths
        && !features.wants_archive_paths
}

fn requested_symbol_types(query: &str) -> Vec<SymbolType> {
    let mut symbol_types = Vec::new();
    if query.contains("trait") {
        symbol_types.push(SymbolType::Trait);
    }
    if query.contains("class") {
        symbol_types.push(SymbolType::Class);
    }
    if query.contains("interface") {
        symbol_types.push(SymbolType::Interface);
    }
    if query.contains("struct") {
        symbol_types.push(SymbolType::Struct);
    }
    if query.contains("enum") {
        symbol_types.push(SymbolType::Enum);
    }
    if query.contains("function") {
        symbol_types.push(SymbolType::Function);
    }
    if query.contains("method") {
        symbol_types.push(SymbolType::Method);
    }
    symbol_types
}

fn requested_versions(tokens: &[String]) -> Vec<String> {
    tokens
        .iter()
        .filter(|token| {
            token.len() >= 2
                && token.starts_with('v')
                && token[1..].chars().all(|ch| ch.is_ascii_digit())
        })
        .cloned()
        .collect()
}

fn file_relevance_counts(
    features: &QueryFeatures,
    results: &[SearchResult],
) -> HashMap<String, usize> {
    let mut counts = HashMap::new();
    for result in results {
        if result_matches_query_features(features, result) {
            *counts.entry(result.file_path.clone()).or_insert(0) += 1;
        }
    }
    counts
}

fn result_matches_query_features(features: &QueryFeatures, result: &SearchResult) -> bool {
    if let Some(identifier) = features.exact_identifier.as_deref() {
        if result
            .symbol_name
            .as_deref()
            .is_some_and(|name| name.eq_ignore_ascii_case(identifier))
            || file_stem(file_name(&result.file_path)).eq_ignore_ascii_case(identifier)
        {
            return true;
        }
    }

    if features.keywords.is_empty() {
        return false;
    }

    let path = result.file_path.to_ascii_lowercase();
    let content = result.content.to_ascii_lowercase();
    let symbol_stems = result
        .symbol_name
        .as_deref()
        .map(identifier_stems)
        .unwrap_or_default();
    let filename_stems = identifier_stems(file_stem(file_name(&path)));
    let parent_stems = parent_dir_stems(&path);

    features.keywords.iter().any(|keyword| {
        content.contains(keyword)
            || symbol_stems
                .iter()
                .chain(filename_stems.iter())
                .chain(parent_stems.iter())
                .any(|stem| stem == keyword || shares_keyword_stem(stem, keyword))
    })
}

/// Maximum chunks from the same file before saturation decay kicks in.
const FILE_SATURATION_THRESHOLD: usize = 1;

/// Multiplicative penalty per extra chunk from the same file beyond the threshold.
/// 0.35 means each successive same-file chunk keeps 35% of its score, pushing
/// it below results from other files in most cases.
const FILE_SATURATION_DECAY: f64 = 0.35;

fn diversify_by_file(results: Vec<SearchResult>) -> Vec<SearchResult> {
    if results.len() <= 1 {
        return results;
    }

    use std::collections::HashMap;

    let mut file_counts: HashMap<String, usize> = HashMap::new();
    let mut scored: Vec<(f64, usize, SearchResult)> = results
        .into_iter()
        .enumerate()
        .map(|(idx, result)| {
            let count = file_counts.entry(result.file_path.clone()).or_insert(0);
            *count += 1;
            let effective_score = if *count > FILE_SATURATION_THRESHOLD {
                let excess = (*count - FILE_SATURATION_THRESHOLD) as f64;
                result.score * FILE_SATURATION_DECAY.powf(excess)
            } else {
                result.score
            };
            (effective_score, idx, result)
        })
        .collect();

    scored.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });

    scored.into_iter().map(|(_, _, result)| result).collect()
}

fn stamp_rank_scores(mut results: Vec<SearchResult>) -> Vec<SearchResult> {
    let len = results.len().max(1) as f64;
    for (idx, result) in results.iter_mut().enumerate() {
        result.score = 1.0 - (idx as f64 / len);
    }
    results
}

fn looks_like_path_fragment(token: &str) -> bool {
    token.contains('/') || token.contains('\\')
}

fn clean_query_token(token: &str) -> String {
    trim_query_token(token).to_ascii_lowercase()
}

fn mentions_any(query: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| query.contains(needle))
}

fn is_query_stopword(token: &str) -> bool {
    matches!(
        token,
        "and"
            | "or"
            | "the"
            | "a"
            | "an"
            | "of"
            | "in"
            | "to"
            | "for"
            | "with"
            | "across"
            | "where"
            | "definition"
            | "definitions"
            | "configured"
            | "configuration"
    )
}

fn normalize_token(token: &str) -> String {
    let token = token.to_ascii_lowercase();
    let trimmed = token.trim_end_matches('s');
    if trimmed.len() >= 3 {
        trimmed.to_string()
    } else {
        token
    }
}

fn tokenize_path(path: &str) -> Vec<&str> {
    path.split(|ch: char| !ch.is_ascii_alphanumeric())
        .filter(|part| !part.is_empty())
        .collect()
}

fn contains_token(tokens: &[&str], expected: &[&str]) -> bool {
    tokens.iter().any(|token| expected.contains(token))
}

fn is_internal_definition_path(path: &str) -> bool {
    let tokens = tokenize_path(path);
    contains_token(&tokens, &["sansio", "internal", "bindings"])
}

fn path_matches_fragment(path: &str, fragment: &str) -> bool {
    path == fragment || path.ends_with(fragment) || path.contains(fragment)
}

fn file_stem(filename: &str) -> &str {
    filename
        .rsplit_once('.')
        .map(|(stem, _)| stem)
        .unwrap_or(filename)
}

fn identifier_matches_parent_dir(identifier: &str, path: &str) -> bool {
    parent_dir_stems(path)
        .iter()
        .any(|stem| stem.eq_ignore_ascii_case(identifier))
}

fn parent_dir_keyword_bonus(path: &str, keywords: &[String]) -> f64 {
    let stems = parent_dir_stems(path);
    if stems.is_empty() {
        return 0.0;
    }
    if stems
        .iter()
        .any(|stem| keywords.iter().any(|keyword| keyword == stem))
    {
        return 0.2;
    }
    if stems.iter().any(|stem| {
        keywords
            .iter()
            .any(|keyword| shares_keyword_stem(stem, keyword))
    }) {
        return 0.12;
    }
    0.0
}

fn parent_dir_stems(path: &str) -> Vec<String> {
    let Some((dirs, _)) = path.rsplit_once('/') else {
        return Vec::new();
    };
    dirs.split('/')
        .rev()
        .take(3)
        .flat_map(identifier_stems)
        .collect()
}

fn is_public_symbol(result: &SearchResult) -> bool {
    result.content.lines().find_map(|line| {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return None;
        }
        Some(
            trimmed.starts_with("pub ")
                || trimmed.starts_with("export ")
                || trimmed.starts_with("public ")
                || trimmed.starts_with("class ")
                || trimmed.starts_with("interface "),
        )
    }) == Some(true)
}

fn looks_like_impl_block(result: &SearchResult) -> bool {
    result
        .content
        .lines()
        .find(|line| !line.trim().is_empty())
        .is_some_and(|line| line.trim_start().starts_with("impl "))
}

fn shares_keyword_stem(left: &str, right: &str) -> bool {
    common_prefix_len(left, right) >= 6
}

fn common_prefix_len(left: &str, right: &str) -> usize {
    left.chars()
        .zip(right.chars())
        .take_while(|(l, r)| l == r)
        .count()
}

fn symbol_keyword_bonus(symbol_name: &str, keywords: &[String]) -> f64 {
    let tokens = identifier_stems(symbol_name);

    if tokens.is_empty() {
        return 0.0;
    }

    if tokens
        .iter()
        .any(|token| keywords.iter().any(|keyword| keyword == token))
    {
        return 0.5;
    }

    if tokens.iter().any(|token| {
        keywords
            .iter()
            .any(|keyword| shares_keyword_stem(token, keyword))
    }) {
        return 0.32;
    }

    0.0
}

fn identifier_stems(value: &str) -> Vec<String> {
    value
        .split(|ch: char| !ch.is_ascii_alphanumeric())
        .flat_map(split_camel_identifier)
        .map(|part| normalize_token(&part))
        .filter(|part| !part.is_empty() && !is_query_stopword(part))
        .collect()
}

fn split_camel_identifier(value: &str) -> Vec<String> {
    if value.is_empty() {
        return Vec::new();
    }

    let mut parts = Vec::new();
    let mut start = 0;
    let chars: Vec<(usize, char)> = value.char_indices().collect();
    for idx in 1..chars.len() {
        let (_, prev) = chars[idx - 1];
        let (byte_idx, current) = chars[idx];
        let boundary = (prev.is_ascii_lowercase() && current.is_ascii_uppercase())
            || (prev.is_ascii_alphabetic() && current.is_ascii_digit())
            || (prev.is_ascii_digit() && current.is_ascii_alphabetic());
        if boundary {
            parts.push(value[start..byte_idx].to_ascii_lowercase());
            start = byte_idx;
        }
    }
    parts.push(value[start..].to_ascii_lowercase());
    parts
}

fn is_definition_symbol(symbol_type: Option<SymbolType>) -> bool {
    matches!(
        symbol_type,
        Some(
            SymbolType::Class
                | SymbolType::Struct
                | SymbolType::Trait
                | SymbolType::Interface
                | SymbolType::Enum
                | SymbolType::Function
                | SymbolType::Method
                | SymbolType::Module
        )
    )
}

fn is_compat_path(path: &str) -> bool {
    let tokens = tokenize_path(path);
    contains_token(
        &tokens,
        &[
            "compat",
            "compatibility",
            "legacy",
            "shim",
            "shims",
            "polyfill",
            "polyfills",
        ],
    )
}

fn is_typescript_declaration(path: &str) -> bool {
    path.ends_with(".d.ts") || path.ends_with(".d.mts") || path.ends_with(".d.cts")
}

fn is_reexport_barrel(result: &SearchResult) -> bool {
    let filename = file_name(&result.file_path).to_ascii_lowercase();
    if !matches!(
        filename.as_str(),
        "index.ts" | "index.tsx" | "index.js" | "index.jsx" | "mod.rs"
    ) {
        return false;
    }

    let non_empty: Vec<&str> = result
        .content
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with("//"))
        .collect();
    if non_empty.is_empty() || non_empty.len() > 24 {
        return false;
    }

    let reexports = non_empty
        .iter()
        .filter(|line| {
            line.starts_with("export ")
                || line.starts_with("pub use ")
                || line.starts_with("pub mod ")
                || line.starts_with("module.exports")
        })
        .count();
    reexports > 0 && reexports * 4 >= non_empty.len() * 3
}

fn version_path_bonus(features: &QueryFeatures, path: &str) -> f64 {
    if features.requested_versions.is_empty() {
        return 0.0;
    }

    let tokens = tokenize_path(path);
    if tokens.iter().any(|token| {
        features
            .requested_versions
            .iter()
            .any(|version| version == token)
    }) {
        return 0.55;
    }

    if tokens.iter().any(|token| {
        token.len() >= 2
            && token.starts_with('v')
            && token[1..].chars().all(|ch| ch.is_ascii_digit())
    }) {
        return -0.34;
    }

    -0.08
}

fn prefers_structural_chunks(features: &QueryFeatures) -> bool {
    features.query_type == QueryType::NaturalLanguage
        && features.exact_identifier.is_none()
        && features.query_word_count >= 4
        && !features.wants_config_paths
}

fn structural_chunk_bias(result: &SearchResult) -> f64 {
    let lines = chunk_line_span(result);
    let mut bonus = 0.0;

    match result.symbol_type {
        Some(
            SymbolType::Struct | SymbolType::Class | SymbolType::Trait | SymbolType::Interface,
        ) => {
            bonus += 0.38;
        }
        Some(SymbolType::Enum | SymbolType::Module) => {
            bonus += 0.28;
        }
        Some(SymbolType::Block) if looks_like_impl_block(result) || lines >= 24 => {
            bonus += 0.24;
        }
        Some(SymbolType::Variable) => {
            bonus -= 0.45;
        }
        Some(SymbolType::Method | SymbolType::Function) if lines <= 8 => {
            bonus -= 0.32;
        }
        _ => {}
    }

    if lines <= 4 {
        bonus -= 0.2;
    } else if (12..=120).contains(&lines) {
        bonus += 0.12;
    }

    bonus
}

fn chunk_line_span(result: &SearchResult) -> u32 {
    result.line_end.saturating_sub(result.line_start) + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_result(
        file_path: &str,
        symbol_name: Option<&str>,
        symbol_type: Option<SymbolType>,
        content: &str,
    ) -> SearchResult {
        SearchResult {
            file_path: file_path.to_string(),
            line_start: 1,
            line_end: 20,
            content: content.to_string(),
            language: Language::Rust,
            score: 1.0,
            symbol_name: symbol_name.map(ToString::to_string),
            symbol_type,
        }
    }

    #[test]
    fn root_config_file_beats_nested_match() {
        let results = vec![
            make_result(
                "fuzz/Cargo.toml",
                Some("Cargo.toml"),
                Some(SymbolType::Block),
                "[package]\nname = \"fuzz\"",
            ),
            make_result(
                "Cargo.toml",
                Some("Cargo.toml"),
                Some(SymbolType::Block),
                "[workspace]\nmembers = [\"crates/vera-core\"]",
            ),
        ];

        let ranked = apply_query_ranking(
            "Cargo.toml workspace configuration",
            results,
            RankingStage::Initial,
        );

        assert_eq!(ranked[0].file_path, "Cargo.toml");
    }

    #[test]
    fn test_paths_are_demoted_for_non_test_queries() {
        let results = vec![
            make_result(
                "tests/validation.test.ts",
                Some("validation"),
                Some(SymbolType::Function),
                "export function validation() {}",
            ),
            make_result(
                "src/validation.ts",
                Some("validateRequest"),
                Some(SymbolType::Function),
                "export function validateRequest() {}",
            ),
        ];

        let ranked = apply_query_ranking(
            "request validation and schema enforcement",
            results,
            RankingStage::Initial,
        );

        assert_eq!(ranked[0].file_path, "src/validation.ts");
    }

    #[test]
    fn requested_symbol_type_gets_priority() {
        let results = vec![
            make_result(
                "src/blueprint_methods.py",
                Some("register"),
                Some(SymbolType::Method),
                "def register(self): pass",
            ),
            make_result(
                "src/blueprint.py",
                Some("Blueprint"),
                Some(SymbolType::Class),
                "class Blueprint:\n    pass",
            ),
        ];

        let ranked =
            apply_query_ranking("Blueprint class definition", results, RankingStage::Initial);

        assert_eq!(ranked[0].symbol_type, Some(SymbolType::Class));
    }

    #[test]
    fn case_exact_identifier_beats_lowercase_method() {
        let results = vec![
            make_result(
                "src/server.rs",
                Some("run"),
                Some(SymbolType::Method),
                "fn run(&self) {}",
            ),
            make_result(
                "src/run/mod.rs",
                Some("Run"),
                Some(SymbolType::Struct),
                "pub struct Run {}",
            ),
        ];

        let ranked = apply_query_ranking("Run struct definition", results, RankingStage::Initial);

        assert_eq!(ranked[0].symbol_name.as_deref(), Some("Run"));
    }

    #[test]
    fn public_class_definition_beats_internal_variant() {
        let results = vec![
            make_result(
                "src/flask/sansio/blueprints.py",
                Some("Blueprint"),
                Some(SymbolType::Class),
                "class Blueprint:\n    pass",
            ),
            make_result(
                "src/flask/blueprints.py",
                Some("Blueprint"),
                Some(SymbolType::Class),
                "class Blueprint:\n    pass",
            ),
        ];

        let ranked =
            apply_query_ranking("Blueprint class definition", results, RankingStage::Initial);

        assert_eq!(ranked[0].file_path, "src/flask/blueprints.py");
    }

    #[test]
    fn natural_language_queries_promote_file_diversity() {
        let results = vec![
            make_result(
                "src/router.ts",
                Some("register_routes"),
                Some(SymbolType::Function),
                "export function register_routes() {}",
            ),
            make_result(
                "src/router.ts",
                Some("mount_routes"),
                Some(SymbolType::Function),
                "export function mount_routes() {}",
            ),
            make_result(
                "src/app.ts",
                Some("create_app"),
                Some(SymbolType::Function),
                "export function create_app() {}",
            ),
        ];

        let ranked = apply_query_ranking(
            "Blueprint registration and route mounting",
            results,
            RankingStage::Initial,
        );

        assert_eq!(ranked[1].file_path, "src/app.ts");
    }

    #[test]
    fn explicit_path_fragment_beats_root_config_bias() {
        let results = vec![
            make_result(
                "Cargo.toml",
                Some("Cargo.toml"),
                Some(SymbolType::Block),
                "[workspace]\nmembers = [\"crates/vera-core\"]",
            ),
            make_result(
                "fuzz/Cargo.toml",
                Some("Cargo.toml"),
                Some(SymbolType::Block),
                "[package]\nname = \"fuzz\"",
            ),
        ];

        let ranked = apply_query_ranking(
            "fuzz/Cargo.toml package manifest",
            results,
            RankingStage::Initial,
        );

        assert_eq!(ranked[0].file_path, "fuzz/Cargo.toml");
    }

    #[test]
    fn testing_module_is_treated_like_test_noise() {
        let results = vec![
            make_result(
                "src/flask/testing.py",
                Some("session_transaction"),
                Some(SymbolType::Method),
                "def session_transaction(self): pass",
            ),
            make_result(
                "src/flask/sessions.py",
                Some("save_session"),
                Some(SymbolType::Method),
                "def save_session(self): pass",
            ),
        ];

        let ranked = apply_query_ranking(
            "session management and cookie handling",
            results,
            RankingStage::Initial,
        );

        assert_eq!(ranked[0].file_path, "src/flask/sessions.py");
    }

    #[test]
    fn broad_code_queries_prefer_source_over_docs() {
        let results = vec![
            make_result(
                "docs/Reference/Validation-and-Serialization.md",
                None,
                None,
                "Validation and serialization documentation.",
            ),
            make_result(
                "lib/validation.js",
                Some("validate"),
                Some(SymbolType::Function),
                "function validateRequestSchema () {}",
            ),
        ];

        let ranked = apply_query_ranking(
            "request validation and schema enforcement",
            results,
            RankingStage::Initial,
        );

        assert_eq!(ranked[0].file_path, "lib/validation.js");
    }

    #[test]
    fn archived_docs_are_demoted_for_exact_queries() {
        let results = vec![
            make_result(
                "archive/docs/hotkeys.md",
                None,
                None,
                "keybind guide and notes",
            ),
            make_result(
                "src/mod_content/hotkeys.ts",
                Some("registerHotkeys"),
                Some(SymbolType::Function),
                "export function registerHotkeys() {}",
            ),
        ];

        let ranked = apply_query_ranking("hotkeys keybind", results, RankingStage::Initial);

        assert_eq!(ranked[0].file_path, "src/mod_content/hotkeys.ts");
    }

    #[test]
    fn runtime_queries_can_prefer_runtime_extracts() {
        let results = vec![
            make_result(
                "src/mod_loader.ts",
                Some("loadMod"),
                Some(SymbolType::Function),
                "export function loadMod() {}",
            ),
            make_result(
                "/tmp/installed-game-runtime/Game.pretty.js",
                Some("loadMod"),
                Some(SymbolType::Function),
                "function loadMod() {}",
            ),
        ];

        let ranked =
            apply_query_ranking("runtime mod loader extract", results, RankingStage::Initial);

        assert_eq!(
            ranked[0].file_path,
            "/tmp/installed-game-runtime/Game.pretty.js"
        );
    }

    #[test]
    fn filename_stem_match_beats_incidental_request_helper() {
        let results = vec![
            make_result(
                "lib/handle-request.js",
                None,
                Some(SymbolType::Variable),
                "const validateSchema = require('./validation')",
            ),
            make_result(
                "lib/validation.js",
                None,
                Some(SymbolType::Variable),
                "function validate () {}",
            ),
        ];

        let ranked = apply_query_ranking(
            "request validation and schema enforcement",
            results,
            RankingStage::Initial,
        );

        assert_eq!(ranked[0].file_path, "lib/validation.js");
    }

    #[test]
    fn fuzzy_filename_stem_match_beats_unrelated_chunk() {
        let results = vec![
            make_result(
                "src/flask/sansio/blueprints.py",
                Some("BlueprintSetupState"),
                Some(SymbolType::Class),
                "class BlueprintSetupState:\n    pass",
            ),
            make_result(
                "src/flask/templating.py",
                Some("render_template"),
                Some(SymbolType::Function),
                "def render_template(template_name_or_list, **context):\n    return _render(...)",
            ),
        ];

        let ranked = apply_query_ranking(
            "template rendering pipeline",
            results,
            RankingStage::Initial,
        );

        assert_eq!(ranked[0].file_path, "src/flask/templating.py");
    }

    #[test]
    fn explicit_symbol_type_penalizes_mismatched_results() {
        let results = vec![
            make_result(
                "src/run.rs",
                Some("run"),
                Some(SymbolType::Function),
                "pub fn run() {}",
            ),
            make_result(
                "src/run/mod.rs",
                Some("Run"),
                Some(SymbolType::Struct),
                "pub struct Run {}",
            ),
        ];

        let ranked = apply_query_ranking("Run struct definition", results, RankingStage::Initial);

        assert_eq!(ranked[0].symbol_type, Some(SymbolType::Struct));
    }

    #[test]
    fn broad_intent_queries_prefer_structural_chunks() {
        let results = vec![
            SearchResult {
                file_path: "crates/ignore/src/types.rs".to_string(),
                line_start: 132,
                line_end: 137,
                content:
                    "pub fn file_type_def(&self) -> Option<&FileTypeDef> {\n    match self {\n        _ => None,\n    }\n}"
                        .to_string(),
                language: Language::Rust,
                score: 1.0,
                symbol_name: Some("file_type_def".to_string()),
                symbol_type: Some(SymbolType::Method),
            },
            SearchResult {
                file_path: "crates/ignore/src/types.rs".to_string(),
                line_start: 165,
                line_end: 181,
                content: "pub struct Types {\n    defs: Vec<FileTypeDef>,\n    selections: Vec<String>,\n    set: GlobSet,\n}"
                    .to_string(),
                language: Language::Rust,
                score: 1.0,
                symbol_name: Some("Types".to_string()),
                symbol_type: Some(SymbolType::Struct),
            },
        ];

        let ranked = apply_query_ranking(
            "file type detection and filtering",
            results,
            RankingStage::Initial,
        );

        assert_eq!(ranked[0].symbol_name.as_deref(), Some("Types"));
    }

    #[test]
    fn file_coherence_boost_promotes_repeated_relevant_file() {
        let results = vec![
            make_result(
                "src/misc.rs",
                Some("misc"),
                Some(SymbolType::Function),
                "pub fn misc() {}",
            ),
            make_result(
                "src/auth/session.rs",
                Some("renewSession"),
                Some(SymbolType::Function),
                "pub fn renew_session() {}",
            ),
            make_result(
                "src/auth/session.rs",
                Some("validateSession"),
                Some(SymbolType::Function),
                "pub fn validate_session() {}",
            ),
        ];

        let ranked = apply_query_ranking(
            "session renewal and validation flow",
            results,
            RankingStage::Initial,
        );

        assert_eq!(ranked[0].file_path, "src/auth/session.rs");
    }

    #[test]
    fn parent_directory_stem_match_beats_flat_unrelated_file() {
        let results = vec![
            make_result(
                "src/router.rs",
                Some("route"),
                Some(SymbolType::Function),
                "pub fn route() {}",
            ),
            make_result(
                "src/auth/middleware.rs",
                Some("middleware"),
                Some(SymbolType::Function),
                "pub fn middleware() {}",
            ),
        ];

        let ranked = apply_query_ranking("auth middleware routing", results, RankingStage::Initial);

        assert_eq!(ranked[0].file_path, "src/auth/middleware.rs");
    }

    #[test]
    fn compatibility_paths_are_demoted_unless_requested() {
        let results = vec![
            make_result(
                "src/compat/session.rs",
                Some("session"),
                Some(SymbolType::Function),
                "pub fn session() {}",
            ),
            make_result(
                "src/session.rs",
                Some("session"),
                Some(SymbolType::Function),
                "pub fn session() {}",
            ),
        ];

        let ranked = apply_query_ranking("session handling", results, RankingStage::Initial);
        assert_eq!(ranked[0].file_path, "src/session.rs");

        let ranked = apply_query_ranking("legacy compat session", ranked, RankingStage::Initial);
        assert_eq!(ranked[0].file_path, "src/compat/session.rs");
    }

    #[test]
    fn version_intent_prefers_matching_path() {
        let results = vec![
            make_result(
                "src/v3/router.rs",
                Some("router"),
                Some(SymbolType::Function),
                "pub fn router() {}",
            ),
            make_result(
                "src/v4/router.rs",
                Some("router"),
                Some(SymbolType::Function),
                "pub fn router() {}",
            ),
        ];

        let ranked = apply_query_ranking("v4 router", results, RankingStage::Initial);

        assert_eq!(ranked[0].file_path, "src/v4/router.rs");
    }

    #[test]
    fn declaration_files_and_reexport_barrels_are_demoted() {
        let results = vec![
            make_result(
                "src/index.ts",
                Some("index"),
                Some(SymbolType::Module),
                "export { Session } from './session'\nexport { Auth } from './auth'",
            ),
            make_result(
                "src/session.d.ts",
                Some("Session"),
                Some(SymbolType::Interface),
                "export interface Session {}",
            ),
            make_result(
                "src/session.ts",
                Some("Session"),
                Some(SymbolType::Class),
                "export class Session {}",
            ),
        ];

        let ranked = apply_query_ranking("session implementation", results, RankingStage::Initial);

        assert_eq!(ranked[0].file_path, "src/session.ts");
    }

    #[test]
    fn definition_queries_boost_symbol_definitions() {
        let results = vec![
            make_result(
                "src/parser.rs",
                Some("PARSER"),
                Some(SymbolType::Variable),
                "static PARSER: Parser = Parser::new();",
            ),
            make_result(
                "src/parser.rs",
                Some("Parser"),
                Some(SymbolType::Struct),
                "pub struct Parser {}",
            ),
        ];

        let ranked = apply_query_ranking("Parser definition", results, RankingStage::Initial);

        assert_eq!(ranked[0].symbol_type, Some(SymbolType::Struct));
    }
}
