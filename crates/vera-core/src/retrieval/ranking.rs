//! Query-aware ranking heuristics layered on top of dense + lexical retrieval.
//!
//! These heuristics intentionally stay simple and deterministic. They target
//! recurring benchmark failures that single-vector retrieval struggles with:
//! config files at repo root, test/docs noise, symbol-type disambiguation, and
//! same-file crowding for multi-file questions.

use std::collections::HashSet;

use crate::chunk_text::file_name;
use crate::retrieval::query_classifier::{QueryType, classify_query};
use crate::types::{Language, SearchResult, SymbolType};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RankingStage {
    Initial,
    PostRerank,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FileRole {
    Source,
    Test,
    Docs,
    Example,
    Bench,
    Config,
    Generated,
    Unknown,
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
    wants_multi_file_diversity: bool,
    mentions_implementation: bool,
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
                    &[
                        "configuration",
                        "config",
                        "workspace",
                        "settings",
                    ],
                ),
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
            exact_filename,
            exact_identifier,
            keywords,
            requested_symbol_types,
            query_type,
        }
    }
}

pub(crate) fn apply_query_ranking(
    query: &str,
    results: Vec<SearchResult>,
    stage: RankingStage,
) -> Vec<SearchResult> {
    if results.len() <= 1 {
        return results;
    }

    let features = QueryFeatures::from_query(query);
    let len = results.len() as f64;
    let mut scored: Vec<(f64, usize, SearchResult)> = results
        .into_iter()
        .enumerate()
        .map(|(idx, mut result)| {
            let base_rank = 1.0 - (idx as f64 / len);
            let prior = score_prior(&features, &result, stage);
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

pub(crate) fn classify_file_role(file_path: &str, language: Language) -> FileRole {
    if language == Language::Markdown {
        return FileRole::Docs;
    }
    if language.prefers_file_chunking() {
        return FileRole::Config;
    }

    let path = file_path.to_ascii_lowercase();
    let tokens = tokenize_path(&path);

    if contains_token(
        &tokens,
        &[
            "generated",
            "dist",
            "coverage",
            "vendor",
            "binding",
            "bindings",
        ],
    ) {
        return FileRole::Generated;
    }
    if contains_token(
        &tokens,
        &[
            "test", "tests", "testing", "spec", "specs", "fixture", "fixtures",
        ],
    ) {
        return FileRole::Test;
    }
    if contains_token(&tokens, &["docs", "doc", "readme"]) {
        return FileRole::Docs;
    }
    if contains_token(&tokens, &["example", "examples", "demo", "demos", "sample"]) {
        return FileRole::Example;
    }
    if contains_token(&tokens, &["bench", "benches", "benchmark", "benchmarks"]) {
        return FileRole::Bench;
    }
    if contains_token(
        &tokens,
        &["src", "lib", "app", "apps", "crates", "packages"],
    ) {
        return FileRole::Source;
    }

    FileRole::Unknown
}

pub(crate) fn file_role_label(file_path: &str, language: Language) -> &'static str {
    match classify_file_role(file_path, language) {
        FileRole::Source => "source",
        FileRole::Test => "test",
        FileRole::Docs => "docs",
        FileRole::Example => "example",
        FileRole::Bench => "benchmark",
        FileRole::Config => "config",
        FileRole::Generated => "generated",
        FileRole::Unknown => "unknown",
    }
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

fn score_prior(features: &QueryFeatures, result: &SearchResult, stage: RankingStage) -> f64 {
    let stage_weight = match stage {
        RankingStage::Initial => 1.0,
        RankingStage::PostRerank => 0.55,
    };
    let depth = path_depth(&result.file_path) as f64;
    let role = classify_file_role(&result.file_path, result.language);
    let mut bonus = 0.0;
    let file_path = result.file_path.to_ascii_lowercase();
    let result_filename = file_name(&result.file_path).to_ascii_lowercase();
    let allow_filename_semantic_bonus = matches!(
        role,
        FileRole::Source | FileRole::Config | FileRole::Unknown
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

    if features.wants_config_paths && matches!(role, FileRole::Config) {
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
        }
    }

    let stem_overlap = file_stem(&result_filename);
    if features.query_type == QueryType::NaturalLanguage
        && !features.keywords.is_empty()
        && !features.wants_config_paths
        && allow_filename_semantic_bonus
    {
        let normalized_stem = normalize_token(stem_overlap);
        if features.keywords.contains(&normalized_stem) {
            bonus += stage_weight * 0.6;
        } else if features
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

    if !features.wants_test_paths && matches!(role, FileRole::Test) {
        bonus -= stage_weight * 0.8;
    }
    if !features.wants_docs_paths && matches!(role, FileRole::Docs) {
        bonus -= stage_weight * 0.5;
    }
    if !features.wants_example_paths && matches!(role, FileRole::Example | FileRole::Bench) {
        bonus -= stage_weight * 0.38;
    }
    if matches!(role, FileRole::Generated) {
        bonus -= stage_weight * 0.36;
    }
    if matches!(role, FileRole::Source) {
        bonus += stage_weight * if depth <= 2.0 { 0.08 } else { 0.03 };
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

fn diversify_by_file(results: Vec<SearchResult>) -> Vec<SearchResult> {
    if results.len() <= 1 {
        return results;
    }

    let mut iter = results.into_iter();
    let Some(first) = iter.next() else {
        return Vec::new();
    };

    let mut output = Vec::with_capacity(1);
    let mut remainder = Vec::new();
    let mut seen_files = HashSet::new();
    seen_files.insert(first.file_path.clone());
    output.push(first);

    for result in iter {
        if seen_files.insert(result.file_path.clone()) {
            output.push(result);
        } else {
            remainder.push(result);
        }
    }

    output.extend(remainder);
    output
}

fn stamp_rank_scores(mut results: Vec<SearchResult>) -> Vec<SearchResult> {
    let len = results.len().max(1) as f64;
    for (idx, result) in results.iter_mut().enumerate() {
        result.score = 1.0 - (idx as f64 / len);
    }
    results
}

fn looks_like_filename(token: &str) -> bool {
    matches!(
        token,
        "dockerfile" | "makefile" | "cmakelists.txt" | "nginx.conf"
    ) || token.contains('.')
}

fn looks_like_path_fragment(token: &str) -> bool {
    token.contains('/') || token.contains('\\')
}

fn looks_like_compound_identifier(token: &str) -> bool {
    token.contains('_') || token.contains("::") || token.chars().any(|ch| ch.is_ascii_uppercase())
}

fn clean_query_token(token: &str) -> String {
    trim_query_token(token).to_ascii_lowercase()
}

fn trim_query_token(token: &str) -> &str {
    token.trim_matches(|ch: char| {
        !ch.is_ascii_alphanumeric() && !matches!(ch, '.' | '_' | '-' | '/')
    })
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

fn path_depth(path: &str) -> usize {
    path.matches('/').count() + path.matches('\\').count()
}

fn file_stem(filename: &str) -> &str {
    filename
        .rsplit_once('.')
        .map(|(stem, _)| stem)
        .unwrap_or(filename)
}

fn is_public_symbol(result: &SearchResult) -> bool {
    result.content.lines().find_map(|line| {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return None;
        }
        Some(
            trimmed.starts_with("pub ")
                || trimmed.starts_with("pub(")
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
    let tokens: Vec<String> = symbol_name
        .split(|ch: char| !ch.is_ascii_alphanumeric())
        .filter(|part| !part.is_empty())
        .map(normalize_token)
        .collect();

    if tokens.is_empty() {
        return 0.0;
    }

    if tokens
        .iter()
        .any(|token| keywords.iter().any(|keyword| keyword == token))
    {
        return 0.5;
    }

    if tokens
        .iter()
        .any(|token| keywords.iter().any(|keyword| shares_keyword_stem(token, keyword)))
    {
        return 0.32;
    }

    0.0
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
        Some(SymbolType::Struct | SymbolType::Class | SymbolType::Trait | SymbolType::Interface) => {
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

        let ranked =
            apply_query_ranking("template rendering pipeline", results, RankingStage::Initial);

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
}
