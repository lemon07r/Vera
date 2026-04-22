//! Extract explicit type relations from declaration chunks.
//!
//! This module focuses on explicit declarations only: trait/interface
//! conformance, class inheritance, mixins, and similar header-level
//! relationships. It intentionally does not attempt semantic inference for
//! languages where implementations are implicit (for example Go interfaces).

use std::collections::HashSet;

use crate::types::{Chunk, Language, SymbolType};

use super::signatures;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TypeRelationKind {
    Conforms,
    Extends,
}

impl TypeRelationKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Conforms => "conforms",
            Self::Extends => "extends",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "conforms" => Some(Self::Conforms),
            "extends" => Some(Self::Extends),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawTypeRelation {
    pub owner: String,
    pub target: String,
    pub line: u32,
    pub kind: TypeRelationKind,
}

pub fn extract_type_relations(chunks: &[Chunk]) -> Vec<RawTypeRelation> {
    let mut relations = Vec::new();
    let mut seen = HashSet::new();

    for chunk in chunks {
        for relation in extract_chunk_relations(chunk) {
            let key = format!(
                "{}:{}:{}:{}",
                relation.line,
                relation.owner.to_ascii_lowercase(),
                relation.target.to_ascii_lowercase(),
                relation.kind.as_str()
            );
            if seen.insert(key) {
                relations.push(relation);
            }
        }
    }

    relations
}

fn extract_chunk_relations(chunk: &Chunk) -> Vec<RawTypeRelation> {
    let header = relation_header(chunk);
    if header.is_empty() {
        return Vec::new();
    }

    match chunk.language {
        Language::Rust => rust_relations(chunk, &header),
        Language::ObjectiveC => {
            let Some(owner) = relation_owner(chunk) else {
                return Vec::new();
            };
            objective_c_relations(chunk, &owner)
        }
        Language::Haskell => {
            let Some(owner) = relation_owner(chunk) else {
                return Vec::new();
            };
            haskell_relations(chunk, &owner)
        }
        Language::Ruby => {
            let Some(owner) = relation_owner(chunk) else {
                return Vec::new();
            };
            ruby_relations(chunk, &header, &owner)
        }
        Language::CSharp | Language::Kotlin | Language::Swift | Language::Cpp | Language::Dart => {
            let Some(owner) = relation_owner(chunk) else {
                return Vec::new();
            };
            colon_relations(chunk, &header, &owner)
        }
        _ => {
            let Some(owner) = relation_owner(chunk) else {
                return Vec::new();
            };
            generic_relations(chunk, &header, &owner)
        }
    }
}

fn relation_owner(chunk: &Chunk) -> Option<String> {
    let name = chunk.symbol_name.as_deref()?.trim();
    if name.is_empty() {
        return None;
    }
    if name.starts_with("impl ") {
        return None;
    }
    simple_name(name)
}

fn relation_header(chunk: &Chunk) -> String {
    let raw = if chunk.language == Language::ObjectiveC
        || (chunk.language == Language::Rust
            && chunk
                .symbol_name
                .as_deref()
                .is_some_and(|name| name.starts_with("impl ")))
    {
        chunk
            .content
            .lines()
            .map(str::trim)
            .find(|line| !line.is_empty())
            .unwrap_or_default()
            .to_string()
    } else {
        signatures::extract_signature(&chunk.content, chunk.language)
    };
    normalize_header(&raw)
}

fn normalize_header(value: &str) -> String {
    value
        .replace("{ ... }", "")
        .replace(" = ...", "")
        .replace(" ...", "")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn rust_relations(chunk: &Chunk, header: &str) -> Vec<RawTypeRelation> {
    if let Some(body) = header.strip_prefix("trait ") {
        let Some(owner) = relation_owner(chunk) else {
            return Vec::new();
        };
        if let Some(idx) = top_level_char_index(body, ':') {
            let clause = body[idx + 1..].trim();
            return build_relations(
                chunk.line_start,
                &owner,
                split_targets(clause, &['+', ',']),
                TypeRelationKind::Extends,
            );
        }
        return Vec::new();
    }

    if !header.starts_with("impl ") {
        return Vec::new();
    }

    let mut body = header.trim_start_matches("impl").trim();
    if body.starts_with('<') {
        if let Some(end) = matching_delimiter(body, 0, '<', '>') {
            body = body[end + 1..].trim();
        }
    }

    let Some(idx) = top_level_keyword_index(body, " for ") else {
        return Vec::new();
    };
    let target = body[..idx].trim();
    let owner = body[idx + " for ".len()..]
        .split(" where ")
        .next()
        .unwrap_or("")
        .trim();
    let Some(owner) = simple_name(owner) else {
        return Vec::new();
    };
    let Some(target) = simple_name(target) else {
        return Vec::new();
    };
    vec![RawTypeRelation {
        owner,
        target,
        line: chunk.line_start,
        kind: TypeRelationKind::Conforms,
    }]
}

fn generic_relations(chunk: &Chunk, header: &str, owner: &str) -> Vec<RawTypeRelation> {
    let mut relations = Vec::new();

    if let Some(clause) = clause_after(header, " extends ", &[" implements ", " with "]) {
        relations.extend(build_relations(
            chunk.line_start,
            owner,
            split_targets(&clause, &[',']),
            TypeRelationKind::Extends,
        ));
    }

    if let Some(clause) = clause_after(header, " implements ", &[" extends ", " with "]) {
        relations.extend(build_relations(
            chunk.line_start,
            owner,
            split_targets(&clause, &[',']),
            TypeRelationKind::Conforms,
        ));
    }

    if let Some(clause) = clause_after(header, " with ", &[" implements "]) {
        relations.extend(build_relations(
            chunk.line_start,
            owner,
            split_targets_with_keyword(&clause, " with "),
            TypeRelationKind::Conforms,
        ));
    }

    relations
}

fn colon_relations(chunk: &Chunk, header: &str, owner: &str) -> Vec<RawTypeRelation> {
    let Some(idx) = top_level_char_index(header, ':') else {
        return generic_relations(chunk, header, owner);
    };

    let clause = header[idx + 1..].trim();
    if clause.is_empty() {
        return generic_relations(chunk, header, owner);
    }

    let targets = split_targets(clause, &[',']);
    if targets.is_empty() {
        return generic_relations(chunk, header, owner);
    }

    let mut relations = Vec::new();
    for (index, target) in targets.into_iter().enumerate() {
        let kind = colon_relation_kind(chunk.symbol_type, index);
        if let Some(target) = simple_name(&target) {
            relations.push(RawTypeRelation {
                owner: owner.to_string(),
                target,
                line: chunk.line_start,
                kind,
            });
        }
    }

    relations.extend(generic_relations(chunk, header, owner));
    dedupe_relations(relations)
}

fn colon_relation_kind(symbol_type: Option<SymbolType>, index: usize) -> TypeRelationKind {
    match symbol_type {
        Some(SymbolType::Interface | SymbolType::Trait) => TypeRelationKind::Extends,
        Some(SymbolType::Struct | SymbolType::Enum) => TypeRelationKind::Conforms,
        Some(SymbolType::Class) if index == 0 => TypeRelationKind::Extends,
        _ => TypeRelationKind::Conforms,
    }
}

fn objective_c_relations(chunk: &Chunk, owner: &str) -> Vec<RawTypeRelation> {
    let header = chunk
        .content
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or_default();
    if header.is_empty() {
        return Vec::new();
    }

    let mut relations = Vec::new();
    if let Some(idx) = top_level_char_index(header, ':') {
        let rest = header[idx + 1..].trim();
        let base = rest.split('<').next().unwrap_or("").trim();
        if let Some(target) = simple_name(base) {
            relations.push(RawTypeRelation {
                owner: owner.to_string(),
                target,
                line: chunk.line_start,
                kind: TypeRelationKind::Extends,
            });
        }
    }

    if let Some(start) = header.find('<') {
        if let Some(end) = header[start + 1..].find('>') {
            let protocols = &header[start + 1..start + 1 + end];
            relations.extend(build_relations(
                chunk.line_start,
                owner,
                split_targets(protocols, &[',']),
                TypeRelationKind::Conforms,
            ));
        }
    }

    relations
}

fn ruby_relations(chunk: &Chunk, header: &str, owner: &str) -> Vec<RawTypeRelation> {
    let Some(idx) = top_level_keyword_index(header, " < ") else {
        return Vec::new();
    };
    build_relations(
        chunk.line_start,
        owner,
        vec![header[idx + " < ".len()..].to_string()],
        TypeRelationKind::Extends,
    )
}

fn haskell_relations(chunk: &Chunk, owner: &str) -> Vec<RawTypeRelation> {
    let header = chunk
        .content
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or_default();
    let Some(body) = header.strip_prefix("instance ") else {
        return Vec::new();
    };
    let body = body
        .split(" where")
        .next()
        .unwrap_or(body)
        .split(" => ")
        .last()
        .unwrap_or(body)
        .trim();
    let Some(target) = body.split_whitespace().next().and_then(simple_name) else {
        return Vec::new();
    };
    vec![RawTypeRelation {
        owner: owner.to_string(),
        target,
        line: chunk.line_start,
        kind: TypeRelationKind::Conforms,
    }]
}

fn build_relations(
    line: u32,
    owner: &str,
    targets: Vec<String>,
    kind: TypeRelationKind,
) -> Vec<RawTypeRelation> {
    targets
        .into_iter()
        .filter_map(|target| {
            let target = simple_name(&target)?;
            (target != owner).then_some(RawTypeRelation {
                owner: owner.to_string(),
                target,
                line,
                kind,
            })
        })
        .collect()
}

fn dedupe_relations(relations: Vec<RawTypeRelation>) -> Vec<RawTypeRelation> {
    let mut deduped = Vec::new();
    let mut seen = HashSet::new();
    for relation in relations {
        let key = format!(
            "{}:{}:{}",
            relation.owner.to_ascii_lowercase(),
            relation.target.to_ascii_lowercase(),
            relation.kind.as_str()
        );
        if seen.insert(key) {
            deduped.push(relation);
        }
    }
    deduped
}

fn clause_after(header: &str, keyword: &str, stop_keywords: &[&str]) -> Option<String> {
    let start = top_level_keyword_index(header, keyword)? + keyword.len();
    let rest = &header[start..];
    let mut end = rest.len();
    for stop in stop_keywords {
        if let Some(idx) = top_level_keyword_index(rest, stop) {
            end = end.min(idx);
        }
    }
    Some(rest[..end].trim().to_string())
}

fn split_targets(text: &str, separators: &[char]) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut angle = 0i32;
    let mut paren = 0i32;
    let mut bracket = 0i32;
    let mut brace = 0i32;

    for ch in text.chars() {
        match ch {
            '<' => angle += 1,
            '>' => angle = (angle - 1).max(0),
            '(' => paren += 1,
            ')' => paren = (paren - 1).max(0),
            '[' => bracket += 1,
            ']' => bracket = (bracket - 1).max(0),
            '{' => brace += 1,
            '}' => brace = (brace - 1).max(0),
            _ => {}
        }
        if angle == 0 && paren == 0 && bracket == 0 && brace == 0 && separators.contains(&ch) {
            let part = current.trim();
            if !part.is_empty() {
                parts.push(part.to_string());
            }
            current.clear();
            continue;
        }
        current.push(ch);
    }

    let tail = current.trim();
    if !tail.is_empty() {
        parts.push(tail.to_string());
    }
    parts
}

fn split_targets_with_keyword(text: &str, keyword: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut start = 0usize;
    while let Some(idx) = top_level_keyword_index(&text[start..], keyword) {
        let split = start + idx;
        let part = text[start..split].trim();
        if !part.is_empty() {
            parts.extend(split_targets(part, &[',']));
        }
        start = split + keyword.len();
    }
    let tail = text[start..].trim();
    if !tail.is_empty() {
        parts.extend(split_targets(tail, &[',']));
    }
    parts
}

fn top_level_keyword_index(text: &str, keyword: &str) -> Option<usize> {
    if keyword.is_empty() || text.len() < keyword.len() {
        return None;
    }

    let mut angle = 0i32;
    let mut paren = 0i32;
    let mut bracket = 0i32;
    let mut brace = 0i32;
    let bytes = text.as_bytes();
    let needle = keyword.as_bytes();
    let mut idx = 0usize;

    while idx + needle.len() <= bytes.len() {
        match bytes[idx] as char {
            '<' => angle += 1,
            '>' => angle = (angle - 1).max(0),
            '(' => paren += 1,
            ')' => paren = (paren - 1).max(0),
            '[' => bracket += 1,
            ']' => bracket = (bracket - 1).max(0),
            '{' => brace += 1,
            '}' => brace = (brace - 1).max(0),
            _ => {}
        }
        if angle == 0
            && paren == 0
            && bracket == 0
            && brace == 0
            && &bytes[idx..idx + needle.len()] == needle
        {
            return Some(idx);
        }
        idx += 1;
    }
    None
}

fn top_level_char_index(text: &str, target: char) -> Option<usize> {
    let mut angle = 0i32;
    let mut paren = 0i32;
    let mut bracket = 0i32;
    let mut brace = 0i32;
    for (idx, ch) in text.char_indices() {
        match ch {
            '<' => angle += 1,
            '>' => angle = (angle - 1).max(0),
            '(' => paren += 1,
            ')' => paren = (paren - 1).max(0),
            '[' => bracket += 1,
            ']' => bracket = (bracket - 1).max(0),
            '{' => brace += 1,
            '}' => brace = (brace - 1).max(0),
            _ => {}
        }
        if ch == target && angle == 0 && paren == 0 && bracket == 0 && brace == 0 {
            return Some(idx);
        }
    }
    None
}

fn matching_delimiter(text: &str, start: usize, open: char, close: char) -> Option<usize> {
    let mut depth = 0i32;
    for (idx, ch) in text.char_indices().skip(start) {
        if ch == open {
            depth += 1;
        } else if ch == close {
            depth -= 1;
            if depth == 0 {
                return Some(idx);
            }
        }
    }
    None
}

fn simple_name(text: &str) -> Option<String> {
    let cleaned = text
        .trim()
        .trim_end_matches(';')
        .trim_end_matches(',')
        .trim_end_matches('{')
        .trim_end_matches('}')
        .trim_end_matches(')')
        .trim_end_matches('(')
        .trim();
    if cleaned.is_empty() {
        return None;
    }

    let mut end = None;
    for (idx, ch) in cleaned.char_indices().rev() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            end = Some(idx + ch.len_utf8());
            break;
        }
    }
    let end = end?;

    let mut start = end;
    for (idx, ch) in cleaned[..end].char_indices().rev() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            start = idx;
        } else {
            break;
        }
    }

    let name = cleaned[start..end].trim();
    (!name.is_empty()).then_some(name.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn class_chunk(language: Language, symbol_name: &str, content: &str) -> Chunk {
        Chunk {
            id: "test:0".to_string(),
            file_path: "test".to_string(),
            line_start: 1,
            line_end: content.lines().count() as u32,
            content: content.to_string(),
            language,
            symbol_type: Some(SymbolType::Class),
            symbol_name: Some(symbol_name.to_string()),
        }
    }

    #[test]
    fn rust_trait_impl_relations() {
        let chunk = Chunk {
            id: "test:0".to_string(),
            file_path: "lib.rs".to_string(),
            line_start: 1,
            line_end: 3,
            content: "impl std::fmt::Display for User {\n    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { todo!() }\n}\n".to_string(),
            language: Language::Rust,
            symbol_type: Some(SymbolType::Block),
            symbol_name: Some("impl std::fmt::Display for User".to_string()),
        };

        let relations = extract_type_relations(&[chunk]);
        assert_eq!(relations.len(), 1);
        assert_eq!(relations[0].owner, "User");
        assert_eq!(relations[0].target, "Display");
        assert_eq!(relations[0].kind, TypeRelationKind::Conforms);
    }

    #[test]
    fn typescript_class_relations() {
        let chunk = class_chunk(
            Language::TypeScript,
            "Repo",
            "class Repo extends BaseRepo implements Loader, Saver {\n  run() {}\n}\n",
        );

        let relations = extract_type_relations(&[chunk]);
        assert_eq!(relations.len(), 3);
        assert!(
            relations
                .iter()
                .any(|r| r.target == "BaseRepo" && r.kind == TypeRelationKind::Extends)
        );
        assert!(
            relations
                .iter()
                .any(|r| r.target == "Loader" && r.kind == TypeRelationKind::Conforms)
        );
        assert!(
            relations
                .iter()
                .any(|r| r.target == "Saver" && r.kind == TypeRelationKind::Conforms)
        );
    }

    #[test]
    fn csharp_colon_relations() {
        let chunk = class_chunk(
            Language::CSharp,
            "Worker",
            "public class Worker : BackgroundService, IDisposable, ILoggerProvider {\n}\n",
        );

        let relations = extract_type_relations(&[chunk]);
        assert_eq!(relations.len(), 3);
        assert!(
            relations
                .iter()
                .any(|r| r.target == "BackgroundService" && r.kind == TypeRelationKind::Extends)
        );
        assert!(
            relations
                .iter()
                .filter(|r| r.kind == TypeRelationKind::Conforms)
                .count()
                == 2
        );
    }

    #[test]
    fn objective_c_relations() {
        let chunk = class_chunk(
            Language::ObjectiveC,
            "Calculator",
            "@interface Calculator : NSObject <NSCopying, NSSecureCoding>\n- (NSInteger)add:(NSInteger)a b:(NSInteger)b;\n@end\n",
        );

        let relations = extract_type_relations(&[chunk]);
        assert_eq!(relations.len(), 3);
        assert!(
            relations
                .iter()
                .any(|r| r.target == "NSObject" && r.kind == TypeRelationKind::Extends)
        );
        assert!(
            relations
                .iter()
                .any(|r| r.target == "NSCopying" && r.kind == TypeRelationKind::Conforms)
        );
        assert!(
            relations
                .iter()
                .any(|r| r.target == "NSSecureCoding" && r.kind == TypeRelationKind::Conforms)
        );
    }

    #[test]
    fn implicit_languages_do_not_guess_relations() {
        let chunk = Chunk {
            id: "test:0".to_string(),
            file_path: "repo.go".to_string(),
            line_start: 1,
            line_end: 5,
            content: "type Repository interface {\n    Save() error\n}\n".to_string(),
            language: Language::Go,
            symbol_type: Some(SymbolType::Interface),
            symbol_name: Some("Repository".to_string()),
        };

        assert!(extract_type_relations(&[chunk]).is_empty());
    }
}
