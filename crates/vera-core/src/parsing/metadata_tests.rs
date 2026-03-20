//! Comprehensive metadata accuracy tests for chunk extraction.
//!
//! Validates that every chunk produced by `parse_and_chunk` has correct:
//! - `file_path`: repo-relative path matches input
//! - `line_start` / `line_end`: accurate to source (±1 tolerance)
//! - `language`: matches the file extension
//! - `symbol_type`: matches the AST node kind
//! - `symbol_name`: exact identifier match for named symbols
//!
//! These tests fulfil VAL-IDX-004 requirements (10+ metadata samples validated).

use crate::config::IndexingConfig;
use crate::parsing::parse_and_chunk;
use crate::types::{Language, SymbolType};

fn default_config() -> IndexingConfig {
    IndexingConfig::default()
}

/// Helper: verify that a chunk's content matches the source at the declared line range.
///
/// Extracts lines `[line_start..=line_end]` (1-based) from `source` and compares.
fn assert_content_matches_source(source: &str, chunk: &crate::types::Chunk) {
    let source_lines: Vec<&str> = source.lines().collect();
    let start = (chunk.line_start as usize).saturating_sub(1);
    let end = (chunk.line_end as usize).min(source_lines.len());
    let expected = source_lines[start..end].join("\n");
    assert_eq!(
        chunk.content,
        expected,
        "Chunk content mismatch for '{}' at {}:{}-{}\n  Expected:\n{}\n  Got:\n{}",
        chunk.symbol_name.as_deref().unwrap_or("<unnamed>"),
        chunk.file_path,
        chunk.line_start,
        chunk.line_end,
        expected,
        chunk.content,
    );
}

/// Helper: verify all common metadata invariants for a chunk.
fn assert_chunk_metadata(
    chunk: &crate::types::Chunk,
    expected_path: &str,
    expected_lang: Language,
    expected_sym_type: Option<SymbolType>,
    expected_sym_name: Option<&str>,
) {
    assert_eq!(
        chunk.file_path,
        expected_path,
        "file_path mismatch for chunk '{}'",
        chunk.symbol_name.as_deref().unwrap_or("<unnamed>")
    );
    assert_eq!(
        chunk.language,
        expected_lang,
        "language mismatch for chunk '{}'",
        chunk.symbol_name.as_deref().unwrap_or("<unnamed>")
    );
    if let Some(st) = expected_sym_type {
        assert_eq!(
            chunk.symbol_type,
            Some(st),
            "symbol_type mismatch for chunk '{}'",
            chunk.symbol_name.as_deref().unwrap_or("<unnamed>")
        );
    }
    if let Some(name) = expected_sym_name {
        assert_eq!(
            chunk.symbol_name.as_deref(),
            Some(name),
            "symbol_name mismatch: expected '{name}'"
        );
    }
    assert!(chunk.line_start >= 1, "line_start must be >= 1");
    assert!(
        chunk.line_end >= chunk.line_start,
        "line_end must be >= line_start"
    );
    assert!(!chunk.id.is_empty(), "chunk id must not be empty");
}

// =========================================================
// Sample 1: Rust — function with symbol_name exact match
// =========================================================

#[test]
fn metadata_sample_01_rust_function() {
    let source = r#"/// Computes the factorial of n.
fn factorial(n: u64) -> u64 {
    if n <= 1 { 1 } else { n * factorial(n - 1) }
}"#;
    let chunks = parse_and_chunk(source, "src/math.rs", Language::Rust, &default_config()).unwrap();

    let func = chunks
        .iter()
        .find(|c| c.symbol_name.as_deref() == Some("factorial"))
        .expect("should find function 'factorial'");

    assert_chunk_metadata(
        func,
        "src/math.rs",
        Language::Rust,
        Some(SymbolType::Function),
        Some("factorial"),
    );
    assert_content_matches_source(source, func);
}

// =========================================================
// Sample 2: Rust — struct with fields
// =========================================================

#[test]
fn metadata_sample_02_rust_struct() {
    let source = r#"/// A 2D point.
#[derive(Debug, Clone)]
pub struct Point2D {
    pub x: f64,
    pub y: f64,
}"#;
    let chunks =
        parse_and_chunk(source, "src/geometry.rs", Language::Rust, &default_config()).unwrap();

    let struc = chunks
        .iter()
        .find(|c| c.symbol_name.as_deref() == Some("Point2D"))
        .expect("should find struct 'Point2D'");

    assert_chunk_metadata(
        struc,
        "src/geometry.rs",
        Language::Rust,
        Some(SymbolType::Struct),
        Some("Point2D"),
    );
    assert_content_matches_source(source, struc);
}

// =========================================================
// Sample 3: Rust — method inside impl block
// =========================================================

#[test]
fn metadata_sample_03_rust_method() {
    let source = r#"impl Point2D {
    pub fn magnitude(&self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    pub fn translate(&mut self, dx: f64, dy: f64) {
        self.x += dx;
        self.y += dy;
    }
}"#;
    let chunks =
        parse_and_chunk(source, "src/geometry.rs", Language::Rust, &default_config()).unwrap();

    let mag = chunks
        .iter()
        .find(|c| c.symbol_name.as_deref() == Some("magnitude"))
        .expect("should find method 'magnitude'");
    assert_chunk_metadata(
        mag,
        "src/geometry.rs",
        Language::Rust,
        Some(SymbolType::Method),
        Some("magnitude"),
    );
    assert_content_matches_source(source, mag);

    let translate = chunks
        .iter()
        .find(|c| c.symbol_name.as_deref() == Some("translate"))
        .expect("should find method 'translate'");
    assert_chunk_metadata(
        translate,
        "src/geometry.rs",
        Language::Rust,
        Some(SymbolType::Method),
        Some("translate"),
    );
    assert_content_matches_source(source, translate);
}

// =========================================================
// Sample 4: Rust — enum
// =========================================================

#[test]
fn metadata_sample_04_rust_enum() {
    let source = r#"#[derive(Debug, PartialEq)]
pub enum HttpStatus {
    Ok,
    NotFound,
    InternalError,
}"#;
    let chunks = parse_and_chunk(source, "src/http.rs", Language::Rust, &default_config()).unwrap();

    let enm = chunks
        .iter()
        .find(|c| c.symbol_name.as_deref() == Some("HttpStatus"))
        .expect("should find enum 'HttpStatus'");

    assert_chunk_metadata(
        enm,
        "src/http.rs",
        Language::Rust,
        Some(SymbolType::Enum),
        Some("HttpStatus"),
    );
    assert_content_matches_source(source, enm);
}

// =========================================================
// Sample 5: Rust — trait
// =========================================================

#[test]
fn metadata_sample_05_rust_trait() {
    let source = r#"pub trait Serializable {
    fn serialize(&self) -> Vec<u8>;
    fn deserialize(data: &[u8]) -> Self;
}"#;
    let chunks =
        parse_and_chunk(source, "src/traits.rs", Language::Rust, &default_config()).unwrap();

    let trt = chunks
        .iter()
        .find(|c| c.symbol_name.as_deref() == Some("Serializable"))
        .expect("should find trait 'Serializable'");

    assert_chunk_metadata(
        trt,
        "src/traits.rs",
        Language::Rust,
        Some(SymbolType::Trait),
        Some("Serializable"),
    );
    assert_content_matches_source(source, trt);
}

// =========================================================
// Sample 6: Python — function
// =========================================================

#[test]
fn metadata_sample_06_python_function() {
    let source = r#"def calculate_distance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points."""
    dx = x2 - x1
    dy = y2 - y1
    return (dx ** 2 + dy ** 2) ** 0.5"#;
    let chunks =
        parse_and_chunk(source, "utils/math.py", Language::Python, &default_config()).unwrap();

    let func = chunks
        .iter()
        .find(|c| c.symbol_name.as_deref() == Some("calculate_distance"))
        .expect("should find function 'calculate_distance'");

    assert_chunk_metadata(
        func,
        "utils/math.py",
        Language::Python,
        Some(SymbolType::Function),
        Some("calculate_distance"),
    );
    assert_content_matches_source(source, func);
}

// =========================================================
// Sample 7: Python — class
// =========================================================

#[test]
fn metadata_sample_07_python_class() {
    let source = r#"class DatabaseConnection:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.connected = False

    def connect(self):
        self.connected = True

    def disconnect(self):
        self.connected = False"#;
    let chunks = parse_and_chunk(
        source,
        "db/connection.py",
        Language::Python,
        &default_config(),
    )
    .unwrap();

    let cls = chunks
        .iter()
        .find(|c| c.symbol_name.as_deref() == Some("DatabaseConnection"))
        .expect("should find class 'DatabaseConnection'");

    assert_chunk_metadata(
        cls,
        "db/connection.py",
        Language::Python,
        Some(SymbolType::Class),
        Some("DatabaseConnection"),
    );
    assert_content_matches_source(source, cls);
}

// =========================================================
// Sample 8: TypeScript — function
// =========================================================

#[test]
fn metadata_sample_08_typescript_function() {
    let source = r#"function parseConfig(raw: string): Config {
    const data = JSON.parse(raw);
    return {
        host: data.host || "localhost",
        port: data.port || 3000,
    };
}"#;
    let chunks = parse_and_chunk(
        source,
        "src/config.ts",
        Language::TypeScript,
        &default_config(),
    )
    .unwrap();

    let func = chunks
        .iter()
        .find(|c| c.symbol_name.as_deref() == Some("parseConfig"))
        .expect("should find function 'parseConfig'");

    assert_chunk_metadata(
        func,
        "src/config.ts",
        Language::TypeScript,
        Some(SymbolType::Function),
        Some("parseConfig"),
    );
    assert_content_matches_source(source, func);
}

// =========================================================
// Sample 9: TypeScript — interface
// =========================================================

#[test]
fn metadata_sample_09_typescript_interface() {
    let source = r#"interface ApiResponse<T> {
    data: T;
    status: number;
    message: string;
    timestamp: Date;
}"#;
    let chunks = parse_and_chunk(
        source,
        "src/types.ts",
        Language::TypeScript,
        &default_config(),
    )
    .unwrap();

    let iface = chunks
        .iter()
        .find(|c| c.symbol_name.as_deref() == Some("ApiResponse"))
        .expect("should find interface 'ApiResponse'");

    assert_chunk_metadata(
        iface,
        "src/types.ts",
        Language::TypeScript,
        Some(SymbolType::Interface),
        Some("ApiResponse"),
    );
    assert_content_matches_source(source, iface);
}

// =========================================================
// Sample 10: TypeScript — class
// =========================================================

#[test]
fn metadata_sample_10_typescript_class() {
    let source = r#"class EventEmitter {
    private listeners: Map<string, Function[]>;

    constructor() {
        this.listeners = new Map();
    }

    on(event: string, callback: Function): void {
        const cbs = this.listeners.get(event) || [];
        cbs.push(callback);
        this.listeners.set(event, cbs);
    }
}"#;
    let chunks = parse_and_chunk(
        source,
        "src/events.ts",
        Language::TypeScript,
        &default_config(),
    )
    .unwrap();

    let cls = chunks
        .iter()
        .find(|c| c.symbol_name.as_deref() == Some("EventEmitter"))
        .expect("should find class 'EventEmitter'");

    assert_chunk_metadata(
        cls,
        "src/events.ts",
        Language::TypeScript,
        Some(SymbolType::Class),
        Some("EventEmitter"),
    );
    assert_content_matches_source(source, cls);
}

// =========================================================
// Sample 11: Go — function
// =========================================================

#[test]
fn metadata_sample_11_go_function() {
    let source = r#"package handlers

func HandleRequest(w http.ResponseWriter, r *http.Request) {
    w.WriteHeader(http.StatusOK)
    w.Write([]byte("OK"))
}"#;
    let chunks = parse_and_chunk(
        source,
        "handlers/request.go",
        Language::Go,
        &default_config(),
    )
    .unwrap();

    let func = chunks
        .iter()
        .find(|c| c.symbol_name.as_deref() == Some("HandleRequest"))
        .expect("should find function 'HandleRequest'");

    assert_chunk_metadata(
        func,
        "handlers/request.go",
        Language::Go,
        Some(SymbolType::Function),
        Some("HandleRequest"),
    );
    assert_content_matches_source(source, func);
}

// =========================================================
// Sample 12: Go — struct and interface
// =========================================================

#[test]
fn metadata_sample_12_go_struct_and_interface() {
    let source = r#"package models

type User struct {
    ID    int
    Name  string
    Email string
}

type Repository interface {
    FindByID(id int) (*User, error)
    Save(user *User) error
}"#;
    let chunks =
        parse_and_chunk(source, "models/user.go", Language::Go, &default_config()).unwrap();

    let struc = chunks
        .iter()
        .find(|c| c.symbol_name.as_deref() == Some("User"))
        .expect("should find struct 'User'");
    assert_chunk_metadata(
        struc,
        "models/user.go",
        Language::Go,
        Some(SymbolType::Struct),
        Some("User"),
    );
    assert_content_matches_source(source, struc);

    let iface = chunks
        .iter()
        .find(|c| c.symbol_name.as_deref() == Some("Repository"))
        .expect("should find interface 'Repository'");
    assert_chunk_metadata(
        iface,
        "models/user.go",
        Language::Go,
        Some(SymbolType::Interface),
        Some("Repository"),
    );
    assert_content_matches_source(source, iface);
}

// =========================================================
// Sample 13: Java — class and method
// =========================================================

#[test]
fn metadata_sample_13_java_class() {
    let source = r#"public class UserService {
    private final UserRepository repo;

    public UserService(UserRepository repo) {
        this.repo = repo;
    }

    public User findUser(int id) {
        return repo.findById(id);
    }
}"#;
    let chunks = parse_and_chunk(
        source,
        "src/main/java/UserService.java",
        Language::Java,
        &default_config(),
    )
    .unwrap();

    let cls = chunks
        .iter()
        .find(|c| c.symbol_name.as_deref() == Some("UserService"))
        .expect("should find class 'UserService'");

    assert_chunk_metadata(
        cls,
        "src/main/java/UserService.java",
        Language::Java,
        Some(SymbolType::Class),
        Some("UserService"),
    );
    assert_content_matches_source(source, cls);
}

// =========================================================
// Sample 14: C — function
// =========================================================

#[test]
fn metadata_sample_14_c_function() {
    let source = r#"int binary_search(int arr[], int size, int target) {
    int low = 0, high = size - 1;
    while (low <= high) {
        int mid = low + (high - low) / 2;
        if (arr[mid] == target) return mid;
        if (arr[mid] < target) low = mid + 1;
        else high = mid - 1;
    }
    return -1;
}"#;
    let chunks = parse_and_chunk(source, "src/search.c", Language::C, &default_config()).unwrap();

    let func = chunks
        .iter()
        .find(|c| c.symbol_name.as_deref() == Some("binary_search"))
        .expect("should find function 'binary_search'");

    assert_chunk_metadata(
        func,
        "src/search.c",
        Language::C,
        Some(SymbolType::Function),
        Some("binary_search"),
    );
    assert_content_matches_source(source, func);
}

// =========================================================
// Sample 15: C++ — class
// =========================================================

#[test]
fn metadata_sample_15_cpp_class() {
    let source = r#"class LinkedList {
public:
    LinkedList() : head(nullptr), size(0) {}

    void push_front(int value) {
        Node* node = new Node(value);
        node->next = head;
        head = node;
        size++;
    }

private:
    struct Node {
        int data;
        Node* next;
        Node(int d) : data(d), next(nullptr) {}
    };
    Node* head;
    int size;
};"#;
    let chunks =
        parse_and_chunk(source, "include/list.hpp", Language::Cpp, &default_config()).unwrap();

    let cls = chunks
        .iter()
        .find(|c| c.symbol_type == Some(SymbolType::Class))
        .expect("should find class 'LinkedList'");

    assert_chunk_metadata(
        cls,
        "include/list.hpp",
        Language::Cpp,
        Some(SymbolType::Class),
        Some("LinkedList"),
    );
    assert_content_matches_source(source, cls);
}

// =========================================================
// Sample 16: JavaScript — function and class
// =========================================================

#[test]
fn metadata_sample_16_javascript_function_and_class() {
    let source = r#"function validateEmail(email) {
    const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return re.test(email);
}

class FormValidator {
    constructor(rules) {
        this.rules = rules;
    }

    validate(data) {
        return this.rules.every(rule => rule(data));
    }
}"#;
    let chunks = parse_and_chunk(
        source,
        "lib/validation.js",
        Language::JavaScript,
        &default_config(),
    )
    .unwrap();

    let func = chunks
        .iter()
        .find(|c| c.symbol_name.as_deref() == Some("validateEmail"))
        .expect("should find function 'validateEmail'");
    assert_chunk_metadata(
        func,
        "lib/validation.js",
        Language::JavaScript,
        Some(SymbolType::Function),
        Some("validateEmail"),
    );
    assert_content_matches_source(source, func);

    let cls = chunks
        .iter()
        .find(|c| c.symbol_name.as_deref() == Some("FormValidator"))
        .expect("should find class 'FormValidator'");
    assert_chunk_metadata(
        cls,
        "lib/validation.js",
        Language::JavaScript,
        Some(SymbolType::Class),
        Some("FormValidator"),
    );
    assert_content_matches_source(source, cls);
}

// =========================================================
// Sample 17: Rust — type alias and constant
// =========================================================

#[test]
fn metadata_sample_17_rust_type_alias_and_const() {
    let source = r#"type Result<T> = std::result::Result<T, AppError>;

const MAX_RETRIES: u32 = 5;"#;
    let chunks =
        parse_and_chunk(source, "src/types.rs", Language::Rust, &default_config()).unwrap();

    let ta = chunks
        .iter()
        .find(|c| c.symbol_type == Some(SymbolType::TypeAlias))
        .expect("should find type alias");
    assert_chunk_metadata(
        ta,
        "src/types.rs",
        Language::Rust,
        Some(SymbolType::TypeAlias),
        Some("Result"),
    );
    assert_content_matches_source(source, ta);

    let cst = chunks
        .iter()
        .find(|c| c.symbol_type == Some(SymbolType::Constant))
        .expect("should find constant");
    assert_chunk_metadata(
        cst,
        "src/types.rs",
        Language::Rust,
        Some(SymbolType::Constant),
        Some("MAX_RETRIES"),
    );
    assert_content_matches_source(source, cst);
}

// =========================================================
// Sample 18: Go — method
// =========================================================

#[test]
fn metadata_sample_18_go_method() {
    let source = r#"package models

func (u *User) FullName() string {
    return u.FirstName + " " + u.LastName
}"#;
    let chunks =
        parse_and_chunk(source, "models/user.go", Language::Go, &default_config()).unwrap();

    let method = chunks
        .iter()
        .find(|c| c.symbol_name.as_deref() == Some("FullName"))
        .expect("should find method 'FullName'");

    assert_chunk_metadata(
        method,
        "models/user.go",
        Language::Go,
        Some(SymbolType::Method),
        Some("FullName"),
    );
    assert_content_matches_source(source, method);
}

// =========================================================
// Tier 0 fallback metadata
// =========================================================

#[test]
fn metadata_tier0_fallback_has_correct_metadata() {
    let source = "key1 = value1\nkey2 = value2\nkey3 = value3\n";
    let chunks = parse_and_chunk(
        source,
        "config/settings.toml",
        Language::Toml,
        &default_config(),
    )
    .unwrap();

    assert!(!chunks.is_empty(), "Tier 0 should produce chunks");
    let chunk = &chunks[0];
    assert_eq!(chunk.file_path, "config/settings.toml");
    assert_eq!(chunk.language, Language::Toml);
    assert_eq!(chunk.symbol_type, Some(SymbolType::Block));
    assert!(chunk.symbol_name.is_none());
    assert_eq!(chunk.line_start, 1);
}

// =========================================================
// Cross-cutting: every chunk has all required fields
// =========================================================

#[test]
fn all_chunks_have_required_metadata_fields() {
    let test_cases: Vec<(&str, &str, Language)> = vec![
        ("fn foo() {}\nfn bar() {}", "src/lib.rs", Language::Rust),
        (
            "def foo():\n    pass\ndef bar():\n    pass",
            "lib.py",
            Language::Python,
        ),
        (
            "function foo() {}\nfunction bar() {}",
            "index.ts",
            Language::TypeScript,
        ),
        (
            "package main\nfunc Foo() {}\nfunc Bar() {}",
            "main.go",
            Language::Go,
        ),
        ("class A {\n  void m() {}\n}", "A.java", Language::Java),
        ("void foo() {}\nvoid bar() {}", "util.c", Language::C),
        ("void foo() {}\nvoid bar() {}", "util.cpp", Language::Cpp),
        (
            "function foo() {}\nfunction bar() {}",
            "app.js",
            Language::JavaScript,
        ),
        ("key = 'value'\n", "config.toml", Language::Toml),
        ("# Title\nSome text\n", "README.md", Language::Markdown),
    ];

    for (source, path, lang) in test_cases {
        let chunks = parse_and_chunk(source, path, lang, &default_config()).unwrap();

        for chunk in &chunks {
            // file_path present and matches input
            assert_eq!(chunk.file_path, path, "[{path}] file_path mismatch");
            // language matches input
            assert_eq!(chunk.language, lang, "[{path}] language mismatch");
            // line_start >= 1 (1-based)
            assert!(
                chunk.line_start >= 1,
                "[{path}] line_start must be >= 1, got {}",
                chunk.line_start,
            );
            // line_end >= line_start
            assert!(
                chunk.line_end >= chunk.line_start,
                "[{path}] line_end ({}) < line_start ({})",
                chunk.line_end,
                chunk.line_start,
            );
            // symbol_type is always Some (we always set it)
            assert!(
                chunk.symbol_type.is_some(),
                "[{path}] symbol_type should always be set"
            );
            // content is non-empty
            assert!(
                !chunk.content.trim().is_empty(),
                "[{path}] content should not be empty"
            );
            // id is non-empty
            assert!(!chunk.id.is_empty(), "[{path}] id should not be empty");
        }
    }
}

// =========================================================
// Line range accuracy: content at declared lines matches chunk content
// =========================================================

#[test]
fn line_range_accuracy_across_languages() {
    let test_cases: Vec<(&str, &str, Language)> = vec![
        (
            r#"fn add(a: i32, b: i32) -> i32 {
    a + b
}

fn sub(a: i32, b: i32) -> i32 {
    a - b
}"#,
            "math.rs",
            Language::Rust,
        ),
        (
            r#"def add(a, b):
    return a + b

def sub(a, b):
    return a - b"#,
            "math.py",
            Language::Python,
        ),
        (
            r#"function add(a: number, b: number): number {
    return a + b;
}

function sub(a: number, b: number): number {
    return a - b;
}"#,
            "math.ts",
            Language::TypeScript,
        ),
        (
            r#"package math

func Add(a, b int) int {
    return a + b
}

func Sub(a, b int) int {
    return a - b
}"#,
            "math.go",
            Language::Go,
        ),
    ];

    for (source, path, lang) in test_cases {
        let chunks = parse_and_chunk(source, path, lang, &default_config()).unwrap();

        for chunk in &chunks {
            // For every chunk, verify content matches source at declared lines
            let source_lines: Vec<&str> = source.lines().collect();
            let start = (chunk.line_start as usize).saturating_sub(1);
            let end = (chunk.line_end as usize).min(source_lines.len());

            assert!(
                start < source_lines.len(),
                "[{path}] line_start {} exceeds source length {}",
                chunk.line_start,
                source_lines.len(),
            );
            assert!(
                end <= source_lines.len(),
                "[{path}] line_end {} exceeds source length {}",
                chunk.line_end,
                source_lines.len(),
            );

            let expected = source_lines[start..end].join("\n");
            assert_eq!(
                chunk.content,
                expected,
                "[{path}] content mismatch at lines {}-{} for chunk '{}'",
                chunk.line_start,
                chunk.line_end,
                chunk.symbol_name.as_deref().unwrap_or("<unnamed>"),
            );
        }
    }
}

// =========================================================
// Symbol name exact match for all named symbols
// =========================================================

#[test]
fn symbol_name_exact_match_across_languages() {
    // Each entry: (source, path, language, expected_names)
    let test_cases: Vec<(&str, &str, Language, Vec<&str>)> = vec![
        (
            "fn alpha() {}\nfn beta_test() {}",
            "lib.rs",
            Language::Rust,
            vec!["alpha", "beta_test"],
        ),
        (
            "def process_data():\n    pass\ndef validate_input():\n    pass",
            "handlers.py",
            Language::Python,
            vec!["process_data", "validate_input"],
        ),
        (
            "function fetchUser() {}\nfunction saveUser() {}",
            "api.ts",
            Language::TypeScript,
            vec!["fetchUser", "saveUser"],
        ),
        (
            "package svc\nfunc CreateOrder() {}\nfunc CancelOrder() {}",
            "svc.go",
            Language::Go,
            vec!["CreateOrder", "CancelOrder"],
        ),
        (
            "function render() {}\nfunction hydrate() {}",
            "dom.js",
            Language::JavaScript,
            vec!["render", "hydrate"],
        ),
    ];

    for (source, path, lang, expected_names) in test_cases {
        let chunks = parse_and_chunk(source, path, lang, &default_config()).unwrap();

        for name in &expected_names {
            let found = chunks
                .iter()
                .any(|c| c.symbol_name.as_deref() == Some(*name));
            assert!(
                found,
                "[{path}] expected symbol_name '{name}' not found in chunks: {:?}",
                chunks
                    .iter()
                    .map(|c| c.symbol_name.as_deref().unwrap_or("<none>"))
                    .collect::<Vec<_>>(),
            );
        }
    }
}

// =========================================================
// Language detection matches file extension
// =========================================================

#[test]
fn language_detection_matches_extension() {
    let cases = vec![
        ("rs", Language::Rust),
        ("py", Language::Python),
        ("ts", Language::TypeScript),
        ("tsx", Language::TypeScript),
        ("js", Language::JavaScript),
        ("jsx", Language::JavaScript),
        ("go", Language::Go),
        ("java", Language::Java),
        ("c", Language::C),
        ("h", Language::C),
        ("cpp", Language::Cpp),
        ("hpp", Language::Cpp),
        ("rb", Language::Ruby),
        ("swift", Language::Swift),
        ("kt", Language::Kotlin),
        ("scala", Language::Scala),
        ("zig", Language::Zig),
        ("lua", Language::Lua),
        ("sh", Language::Bash),
        ("toml", Language::Toml),
        ("yaml", Language::Yaml),
        ("yml", Language::Yaml),
        ("json", Language::Json),
        ("md", Language::Markdown),
        ("xyz", Language::Unknown),
    ];

    for (ext, expected) in cases {
        let detected = Language::from_extension(ext);
        assert_eq!(
            detected, expected,
            "Extension '{ext}' should map to {expected:?}, got {detected:?}"
        );
    }
}

// =========================================================
// Chunks preserve file_path exactly as given (repo-relative)
// =========================================================

#[test]
fn file_path_preserved_exactly() {
    let paths = vec![
        "src/main.rs",
        "packages/core/lib/index.ts",
        "deeply/nested/path/to/file.py",
        "Cargo.toml",
        "README.md",
    ];

    for path in paths {
        let chunks =
            parse_and_chunk("content\n", path, Language::Unknown, &default_config()).unwrap();

        assert!(
            !chunks.is_empty(),
            "should produce at least one chunk for '{path}'"
        );
        for chunk in &chunks {
            assert_eq!(
                chunk.file_path, path,
                "file_path should be preserved exactly"
            );
        }
    }
}
