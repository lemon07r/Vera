//! Shared types used across Vera's core modules.

use serde::{Deserialize, Serialize};

/// Filters that can be applied to search results.
///
/// All filters are optional. When set, they restrict results to only those
/// matching all specified criteria (AND semantics).
#[derive(Debug, Clone, Default)]
pub struct SearchFilters {
    /// Filter by programming language (case-insensitive match).
    pub language: Option<String>,
    /// Filter by file path glob pattern (e.g., `src/**/*.rs`).
    pub path_glob: Option<String>,
    /// Filter by symbol type (case-insensitive match).
    pub symbol_type: Option<String>,
}

impl SearchFilters {
    /// Returns true if no filters are set.
    pub fn is_empty(&self) -> bool {
        self.language.is_none() && self.path_glob.is_none() && self.symbol_type.is_none()
    }

    /// Check whether a search result matches all active filters.
    pub fn matches(&self, result: &SearchResult) -> bool {
        // Language filter (case-insensitive).
        if let Some(ref lang) = self.language {
            if !result.language.to_string().eq_ignore_ascii_case(lang) {
                return false;
            }
        }

        // Path glob filter.
        if let Some(ref pattern) = self.path_glob {
            if !glob_matches(pattern, &result.file_path) {
                return false;
            }
        }

        // Symbol type filter (case-insensitive).
        if let Some(ref stype) = self.symbol_type {
            match &result.symbol_type {
                Some(st) => {
                    if !st.to_string().eq_ignore_ascii_case(stype) {
                        return false;
                    }
                }
                None => return false,
            }
        }

        true
    }
}

/// Simple glob matching supporting `*` (any segment) and `**` (any path).
///
/// Supports common patterns: `*.rs`, `src/**/*.ts`, `**/test_*`.
/// Does not support character classes or brace expansion.
fn glob_matches(pattern: &str, path: &str) -> bool {
    // Normalize separators.
    let pattern = pattern.replace('\\', "/");
    let path = path.replace('\\', "/");

    glob_match_recursive(&pattern, &path)
}

/// Recursive glob matching helper.
fn glob_match_recursive(pattern: &str, text: &str) -> bool {
    // Handle standalone `**` — matches everything (any path, any depth).
    if pattern == "**" {
        return true;
    }

    // Handle `**` patterns (match any path segments).
    if let Some(rest) = pattern.strip_prefix("**/") {
        // `**/X` matches X at any depth.
        if glob_match_recursive(rest, text) {
            return true;
        }
        // Try skipping path segments.
        for (i, _) in text.char_indices() {
            if text.as_bytes().get(i) == Some(&b'/') && glob_match_recursive(rest, &text[i + 1..]) {
                return true;
            }
        }
        return false;
    }

    if pattern.is_empty() && text.is_empty() {
        return true;
    }
    if pattern.is_empty() {
        return false;
    }

    // Handle `*` within a segment (matches anything except `/`).
    if let Some(rest) = pattern.strip_prefix('*') {
        // Try matching * against 0..n characters (not crossing `/`).
        if glob_match_recursive(rest, text) {
            return true;
        }
        for (i, ch) in text.char_indices() {
            if ch == '/' {
                break;
            }
            if glob_match_recursive(rest, &text[i + 1..]) {
                return true;
            }
        }
        return false;
    }

    // Match literal characters.
    let mut p_chars = pattern.chars();
    let mut t_chars = text.chars();
    if let (Some(pc), Some(tc)) = (p_chars.next(), t_chars.next()) {
        if pc == tc {
            return glob_match_recursive(p_chars.as_str(), t_chars.as_str());
        }
    }

    false
}

/// A chunk of source code extracted from a parsed file.
///
/// This is the fundamental unit that gets indexed, embedded, and retrieved.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    /// Unique identifier for this chunk.
    pub id: String,
    /// Repository-relative file path.
    pub file_path: String,
    /// 1-based start line in the source file.
    pub line_start: u32,
    /// 1-based end line in the source file (inclusive).
    pub line_end: u32,
    /// The actual source code content of this chunk.
    pub content: String,
    /// Detected programming language.
    pub language: Language,
    /// Type of symbol this chunk represents (if any).
    pub symbol_type: Option<SymbolType>,
    /// Name of the symbol (if applicable).
    pub symbol_name: Option<String>,
}

/// Programming language of a source file or chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Language {
    Rust,
    TypeScript,
    JavaScript,
    Python,
    Go,
    Java,
    C,
    Cpp,
    Ruby,
    Swift,
    Kotlin,
    Scala,
    Zig,
    Lua,
    Bash,
    CSharp,
    Php,
    Haskell,
    Elixir,
    Dart,
    Sql,
    Hcl,
    Protobuf,
    /// Structural / config / web formats (Tier 1B).
    Html,
    Css,
    Scss,
    Vue,
    GraphQl,
    CMake,
    Dockerfile,
    Xml,
    /// Tier 2A code languages.
    ObjectiveC,
    Perl,
    Julia,
    Nix,
    OCaml,
    Groovy,
    Clojure,
    CommonLisp,
    Erlang,
    FSharp,
    Fortran,
    PowerShell,
    R,
    /// Tier 2A code languages batch 2.
    Matlab,
    DLang,
    Fish,
    Zsh,
    Luau,
    Scheme,
    Racket,
    Elm,
    Glsl,
    Hlsl,
    /// Tier 2B structural/config/frontend languages.
    Svelte,
    Astro,
    Makefile,
    Ini,
    Nginx,
    Prisma,
    /// Data / config formats (Tier 0 — no tree-sitter grammar).
    Toml,
    Yaml,
    Json,
    Markdown,
    /// Fallback for unrecognized file types (Tier 0).
    Unknown,
}

impl Language {
    /// Detect language from a full filename (for extensionless files like Dockerfile, CMakeLists.txt, Makefile).
    pub fn from_filename(name: &str) -> Option<Self> {
        match name {
            "Dockerfile" | "dockerfile" => Some(Self::Dockerfile),
            "CMakeLists.txt" => Some(Self::CMake),
            "Makefile" | "makefile" | "GNUmakefile" => Some(Self::Makefile),
            "nginx.conf" => Some(Self::Nginx),
            _ => None,
        }
    }

    /// Detect language from a file extension.
    pub fn from_extension(ext: &str) -> Self {
        match ext.to_lowercase().as_str() {
            "rs" => Self::Rust,
            "ts" | "tsx" => Self::TypeScript,
            "js" | "jsx" | "mjs" | "cjs" => Self::JavaScript,
            "py" | "pyi" => Self::Python,
            "go" => Self::Go,
            "java" => Self::Java,
            "c" | "h" => Self::C,
            "cpp" | "cc" | "cxx" | "hpp" | "hxx" | "hh" => Self::Cpp,
            "rb" => Self::Ruby,
            "swift" => Self::Swift,
            "kt" | "kts" => Self::Kotlin,
            "scala" | "sc" => Self::Scala,
            "zig" => Self::Zig,
            "lua" => Self::Lua,
            "sh" | "bash" => Self::Bash,
            "cs" => Self::CSharp,
            "php" => Self::Php,
            "hs" => Self::Haskell,
            "ex" | "exs" => Self::Elixir,
            "dart" => Self::Dart,
            "sql" => Self::Sql,
            "tf" | "hcl" => Self::Hcl,
            "proto" => Self::Protobuf,
            "html" | "htm" => Self::Html,
            "css" => Self::Css,
            "scss" => Self::Scss,
            "vue" => Self::Vue,
            "graphql" | "gql" => Self::GraphQl,
            "cmake" => Self::CMake,
            "xml" | "xsl" | "xsd" | "svg" => Self::Xml,
            "m" | "mm" => Self::ObjectiveC,
            "pl" | "pm" => Self::Perl,
            "jl" => Self::Julia,
            "nix" => Self::Nix,
            "ml" | "mli" => Self::OCaml,
            "groovy" => Self::Groovy,
            "clj" | "cljs" | "cljc" => Self::Clojure,
            "lisp" | "cl" | "lsp" => Self::CommonLisp,
            "erl" | "hrl" => Self::Erlang,
            "fs" | "fsi" | "fsx" => Self::FSharp,
            "f" | "f90" | "f95" => Self::Fortran,
            "ps1" | "psm1" => Self::PowerShell,
            "r" => Self::R,
            "mlx" => Self::Matlab,
            "d" | "di" => Self::DLang,
            "fish" => Self::Fish,
            "zsh" => Self::Zsh,
            "luau" => Self::Luau,
            "scm" | "ss" => Self::Scheme,
            "rkt" => Self::Racket,
            "elm" => Self::Elm,
            "glsl" | "vert" | "frag" | "geom" | "comp" | "tesc" | "tese" => Self::Glsl,
            "hlsl" | "hlsli" | "fx" => Self::Hlsl,
            "svelte" => Self::Svelte,
            "astro" => Self::Astro,
            "ini" | "cfg" | "conf" => Self::Ini,
            "nginx" => Self::Nginx,
            "prisma" => Self::Prisma,
            "toml" => Self::Toml,
            "yaml" | "yml" => Self::Yaml,
            "json" => Self::Json,
            "md" | "markdown" => Self::Markdown,
            _ => Self::Unknown,
        }
    }
}

impl std::fmt::Display for Language {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::Rust => "rust",
            Self::TypeScript => "typescript",
            Self::JavaScript => "javascript",
            Self::Python => "python",
            Self::Go => "go",
            Self::Java => "java",
            Self::C => "c",
            Self::Cpp => "cpp",
            Self::Ruby => "ruby",
            Self::Swift => "swift",
            Self::Kotlin => "kotlin",
            Self::Scala => "scala",
            Self::Zig => "zig",
            Self::Lua => "lua",
            Self::Bash => "bash",
            Self::CSharp => "csharp",
            Self::Php => "php",
            Self::Haskell => "haskell",
            Self::Elixir => "elixir",
            Self::Dart => "dart",
            Self::Sql => "sql",
            Self::Hcl => "hcl",
            Self::Protobuf => "protobuf",
            Self::Html => "html",
            Self::Css => "css",
            Self::Scss => "scss",
            Self::Vue => "vue",
            Self::GraphQl => "graphql",
            Self::CMake => "cmake",
            Self::Dockerfile => "dockerfile",
            Self::Xml => "xml",
            Self::ObjectiveC => "objectivec",
            Self::Perl => "perl",
            Self::Julia => "julia",
            Self::Nix => "nix",
            Self::OCaml => "ocaml",
            Self::Groovy => "groovy",
            Self::Clojure => "clojure",
            Self::CommonLisp => "commonlisp",
            Self::Erlang => "erlang",
            Self::FSharp => "fsharp",
            Self::Fortran => "fortran",
            Self::PowerShell => "powershell",
            Self::R => "r",
            Self::Matlab => "matlab",
            Self::DLang => "d",
            Self::Fish => "fish",
            Self::Zsh => "zsh",
            Self::Luau => "luau",
            Self::Scheme => "scheme",
            Self::Racket => "racket",
            Self::Elm => "elm",
            Self::Glsl => "glsl",
            Self::Hlsl => "hlsl",
            Self::Svelte => "svelte",
            Self::Astro => "astro",
            Self::Makefile => "makefile",
            Self::Ini => "ini",
            Self::Nginx => "nginx",
            Self::Prisma => "prisma",
            Self::Toml => "toml",
            Self::Yaml => "yaml",
            Self::Json => "json",
            Self::Markdown => "markdown",
            Self::Unknown => "unknown",
        };
        write!(f, "{name}")
    }
}

/// Type of symbol extracted from source code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SymbolType {
    Function,
    Method,
    Class,
    Struct,
    Enum,
    Trait,
    Interface,
    TypeAlias,
    Constant,
    Variable,
    Module,
    /// A fallback chunk not aligned to a specific symbol.
    Block,
}

impl std::fmt::Display for SymbolType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::Function => "function",
            Self::Method => "method",
            Self::Class => "class",
            Self::Struct => "struct",
            Self::Enum => "enum",
            Self::Trait => "trait",
            Self::Interface => "interface",
            Self::TypeAlias => "type_alias",
            Self::Constant => "constant",
            Self::Variable => "variable",
            Self::Module => "module",
            Self::Block => "block",
        };
        write!(f, "{name}")
    }
}

/// A search result returned by the retrieval pipeline ("context capsule").
///
/// Every field is always present in JSON serialization for schema consistency.
/// `symbol_name` and `symbol_type` serialize as `null` when not applicable
/// (e.g., for fallback/block chunks that don't correspond to a named symbol).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Repository-relative file path.
    pub file_path: String,
    /// 1-based start line.
    pub line_start: u32,
    /// 1-based end line (inclusive).
    pub line_end: u32,
    /// The code content of this result (complete symbol body, not truncated).
    pub content: String,
    /// Programming language.
    pub language: Language,
    /// Relevance score (higher is better).
    pub score: f64,
    /// Symbol name (`null` if the result doesn't correspond to a named symbol).
    pub symbol_name: Option<String>,
    /// Symbol type (`null` if the result doesn't correspond to a typed symbol).
    pub symbol_type: Option<SymbolType>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn language_from_extension_rust() {
        assert_eq!(Language::from_extension("rs"), Language::Rust);
    }

    #[test]
    fn language_from_extension_typescript() {
        assert_eq!(Language::from_extension("ts"), Language::TypeScript);
        assert_eq!(Language::from_extension("tsx"), Language::TypeScript);
    }

    #[test]
    fn language_from_extension_python() {
        assert_eq!(Language::from_extension("py"), Language::Python);
        assert_eq!(Language::from_extension("pyi"), Language::Python);
    }

    #[test]
    fn language_from_extension_unknown() {
        assert_eq!(Language::from_extension("xyz"), Language::Unknown);
    }

    #[test]
    fn language_from_extension_case_insensitive() {
        assert_eq!(Language::from_extension("RS"), Language::Rust);
        assert_eq!(Language::from_extension("Py"), Language::Python);
    }

    // ── Tier 1B extension mapping tests ─────────────────────────

    #[test]
    fn language_from_extension_html() {
        assert_eq!(Language::from_extension("html"), Language::Html);
        assert_eq!(Language::from_extension("htm"), Language::Html);
    }

    #[test]
    fn language_from_extension_css() {
        assert_eq!(Language::from_extension("css"), Language::Css);
    }

    #[test]
    fn language_from_extension_scss() {
        assert_eq!(Language::from_extension("scss"), Language::Scss);
    }

    #[test]
    fn language_from_extension_vue() {
        assert_eq!(Language::from_extension("vue"), Language::Vue);
    }

    #[test]
    fn language_from_extension_graphql() {
        assert_eq!(Language::from_extension("graphql"), Language::GraphQl);
        assert_eq!(Language::from_extension("gql"), Language::GraphQl);
    }

    #[test]
    fn language_from_extension_cmake() {
        assert_eq!(Language::from_extension("cmake"), Language::CMake);
    }

    #[test]
    fn language_from_extension_xml() {
        assert_eq!(Language::from_extension("xml"), Language::Xml);
        assert_eq!(Language::from_extension("xsl"), Language::Xml);
        assert_eq!(Language::from_extension("xsd"), Language::Xml);
        assert_eq!(Language::from_extension("svg"), Language::Xml);
    }

    #[test]
    fn language_from_filename_dockerfile() {
        assert_eq!(
            Language::from_filename("Dockerfile"),
            Some(Language::Dockerfile)
        );
        assert_eq!(
            Language::from_filename("dockerfile"),
            Some(Language::Dockerfile)
        );
    }

    #[test]
    fn language_from_filename_cmakelists() {
        assert_eq!(
            Language::from_filename("CMakeLists.txt"),
            Some(Language::CMake)
        );
    }

    #[test]
    fn language_from_filename_unknown() {
        assert_eq!(Language::from_filename("main.rs"), None);
        assert_eq!(Language::from_filename("README.md"), None);
    }

    #[test]
    fn language_display() {
        assert_eq!(Language::Rust.to_string(), "rust");
        assert_eq!(Language::TypeScript.to_string(), "typescript");
        assert_eq!(Language::Unknown.to_string(), "unknown");
    }

    #[test]
    fn language_display_tier1b() {
        assert_eq!(Language::Html.to_string(), "html");
        assert_eq!(Language::Css.to_string(), "css");
        assert_eq!(Language::Scss.to_string(), "scss");
        assert_eq!(Language::Vue.to_string(), "vue");
        assert_eq!(Language::GraphQl.to_string(), "graphql");
        assert_eq!(Language::CMake.to_string(), "cmake");
        assert_eq!(Language::Dockerfile.to_string(), "dockerfile");
        assert_eq!(Language::Xml.to_string(), "xml");
    }

    // ── Tier 2A extension mapping tests ─────────────────────────

    #[test]
    fn language_from_extension_objectivec() {
        assert_eq!(Language::from_extension("m"), Language::ObjectiveC);
        assert_eq!(Language::from_extension("mm"), Language::ObjectiveC);
    }

    #[test]
    fn language_from_extension_perl() {
        assert_eq!(Language::from_extension("pl"), Language::Perl);
        assert_eq!(Language::from_extension("pm"), Language::Perl);
    }

    #[test]
    fn language_from_extension_julia() {
        assert_eq!(Language::from_extension("jl"), Language::Julia);
    }

    #[test]
    fn language_from_extension_nix() {
        assert_eq!(Language::from_extension("nix"), Language::Nix);
    }

    #[test]
    fn language_from_extension_ocaml() {
        assert_eq!(Language::from_extension("ml"), Language::OCaml);
        assert_eq!(Language::from_extension("mli"), Language::OCaml);
    }

    #[test]
    fn language_from_extension_groovy() {
        assert_eq!(Language::from_extension("groovy"), Language::Groovy);
    }

    #[test]
    fn language_from_extension_clojure() {
        assert_eq!(Language::from_extension("clj"), Language::Clojure);
        assert_eq!(Language::from_extension("cljs"), Language::Clojure);
        assert_eq!(Language::from_extension("cljc"), Language::Clojure);
    }

    #[test]
    fn language_from_extension_commonlisp() {
        assert_eq!(Language::from_extension("lisp"), Language::CommonLisp);
        assert_eq!(Language::from_extension("cl"), Language::CommonLisp);
        assert_eq!(Language::from_extension("lsp"), Language::CommonLisp);
    }

    #[test]
    fn language_from_extension_erlang() {
        assert_eq!(Language::from_extension("erl"), Language::Erlang);
        assert_eq!(Language::from_extension("hrl"), Language::Erlang);
    }

    #[test]
    fn language_from_extension_fsharp() {
        assert_eq!(Language::from_extension("fs"), Language::FSharp);
        assert_eq!(Language::from_extension("fsi"), Language::FSharp);
        assert_eq!(Language::from_extension("fsx"), Language::FSharp);
    }

    #[test]
    fn language_from_extension_fortran() {
        assert_eq!(Language::from_extension("f"), Language::Fortran);
        assert_eq!(Language::from_extension("f90"), Language::Fortran);
        assert_eq!(Language::from_extension("f95"), Language::Fortran);
    }

    #[test]
    fn language_from_extension_powershell() {
        assert_eq!(Language::from_extension("ps1"), Language::PowerShell);
        assert_eq!(Language::from_extension("psm1"), Language::PowerShell);
    }

    #[test]
    fn language_from_extension_r() {
        assert_eq!(Language::from_extension("r"), Language::R);
        assert_eq!(Language::from_extension("R"), Language::R);
    }

    // ── Tier 2A batch 2 extension mapping tests ─────────────────

    #[test]
    fn language_from_extension_matlab() {
        assert_eq!(Language::from_extension("mlx"), Language::Matlab);
    }

    #[test]
    fn language_from_extension_dlang() {
        assert_eq!(Language::from_extension("d"), Language::DLang);
        assert_eq!(Language::from_extension("di"), Language::DLang);
    }

    #[test]
    fn language_from_extension_fish() {
        assert_eq!(Language::from_extension("fish"), Language::Fish);
    }

    #[test]
    fn language_from_extension_zsh() {
        assert_eq!(Language::from_extension("zsh"), Language::Zsh);
    }

    #[test]
    fn language_from_extension_luau() {
        assert_eq!(Language::from_extension("luau"), Language::Luau);
    }

    #[test]
    fn language_from_extension_scheme() {
        assert_eq!(Language::from_extension("scm"), Language::Scheme);
        assert_eq!(Language::from_extension("ss"), Language::Scheme);
    }

    #[test]
    fn language_from_extension_racket() {
        assert_eq!(Language::from_extension("rkt"), Language::Racket);
    }

    #[test]
    fn language_from_extension_elm() {
        assert_eq!(Language::from_extension("elm"), Language::Elm);
    }

    #[test]
    fn language_from_extension_glsl() {
        assert_eq!(Language::from_extension("glsl"), Language::Glsl);
        assert_eq!(Language::from_extension("vert"), Language::Glsl);
        assert_eq!(Language::from_extension("frag"), Language::Glsl);
    }

    #[test]
    fn language_from_extension_hlsl() {
        assert_eq!(Language::from_extension("hlsl"), Language::Hlsl);
        assert_eq!(Language::from_extension("hlsli"), Language::Hlsl);
        assert_eq!(Language::from_extension("fx"), Language::Hlsl);
    }

    #[test]
    fn language_display_tier2a_batch2() {
        assert_eq!(Language::Matlab.to_string(), "matlab");
        assert_eq!(Language::DLang.to_string(), "d");
        assert_eq!(Language::Fish.to_string(), "fish");
        assert_eq!(Language::Zsh.to_string(), "zsh");
        assert_eq!(Language::Luau.to_string(), "luau");
        assert_eq!(Language::Scheme.to_string(), "scheme");
        assert_eq!(Language::Racket.to_string(), "racket");
        assert_eq!(Language::Elm.to_string(), "elm");
        assert_eq!(Language::Glsl.to_string(), "glsl");
        assert_eq!(Language::Hlsl.to_string(), "hlsl");
    }

    // ── Tier 2B extension mapping tests ─────────────────

    #[test]
    fn language_from_extension_svelte() {
        assert_eq!(Language::from_extension("svelte"), Language::Svelte);
    }

    #[test]
    fn language_from_extension_astro() {
        assert_eq!(Language::from_extension("astro"), Language::Astro);
    }

    #[test]
    fn language_from_extension_ini() {
        assert_eq!(Language::from_extension("ini"), Language::Ini);
        assert_eq!(Language::from_extension("cfg"), Language::Ini);
        assert_eq!(Language::from_extension("conf"), Language::Ini);
    }

    #[test]
    fn language_from_extension_nginx() {
        assert_eq!(Language::from_extension("nginx"), Language::Nginx);
    }

    #[test]
    fn language_from_extension_prisma() {
        assert_eq!(Language::from_extension("prisma"), Language::Prisma);
    }

    #[test]
    fn language_from_filename_makefile() {
        assert_eq!(
            Language::from_filename("Makefile"),
            Some(Language::Makefile)
        );
        assert_eq!(
            Language::from_filename("makefile"),
            Some(Language::Makefile)
        );
        assert_eq!(
            Language::from_filename("GNUmakefile"),
            Some(Language::Makefile)
        );
    }

    #[test]
    fn language_from_filename_nginx_conf() {
        assert_eq!(Language::from_filename("nginx.conf"), Some(Language::Nginx));
    }

    #[test]
    fn language_display_tier2b() {
        assert_eq!(Language::Svelte.to_string(), "svelte");
        assert_eq!(Language::Astro.to_string(), "astro");
        assert_eq!(Language::Makefile.to_string(), "makefile");
        assert_eq!(Language::Ini.to_string(), "ini");
        assert_eq!(Language::Nginx.to_string(), "nginx");
        assert_eq!(Language::Prisma.to_string(), "prisma");
    }

    #[test]
    fn language_display_tier2a() {
        assert_eq!(Language::ObjectiveC.to_string(), "objectivec");
        assert_eq!(Language::Perl.to_string(), "perl");
        assert_eq!(Language::Julia.to_string(), "julia");
        assert_eq!(Language::Nix.to_string(), "nix");
        assert_eq!(Language::OCaml.to_string(), "ocaml");
        assert_eq!(Language::Groovy.to_string(), "groovy");
        assert_eq!(Language::Clojure.to_string(), "clojure");
        assert_eq!(Language::CommonLisp.to_string(), "commonlisp");
        assert_eq!(Language::Erlang.to_string(), "erlang");
        assert_eq!(Language::FSharp.to_string(), "fsharp");
        assert_eq!(Language::Fortran.to_string(), "fortran");
        assert_eq!(Language::PowerShell.to_string(), "powershell");
        assert_eq!(Language::R.to_string(), "r");
    }

    #[test]
    fn symbol_type_display() {
        assert_eq!(SymbolType::Function.to_string(), "function");
        assert_eq!(SymbolType::Class.to_string(), "class");
        assert_eq!(SymbolType::Block.to_string(), "block");
    }

    #[test]
    fn chunk_serialization_round_trip() {
        let chunk = Chunk {
            id: "test-1".to_string(),
            file_path: "src/main.rs".to_string(),
            line_start: 1,
            line_end: 10,
            content: "fn main() {}".to_string(),
            language: Language::Rust,
            symbol_type: Some(SymbolType::Function),
            symbol_name: Some("main".to_string()),
        };
        let json = serde_json::to_string(&chunk).unwrap();
        let deserialized: Chunk = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.id, "test-1");
        assert_eq!(deserialized.file_path, "src/main.rs");
        assert_eq!(deserialized.language, Language::Rust);
        assert_eq!(deserialized.symbol_name, Some("main".to_string()));
    }

    #[test]
    fn search_result_serialization_includes_null_fields() {
        let result = SearchResult {
            file_path: "lib.rs".to_string(),
            line_start: 5,
            line_end: 20,
            content: "pub fn example() {}".to_string(),
            language: Language::Rust,
            score: 0.95,
            symbol_name: None,
            symbol_type: None,
        };
        let json = serde_json::to_string(&result).unwrap();
        // Null fields must be present (not omitted) for schema consistency.
        assert!(json.contains("symbol_name"));
        assert!(json.contains("symbol_type"));
        // Parse and verify they are JSON null.
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(parsed["symbol_name"].is_null());
        assert!(parsed["symbol_type"].is_null());
    }

    #[test]
    fn search_result_serialization_includes_symbol_fields() {
        let result = SearchResult {
            file_path: "lib.rs".to_string(),
            line_start: 5,
            line_end: 20,
            content: "pub fn example() {}".to_string(),
            language: Language::Rust,
            score: 0.95,
            symbol_name: Some("example".to_string()),
            symbol_type: Some(SymbolType::Function),
        };
        let json = serde_json::to_string(&result).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["symbol_name"], "example");
        assert_eq!(parsed["symbol_type"], "function");
    }

    #[test]
    fn search_result_schema_consistent_with_and_without_symbols() {
        let with_symbols = SearchResult {
            file_path: "a.rs".to_string(),
            line_start: 1,
            line_end: 10,
            content: "fn foo() {}".to_string(),
            language: Language::Rust,
            score: 0.9,
            symbol_name: Some("foo".to_string()),
            symbol_type: Some(SymbolType::Function),
        };
        let without_symbols = SearchResult {
            file_path: "b.rs".to_string(),
            line_start: 1,
            line_end: 5,
            content: "// some code".to_string(),
            language: Language::Rust,
            score: 0.5,
            symbol_name: None,
            symbol_type: None,
        };

        let json_with: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&with_symbols).unwrap()).unwrap();
        let json_without: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&without_symbols).unwrap()).unwrap();

        // Both must have exactly the same set of keys.
        let keys_with: std::collections::BTreeSet<_> =
            json_with.as_object().unwrap().keys().collect();
        let keys_without: std::collections::BTreeSet<_> =
            json_without.as_object().unwrap().keys().collect();
        assert_eq!(
            keys_with, keys_without,
            "schema must be consistent: same keys regardless of symbol presence"
        );
    }

    // ── SearchFilters tests ─────────────────────────────────────

    fn make_test_result(
        file: &str,
        lang: Language,
        sym_name: Option<&str>,
        sym_type: Option<SymbolType>,
    ) -> SearchResult {
        SearchResult {
            file_path: file.to_string(),
            line_start: 1,
            line_end: 10,
            content: "test content".to_string(),
            language: lang,
            score: 1.0,
            symbol_name: sym_name.map(|s| s.to_string()),
            symbol_type: sym_type,
        }
    }

    #[test]
    fn filters_empty_matches_everything() {
        let filters = SearchFilters::default();
        assert!(filters.is_empty());
        let result = make_test_result("src/main.rs", Language::Rust, None, None);
        assert!(filters.matches(&result));
    }

    #[test]
    fn filter_by_language() {
        let filters = SearchFilters {
            language: Some("rust".to_string()),
            ..Default::default()
        };
        let rust_result = make_test_result("a.rs", Language::Rust, None, None);
        let py_result = make_test_result("a.py", Language::Python, None, None);
        assert!(filters.matches(&rust_result));
        assert!(!filters.matches(&py_result));
    }

    #[test]
    fn filter_by_language_case_insensitive() {
        let filters = SearchFilters {
            language: Some("Rust".to_string()),
            ..Default::default()
        };
        let result = make_test_result("a.rs", Language::Rust, None, None);
        assert!(filters.matches(&result));
    }

    #[test]
    fn filter_by_symbol_type() {
        let filters = SearchFilters {
            symbol_type: Some("function".to_string()),
            ..Default::default()
        };
        let func = make_test_result(
            "a.rs",
            Language::Rust,
            Some("foo"),
            Some(SymbolType::Function),
        );
        let cls = make_test_result(
            "a.py",
            Language::Python,
            Some("Bar"),
            Some(SymbolType::Class),
        );
        let none_sym = make_test_result("a.rs", Language::Rust, None, None);
        assert!(filters.matches(&func));
        assert!(!filters.matches(&cls));
        assert!(!filters.matches(&none_sym));
    }

    #[test]
    fn filter_by_symbol_type_case_insensitive() {
        let filters = SearchFilters {
            symbol_type: Some("Function".to_string()),
            ..Default::default()
        };
        let func = make_test_result(
            "a.rs",
            Language::Rust,
            Some("foo"),
            Some(SymbolType::Function),
        );
        assert!(filters.matches(&func));
    }

    #[test]
    fn filter_by_path_glob_extension() {
        let filters = SearchFilters {
            path_glob: Some("*.rs".to_string()),
            ..Default::default()
        };
        let rs = make_test_result("main.rs", Language::Rust, None, None);
        let py = make_test_result("main.py", Language::Python, None, None);
        assert!(filters.matches(&rs));
        assert!(!filters.matches(&py));
    }

    #[test]
    fn filter_by_path_glob_directory() {
        let filters = SearchFilters {
            path_glob: Some("src/**/*.rs".to_string()),
            ..Default::default()
        };
        let in_src = make_test_result("src/lib.rs", Language::Rust, None, None);
        let deep = make_test_result("src/a/b/c.rs", Language::Rust, None, None);
        let outside = make_test_result("tests/test.rs", Language::Rust, None, None);
        assert!(filters.matches(&in_src));
        assert!(filters.matches(&deep));
        assert!(!filters.matches(&outside));
    }

    #[test]
    fn filter_by_path_glob_doublestar_prefix() {
        let filters = SearchFilters {
            path_glob: Some("**/test_*.py".to_string()),
            ..Default::default()
        };
        let deep = make_test_result("tests/unit/test_auth.py", Language::Python, None, None);
        let top = make_test_result("test_main.py", Language::Python, None, None);
        let no_match = make_test_result("src/auth.py", Language::Python, None, None);
        assert!(filters.matches(&deep));
        assert!(filters.matches(&top));
        assert!(!filters.matches(&no_match));
    }

    #[test]
    fn filter_combined_lang_and_type() {
        let filters = SearchFilters {
            language: Some("rust".to_string()),
            symbol_type: Some("struct".to_string()),
            ..Default::default()
        };
        let rust_struct = make_test_result(
            "a.rs",
            Language::Rust,
            Some("Foo"),
            Some(SymbolType::Struct),
        );
        let rust_func = make_test_result(
            "b.rs",
            Language::Rust,
            Some("bar"),
            Some(SymbolType::Function),
        );
        let py_class = make_test_result(
            "c.py",
            Language::Python,
            Some("Baz"),
            Some(SymbolType::Class),
        );
        assert!(filters.matches(&rust_struct));
        assert!(!filters.matches(&rust_func));
        assert!(!filters.matches(&py_class));
    }

    // ── glob_matches tests ──────────────────────────────────────

    #[test]
    fn glob_star_matches_extension() {
        assert!(glob_matches("*.rs", "main.rs"));
        assert!(!glob_matches("*.rs", "main.py"));
    }

    #[test]
    fn glob_star_does_not_cross_slash() {
        assert!(!glob_matches("*.rs", "src/main.rs"));
    }

    #[test]
    fn glob_doublestar_matches_any_depth() {
        assert!(glob_matches("**/*.rs", "main.rs"));
        assert!(glob_matches("**/*.rs", "src/main.rs"));
        assert!(glob_matches("**/*.rs", "src/a/b/main.rs"));
    }

    #[test]
    fn glob_literal_prefix() {
        assert!(glob_matches("src/*.rs", "src/lib.rs"));
        assert!(!glob_matches("src/*.rs", "tests/lib.rs"));
    }

    #[test]
    fn glob_exact_match() {
        assert!(glob_matches("src/main.rs", "src/main.rs"));
        assert!(!glob_matches("src/main.rs", "src/lib.rs"));
    }

    #[test]
    fn glob_empty_pattern_matches_empty() {
        assert!(glob_matches("", ""));
        assert!(!glob_matches("", "something"));
    }

    #[test]
    fn glob_standalone_doublestar_matches_everything() {
        assert!(glob_matches("**", "main.rs"));
        assert!(glob_matches("**", "src/main.rs"));
        assert!(glob_matches("**", "src/a/b/c/main.rs"));
        assert!(glob_matches("**", ""));
    }

    #[test]
    fn glob_prefix_with_standalone_doublestar() {
        // Pattern like `src/**` should match any file under src/
        assert!(glob_matches("src/**", "src/main.rs"));
        assert!(glob_matches("src/**", "src/a/b/c.rs"));
        assert!(!glob_matches("src/**", "tests/main.rs"));
    }
}
