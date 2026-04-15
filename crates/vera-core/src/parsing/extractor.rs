//! AST symbol extraction rules per language.
//!
//! Defines which tree-sitter node types correspond to which [`SymbolType`]
//! for each supported language. Walks the AST to extract top-level symbols.

use crate::types::Language;
use crate::types::SymbolType;

/// A raw symbol extracted from the AST before chunking.
#[derive(Debug, Clone)]
pub struct RawSymbol {
    /// Name of the symbol (e.g., function name, class name).
    pub name: Option<String>,
    /// Type of symbol.
    pub symbol_type: SymbolType,
    /// 0-based byte offset of the symbol start in the source.
    pub start_byte: usize,
    /// 0-based byte offset of the symbol end in the source.
    pub end_byte: usize,
    /// 0-based start row in the source.
    pub start_row: usize,
    /// 0-based end row in the source.
    pub end_row: usize,
}

/// Maps a tree-sitter node kind to a [`SymbolType`] for the given language.
///
/// Returns `None` if the node kind is not a top-level symbol we extract.
pub fn classify_node(lang: Language, kind: &str) -> Option<SymbolType> {
    match lang {
        Language::Rust => classify_rust(kind),
        Language::TypeScript | Language::JavaScript => classify_typescript(kind),
        Language::Python => classify_python(kind),
        Language::Go => classify_go(kind),
        Language::Java => classify_java(kind),
        Language::C => classify_c(kind),
        Language::Cpp => classify_cpp(kind),
        Language::Ruby => classify_ruby(kind),
        Language::Bash => classify_bash(kind),
        Language::Kotlin => classify_kotlin(kind),
        Language::Swift => classify_swift(kind),
        Language::Zig => classify_zig(kind),
        Language::Lua => classify_lua(kind),
        Language::Scala => classify_scala(kind),
        Language::CSharp => classify_csharp(kind),
        Language::Php => classify_php(kind),
        Language::Haskell => classify_haskell(kind),
        Language::Elixir => classify_elixir(kind),
        Language::Dart => classify_dart(kind),
        Language::Sql => classify_sql(kind),
        Language::Hcl => classify_hcl(kind),
        Language::Protobuf => classify_protobuf(kind),
        Language::Html => classify_html(kind),
        Language::Css => classify_css(kind),
        Language::Scss => classify_scss(kind),
        Language::Vue => classify_vue(kind),
        Language::GraphQl => classify_graphql(kind),
        Language::CMake => classify_cmake(kind),
        Language::Dockerfile => classify_dockerfile(kind),
        Language::Xml => classify_xml(kind),
        Language::ObjectiveC => classify_objectivec(kind),
        Language::Perl => classify_perl(kind),
        Language::Julia => classify_julia(kind),
        Language::Nix => classify_nix(kind),
        Language::OCaml => classify_ocaml(kind),
        Language::Groovy => classify_groovy(kind),
        Language::Clojure => classify_clojure(kind),
        Language::CommonLisp => classify_commonlisp(kind),
        Language::Erlang => classify_erlang(kind),
        Language::FSharp => classify_fsharp(kind),
        Language::Fortran => classify_fortran(kind),
        Language::PowerShell => classify_powershell(kind),
        Language::R => classify_r(kind),
        Language::Matlab => classify_matlab(kind),
        Language::DLang => classify_dlang(kind),
        Language::Fish => classify_fish(kind),
        Language::Zsh => classify_zsh(kind),
        Language::Luau => classify_luau(kind),
        Language::Scheme => classify_scheme(kind),
        Language::Racket => classify_racket(kind),
        Language::Elm => classify_elm(kind),
        Language::Glsl => classify_glsl(kind),
        Language::Hlsl => classify_hlsl(kind),
        Language::Svelte => classify_svelte(kind),
        Language::Astro => classify_astro(kind),
        Language::Makefile => classify_makefile(kind),
        Language::Ini => classify_ini(kind),
        Language::Nginx => classify_nginx(kind),
        Language::Prisma => classify_prisma(kind),
        _ => None,
    }
}

fn classify_sql(kind: &str) -> Option<SymbolType> {
    match kind {
        "create_table" | "create_table_statement" | "table_definition" => Some(SymbolType::Struct),
        "create_function"
        | "create_function_statement"
        | "function_definition"
        | "create_procedure_statement"
        | "create_procedure"
        | "create_view"
        | "create_view_statement"
        | "view_definition" => Some(SymbolType::Function),
        _ => None,
    }
}

fn classify_hcl(kind: &str) -> Option<SymbolType> {
    match kind {
        "block" => Some(SymbolType::Struct),
        _ => None,
    }
}

fn classify_protobuf(kind: &str) -> Option<SymbolType> {
    match kind {
        "message" | "message_definition" => Some(SymbolType::Struct),
        "enum" | "enum_definition" => Some(SymbolType::Enum),
        "service" | "service_definition" => Some(SymbolType::Class),
        "rpc" | "rpc_definition" | "rpc_declaration" => Some(SymbolType::Method),
        _ => None,
    }
}

fn classify_html(kind: &str) -> Option<SymbolType> {
    match kind {
        "element" | "script_element" | "style_element" => Some(SymbolType::Block),
        _ => None,
    }
}

fn classify_css(kind: &str) -> Option<SymbolType> {
    match kind {
        "rule_set" => Some(SymbolType::Block),
        "media_statement" => Some(SymbolType::Block),
        "keyframes_statement" => Some(SymbolType::Block),
        "import_statement" => Some(SymbolType::Variable),
        _ => None,
    }
}

fn classify_scss(kind: &str) -> Option<SymbolType> {
    match kind {
        "rule_set" => Some(SymbolType::Block),
        "mixin_statement" => Some(SymbolType::Function),
        "function_statement" => Some(SymbolType::Function),
        "include_statement" => Some(SymbolType::Variable),
        "media_statement" => Some(SymbolType::Block),
        "keyframes_statement" => Some(SymbolType::Block),
        _ => None,
    }
}

fn classify_vue(kind: &str) -> Option<SymbolType> {
    match kind {
        "template_element" => Some(SymbolType::Block),
        "script_element" => Some(SymbolType::Block),
        "style_element" => Some(SymbolType::Block),
        _ => None,
    }
}

fn classify_graphql(kind: &str) -> Option<SymbolType> {
    match kind {
        "object_type_definition" | "input_object_type_definition" => Some(SymbolType::Struct),
        "interface_type_definition" => Some(SymbolType::Interface),
        "enum_type_definition" => Some(SymbolType::Enum),
        "union_type_definition" => Some(SymbolType::TypeAlias),
        "scalar_type_definition" => Some(SymbolType::TypeAlias),
        "schema_definition" => Some(SymbolType::Block),
        "operation_definition" => Some(SymbolType::Function),
        "fragment_definition" => Some(SymbolType::Function),
        "directive_definition" => Some(SymbolType::Function),
        _ => None,
    }
}

fn classify_cmake(kind: &str) -> Option<SymbolType> {
    match kind {
        "function_def" | "macro_def" => Some(SymbolType::Function),
        "if_condition" | "foreach_loop" | "while_loop" => Some(SymbolType::Block),
        "normal_command" => Some(SymbolType::Variable),
        _ => None,
    }
}

fn classify_dockerfile(kind: &str) -> Option<SymbolType> {
    match kind {
        "from_instruction" => Some(SymbolType::Block),
        "run_instruction" => Some(SymbolType::Block),
        "copy_instruction" | "add_instruction" => Some(SymbolType::Variable),
        "cmd_instruction" | "entrypoint_instruction" => Some(SymbolType::Function),
        "env_instruction" | "arg_instruction" | "label_instruction" => Some(SymbolType::Constant),
        "expose_instruction" => Some(SymbolType::Variable),
        "workdir_instruction" | "user_instruction" | "volume_instruction" => {
            Some(SymbolType::Variable)
        }
        _ => None,
    }
}

fn classify_xml(kind: &str) -> Option<SymbolType> {
    match kind {
        "element" => Some(SymbolType::Block),
        _ => None,
    }
}

fn classify_objectivec(kind: &str) -> Option<SymbolType> {
    match kind {
        "function_definition" => Some(SymbolType::Function),
        "method_declaration" | "method_definition" | "implementation_definition" => {
            Some(SymbolType::Method)
        }
        "class_interface" | "class_implementation" => Some(SymbolType::Class),
        "protocol_declaration" => Some(SymbolType::Interface),
        "category_interface" | "category_implementation" => Some(SymbolType::Class),
        _ => None,
    }
}

fn classify_perl(kind: &str) -> Option<SymbolType> {
    match kind {
        "function_definition" | "subroutine_declaration_statement" => Some(SymbolType::Function),
        "package_statement" => Some(SymbolType::Module),
        _ => None,
    }
}

fn classify_julia(kind: &str) -> Option<SymbolType> {
    match kind {
        "function_definition" | "short_function_definition" => Some(SymbolType::Function),
        "struct_definition" => Some(SymbolType::Struct),
        "module_definition" => Some(SymbolType::Module),
        "abstract_definition" => Some(SymbolType::TypeAlias),
        "macro_definition" => Some(SymbolType::Function),
        _ => None,
    }
}

fn classify_nix(kind: &str) -> Option<SymbolType> {
    match kind {
        "function_expression" | "function" => Some(SymbolType::Function),
        "binding" | "attrset_expression" => Some(SymbolType::Variable),
        "let_expression" => Some(SymbolType::Block),
        _ => None,
    }
}

fn classify_ocaml(kind: &str) -> Option<SymbolType> {
    match kind {
        "value_definition" | "let_binding" => Some(SymbolType::Function),
        "type_definition" => Some(SymbolType::TypeAlias),
        "module_definition" => Some(SymbolType::Module),
        "module_type_definition" => Some(SymbolType::Interface),
        "class_definition" => Some(SymbolType::Class),
        "external" => Some(SymbolType::Function),
        _ => None,
    }
}

fn classify_groovy(kind: &str) -> Option<SymbolType> {
    match kind {
        "function_definition" | "method_declaration" => Some(SymbolType::Function),
        "class_definition" | "class_declaration" => Some(SymbolType::Class),
        "interface_definition" | "interface_declaration" => Some(SymbolType::Interface),
        _ => None,
    }
}

fn classify_clojure(kind: &str) -> Option<SymbolType> {
    match kind {
        "list_lit" => None, // handled specially — (defn ...) etc
        "defn" => Some(SymbolType::Function),
        "ns" => Some(SymbolType::Module),
        _ => None,
    }
}

fn classify_commonlisp(kind: &str) -> Option<SymbolType> {
    match kind {
        "defun" | "defmacro" | "defgeneric" | "defmethod" => Some(SymbolType::Function),
        "defclass" => Some(SymbolType::Class),
        "defvar" | "defparameter" | "defconstant" => Some(SymbolType::Variable),
        "defpackage" => Some(SymbolType::Module),
        "list_lit" => None, // handled via recursion
        _ => None,
    }
}

fn classify_erlang(kind: &str) -> Option<SymbolType> {
    match kind {
        "function_clause" | "fun_expr" => Some(SymbolType::Function),
        "type_declaration" | "record_declaration" => Some(SymbolType::TypeAlias),
        "module_attribute" => Some(SymbolType::Module),
        _ => None,
    }
}

fn classify_fsharp(kind: &str) -> Option<SymbolType> {
    match kind {
        "function_or_value_defn" | "value_declaration" => Some(SymbolType::Function),
        "type_definition" | "type_abbrev_defn" => Some(SymbolType::TypeAlias),
        "module_defn" => Some(SymbolType::Module),
        "class_defn" => Some(SymbolType::Class),
        _ => None,
    }
}

fn classify_fortran(kind: &str) -> Option<SymbolType> {
    match kind {
        "function" | "function_statement" | "function_subprogram" => Some(SymbolType::Function),
        "subroutine" | "subroutine_statement" | "subroutine_subprogram" => {
            Some(SymbolType::Function)
        }
        "module" | "module_statement" => Some(SymbolType::Module),
        "derived_type_definition" | "type_statement" => Some(SymbolType::Struct),
        "program" | "program_statement" => Some(SymbolType::Block),
        _ => None,
    }
}

fn classify_powershell(kind: &str) -> Option<SymbolType> {
    match kind {
        "function_statement" => Some(SymbolType::Function),
        "class_statement" => Some(SymbolType::Class),
        "enum_statement" => Some(SymbolType::Enum),
        _ => None,
    }
}

fn classify_r(kind: &str) -> Option<SymbolType> {
    match kind {
        "function_definition" => Some(SymbolType::Function),
        _ => None,
    }
}

fn classify_matlab(kind: &str) -> Option<SymbolType> {
    match kind {
        "function_definition" => Some(SymbolType::Function),
        "class_definition" => Some(SymbolType::Class),
        _ => None,
    }
}

fn classify_dlang(kind: &str) -> Option<SymbolType> {
    match kind {
        "function_declaration" | "auto_declaration" => Some(SymbolType::Function),
        "class_declaration" => Some(SymbolType::Class),
        "struct_declaration" => Some(SymbolType::Struct),
        "enum_declaration" => Some(SymbolType::Enum),
        "interface_declaration" => Some(SymbolType::Interface),
        "module_declaration" => Some(SymbolType::Module),
        "template_declaration" => Some(SymbolType::TypeAlias),
        _ => None,
    }
}

fn classify_fish(kind: &str) -> Option<SymbolType> {
    match kind {
        "function_definition" => Some(SymbolType::Function),
        _ => None,
    }
}

fn classify_zsh(kind: &str) -> Option<SymbolType> {
    match kind {
        "function_definition" => Some(SymbolType::Function),
        _ => None,
    }
}

fn classify_luau(kind: &str) -> Option<SymbolType> {
    match kind {
        "function_declaration" | "local_function" => Some(SymbolType::Function),
        "type_definition" => Some(SymbolType::TypeAlias),
        _ => None,
    }
}

fn classify_scheme(kind: &str) -> Option<SymbolType> {
    match kind {
        "define" => Some(SymbolType::Function),
        "lambda" => Some(SymbolType::Function),
        _ => None,
    }
}

fn classify_racket(kind: &str) -> Option<SymbolType> {
    match kind {
        "define" => Some(SymbolType::Function),
        "lambda" => Some(SymbolType::Function),
        "module" => Some(SymbolType::Module),
        "struct" => Some(SymbolType::Struct),
        _ => None,
    }
}

fn classify_elm(kind: &str) -> Option<SymbolType> {
    match kind {
        "value_declaration" => Some(SymbolType::Function),
        "type_alias_declaration" => Some(SymbolType::TypeAlias),
        "type_declaration" => Some(SymbolType::Enum),
        "module_declaration" => Some(SymbolType::Module),
        _ => None,
    }
}

fn classify_glsl(kind: &str) -> Option<SymbolType> {
    match kind {
        "function_definition" => Some(SymbolType::Function),
        "struct_specifier" => Some(SymbolType::Struct),
        "declaration" => Some(SymbolType::Variable),
        _ => None,
    }
}

fn classify_hlsl(kind: &str) -> Option<SymbolType> {
    match kind {
        "function_definition" => Some(SymbolType::Function),
        "class_specifier" => Some(SymbolType::Class),
        "struct_specifier" => Some(SymbolType::Struct),
        "declaration" => Some(SymbolType::Variable),
        _ => None,
    }
}

fn classify_svelte(kind: &str) -> Option<SymbolType> {
    match kind {
        "script_element" => Some(SymbolType::Block),
        "style_element" => Some(SymbolType::Block),
        "element" => Some(SymbolType::Block),
        "if_statement" | "each_statement" | "await_statement" => Some(SymbolType::Block),
        _ => None,
    }
}

fn classify_astro(kind: &str) -> Option<SymbolType> {
    match kind {
        "frontmatter" => Some(SymbolType::Block),
        "element" | "script_element" | "style_element" | "component" => Some(SymbolType::Block),
        _ => None,
    }
}

fn classify_makefile(kind: &str) -> Option<SymbolType> {
    match kind {
        "rule" => Some(SymbolType::Function),
        "variable_assignment" => Some(SymbolType::Variable),
        "define_directive" => Some(SymbolType::Function),
        "include_directive" => Some(SymbolType::Variable),
        _ => None,
    }
}

fn classify_ini(kind: &str) -> Option<SymbolType> {
    match kind {
        "section" => Some(SymbolType::Block),
        "setting" => Some(SymbolType::Variable),
        _ => None,
    }
}

fn classify_nginx(kind: &str) -> Option<SymbolType> {
    match kind {
        "block" => Some(SymbolType::Block),
        "directive" => Some(SymbolType::Variable),
        _ => None,
    }
}

fn classify_prisma(kind: &str) -> Option<SymbolType> {
    match kind {
        "model_declaration" => Some(SymbolType::Struct),
        "enum_declaration" => Some(SymbolType::Enum),
        "generator_declaration" | "datasource_declaration" => Some(SymbolType::Block),
        "type_declaration" => Some(SymbolType::TypeAlias),
        _ => None,
    }
}

fn classify_rust(kind: &str) -> Option<SymbolType> {
    match kind {
        "function_item" => Some(SymbolType::Function),
        "impl_item" => Some(SymbolType::Block),
        "struct_item" => Some(SymbolType::Struct),
        "enum_item" => Some(SymbolType::Enum),
        "trait_item" => Some(SymbolType::Trait),
        "type_item" => Some(SymbolType::TypeAlias),
        "const_item" => Some(SymbolType::Constant),
        "static_item" => Some(SymbolType::Constant),
        "mod_item" => Some(SymbolType::Module),
        _ => None,
    }
}

fn classify_typescript(kind: &str) -> Option<SymbolType> {
    match kind {
        "function_declaration" => Some(SymbolType::Function),
        "class_declaration" => Some(SymbolType::Class),
        "interface_declaration" => Some(SymbolType::Interface),
        "type_alias_declaration" => Some(SymbolType::TypeAlias),
        "enum_declaration" => Some(SymbolType::Enum),
        "method_definition" => Some(SymbolType::Method),
        "lexical_declaration" | "variable_declaration" => Some(SymbolType::Variable),
        "export_statement" => None, // recurse into children
        _ => None,
    }
}

fn classify_python(kind: &str) -> Option<SymbolType> {
    match kind {
        "function_definition" => Some(SymbolType::Function),
        "class_definition" => Some(SymbolType::Class),
        "decorated_definition" => None, // recurse into children
        _ => None,
    }
}

fn classify_go(kind: &str) -> Option<SymbolType> {
    match kind {
        "function_declaration" => Some(SymbolType::Function),
        "method_declaration" => Some(SymbolType::Method),
        "type_declaration" => None, // contains type_spec children
        "type_spec" => Some(SymbolType::TypeAlias), // refined by child kind
        _ => None,
    }
}

fn classify_java(kind: &str) -> Option<SymbolType> {
    match kind {
        "method_declaration" => Some(SymbolType::Method),
        "class_declaration" => Some(SymbolType::Class),
        "interface_declaration" => Some(SymbolType::Interface),
        "enum_declaration" => Some(SymbolType::Enum),
        "constructor_declaration" => Some(SymbolType::Method),
        _ => None,
    }
}

fn classify_c(kind: &str) -> Option<SymbolType> {
    match kind {
        "function_definition" => Some(SymbolType::Function),
        "struct_specifier" => Some(SymbolType::Struct),
        "enum_specifier" => Some(SymbolType::Enum),
        "type_definition" => Some(SymbolType::TypeAlias),
        "declaration" => Some(SymbolType::Variable),
        _ => None,
    }
}

fn classify_cpp(kind: &str) -> Option<SymbolType> {
    match kind {
        "function_definition" => Some(SymbolType::Function),
        "class_specifier" => Some(SymbolType::Class),
        "struct_specifier" => Some(SymbolType::Struct),
        "enum_specifier" => Some(SymbolType::Enum),
        "type_definition" => Some(SymbolType::TypeAlias),
        "namespace_definition" => Some(SymbolType::Module),
        "template_declaration" => None, // recurse into children
        "declaration" => Some(SymbolType::Variable),
        _ => None,
    }
}

fn classify_ruby(kind: &str) -> Option<SymbolType> {
    match kind {
        "method" | "singleton_method" => Some(SymbolType::Function),
        "class" => Some(SymbolType::Class),
        "module" => Some(SymbolType::Module),
        _ => None,
    }
}

fn classify_bash(kind: &str) -> Option<SymbolType> {
    match kind {
        "function_definition" => Some(SymbolType::Function),
        _ => None,
    }
}

fn classify_kotlin(kind: &str) -> Option<SymbolType> {
    match kind {
        "function_declaration" => Some(SymbolType::Function),
        "class_declaration" => Some(SymbolType::Class), // refined later
        "object_declaration" => Some(SymbolType::Class), // objects treated as classes/singletons
        _ => None,
    }
}

fn classify_swift(kind: &str) -> Option<SymbolType> {
    match kind {
        "function_declaration" => Some(SymbolType::Function),
        "class_declaration" => Some(SymbolType::Class), // refined later
        "protocol_declaration" => Some(SymbolType::Interface),
        _ => None,
    }
}

fn classify_zig(kind: &str) -> Option<SymbolType> {
    match kind {
        "function_declaration" => Some(SymbolType::Function),
        // variable_declaration handled in collect_symbols to extract structs
        _ => None,
    }
}

fn classify_lua(kind: &str) -> Option<SymbolType> {
    match kind {
        "function_declaration" => Some(SymbolType::Function),
        _ => None,
    }
}

fn classify_scala(kind: &str) -> Option<SymbolType> {
    match kind {
        "function_definition" => Some(SymbolType::Function),
        "class_definition" => Some(SymbolType::Class),
        "trait_definition" => Some(SymbolType::Trait),
        "object_definition" => Some(SymbolType::Module), // objects map well to modules in scala
        _ => None,
    }
}

fn classify_csharp(kind: &str) -> Option<SymbolType> {
    match kind {
        "class_declaration" => Some(SymbolType::Class),
        "interface_declaration" => Some(SymbolType::Interface),
        "struct_declaration" => Some(SymbolType::Struct),
        "enum_declaration" => Some(SymbolType::Enum),
        "method_declaration" | "local_function_statement" => Some(SymbolType::Method),
        "namespace_declaration" | "file_scoped_namespace_declaration" => Some(SymbolType::Module),
        _ => None,
    }
}

fn classify_php(kind: &str) -> Option<SymbolType> {
    match kind {
        "function_definition" => Some(SymbolType::Function),
        "class_declaration" => Some(SymbolType::Class),
        "interface_declaration" => Some(SymbolType::Interface),
        "method_declaration" => Some(SymbolType::Method),
        _ => None,
    }
}

fn classify_haskell(kind: &str) -> Option<SymbolType> {
    match kind {
        "function" | "signature" => Some(SymbolType::Function),
        "data_type" => Some(SymbolType::Struct),
        "type_alias" | "type_synomym" => Some(SymbolType::TypeAlias),
        "newtype" => Some(SymbolType::TypeAlias),
        _ => None,
    }
}

fn classify_elixir(_kind: &str) -> Option<SymbolType> {
    None
}

fn classify_dart(kind: &str) -> Option<SymbolType> {
    match kind {
        "class_declaration" | "class_definition" => Some(SymbolType::Class),
        "enum_declaration" => Some(SymbolType::Enum),
        "function_signature" | "function_definition" => Some(SymbolType::Function),
        "method_signature" | "method_definition" => Some(SymbolType::Method),
        _ => None,
    }
}

/// Extract the name of a symbol from a tree-sitter node.
///
/// Looks for the first `name` or `identifier`-type child node.
pub fn extract_name(node: &tree_sitter::Node<'_>, source: &[u8]) -> Option<String> {
    if node.kind() == "impl_item" {
        return extract_impl_name(node, source);
    }

    // HCL block name (second child that is an identifier or string_lit)
    if node.kind() == "block" {
        let mut cursor = node.walk();
        let mut found_type = false;
        for child in node.children(&mut cursor) {
            if !found_type && child.kind() == "identifier" {
                found_type = true;
                continue;
            }
            if found_type && (child.kind() == "string_lit" || child.kind() == "identifier") {
                return child
                    .utf8_text(source)
                    .ok()
                    .map(|s| s.trim_matches('"').to_string());
            }
        }
    }

    // Try common name field patterns
    for field in &["name", "declarator"] {
        if let Some(child) = node.child_by_field_name(field) {
            return name_from_node(&child, source);
        }
    }

    // Protobuf names
    if node.kind() == "message"
        || node.kind() == "enum"
        || node.kind() == "service"
        || node.kind() == "rpc"
        || node.kind().ends_with("_definition")
    {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind().ends_with("_name") {
                let mut inner = child.walk();
                for c in child.children(&mut inner) {
                    if c.kind() == "identifier" {
                        return c.utf8_text(source).ok().map(|s| s.to_string());
                    }
                }
                return child.utf8_text(source).ok().map(|s| s.to_string());
            }
        }
    }
    // Dart: method_signature -> function_signature -> name
    if node.kind() == "method_signature" {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "function_signature" {
                if let Some(name_child) = child.child_by_field_name("name") {
                    return name_from_node(&name_child, source);
                }
            }
        }
    }
    // Fallback: look for first identifier child
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        let kind = child.kind();
        if kind == "identifier"
            || kind == "type_identifier"
            || kind == "property_identifier"
            || kind == "simple_identifier"
            || kind == "word"
            || kind == "constant"
        {
            return Some(child.utf8_text(source).ok()?.to_string());
        }
    }
    None
}

fn extract_impl_name(node: &tree_sitter::Node<'_>, source: &[u8]) -> Option<String> {
    let text = node.utf8_text(source).ok()?;
    let header = text
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or_default();
    if !header.starts_with("impl") {
        return None;
    }

    let mut header = header
        .trim_end_matches('{')
        .trim_end_matches("where")
        .trim();
    if let Some((prefix, _)) = header.split_once(" where ") {
        header = prefix.trim();
    }

    let cleaned = header.split_whitespace().collect::<Vec<_>>().join(" ");
    (!cleaned.is_empty()).then_some(cleaned)
}

/// Extract a name string from a node, handling nested patterns.
fn name_from_node(node: &tree_sitter::Node<'_>, source: &[u8]) -> Option<String> {
    let kind = node.kind();
    // Direct identifier nodes
    if kind == "identifier"
        || kind == "type_identifier"
        || kind == "property_identifier"
        || kind == "field_identifier"
        || kind == "simple_identifier"
        || kind == "word"
        || kind == "constant"
        || kind == "name"
        || kind == "variable"
    {
        return Some(node.utf8_text(source).ok()?.to_string());
    }
    // Pointer declarators, reference declarators, etc. (C/C++)
    if kind.contains("declarator") {
        if let Some(inner) = node.child_by_field_name("declarator") {
            return name_from_node(&inner, source);
        }
        // Or a direct identifier child
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "identifier" || child.kind() == "field_identifier" {
                return Some(child.utf8_text(source).ok()?.to_string());
            }
        }
    }
    None
}

/// Extract top-level symbols from a parsed tree.
///
/// Walks the AST, identifying nodes that match the language's extraction rules.
/// Returns symbols sorted by their position in the source.
pub fn extract_symbols(tree: &tree_sitter::Tree, source: &[u8], lang: Language) -> Vec<RawSymbol> {
    let mut symbols = Vec::new();
    let mut cursor = tree.root_node().walk();
    collect_symbols_cursor(&mut cursor, source, lang, &mut symbols, 0);
    symbols.sort_by_key(|s| s.start_byte);
    symbols
}

/// Extract reStructuredText heading titles from the AST.
///
/// Returns a sorted list of `(start_row, title_text)` pairs. Rows are 0-based.
pub fn extract_rst_section_titles(tree: &tree_sitter::Tree, source: &[u8]) -> Vec<(u32, String)> {
    fn walk(node: tree_sitter::Node<'_>, source: &[u8], out: &mut Vec<(u32, String)>) {
        if node.kind() == "title" {
            if let Ok(text) = node.utf8_text(source) {
                let title = text.split_whitespace().collect::<Vec<_>>().join(" ");
                if !title.is_empty() {
                    out.push((node.start_position().row as u32, title));
                }
            }
        }

        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            walk(child, source, out);
        }
    }

    let mut titles = Vec::new();
    walk(tree.root_node(), source, &mut titles);
    titles.sort_by_key(|(row, _)| *row);
    titles.dedup_by(|a, b| a.0 == b.0);
    titles
}

/// Iterates siblings using a single `TreeCursor`, avoiding per-level cursor
/// allocation. Delegates to [`collect_symbols`] for per-node logic.
fn collect_symbols_cursor(
    cursor: &mut tree_sitter::TreeCursor,
    source: &[u8],
    lang: Language,
    symbols: &mut Vec<RawSymbol>,
    depth: usize,
) {
    if depth > 6 {
        return;
    }
    if !cursor.goto_first_child() {
        return;
    }
    loop {
        collect_symbols(cursor.node(), source, lang, symbols, depth);
        if !cursor.goto_next_sibling() {
            break;
        }
    }
    cursor.goto_parent();
}

/// Recursively collect symbols from AST nodes.
///
/// `depth` limits how deep we recurse to avoid extracting deeply nested items
/// as top-level symbols. We go up to depth 6 to handle patterns like:
/// - export_statement > function_declaration (TS/JS)
/// - decorated_definition > function_definition (Python)
/// - impl_item > function_item (Rust methods)
/// - source_file > document > definition > type_system_definition >
///   type_definition > object_type_definition (GraphQL)
fn collect_symbols(
    node: tree_sitter::Node<'_>,
    source: &[u8],
    lang: Language,
    symbols: &mut Vec<RawSymbol>,
    depth: usize,
) {
    if depth > 6 {
        return;
    }

    let kind = node.kind();

    // Handle Go type_declaration → recurse into type_spec children
    if lang == Language::Go && kind == "type_declaration" {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "type_spec" {
                let sym_type = refine_go_type_spec(&child, source);
                let name = extract_name(&child, source);
                symbols.push(RawSymbol {
                    name,
                    symbol_type: sym_type,
                    start_byte: child.start_byte(),
                    end_byte: child.end_byte(),
                    start_row: child.start_position().row,
                    end_row: child.end_position().row,
                });
            }
        }
        return;
    }

    // Handle R binary_operator (x <- function() { ... }) as named function
    if lang == Language::R && kind == "binary_operator" {
        if let Some(rhs) = node.child_by_field_name("rhs") {
            if rhs.kind() == "function_definition" {
                let name = extract_name(&node, source);
                symbols.push(RawSymbol {
                    name,
                    symbol_type: SymbolType::Function,
                    start_byte: node.start_byte(),
                    end_byte: node.end_byte(),
                    start_row: node.start_position().row,
                    end_row: node.end_position().row,
                });
                return;
            }
        }
    }

    // Handle Zig variable_declaration -> struct_declaration
    if lang == Language::Zig && kind == "variable_declaration" {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "struct_declaration" {
                let name = extract_name(&node, source);
                symbols.push(RawSymbol {
                    name,
                    symbol_type: SymbolType::Struct,
                    start_byte: node.start_byte(),
                    end_byte: node.end_byte(),
                    start_row: node.start_position().row,
                    end_row: node.end_position().row,
                });
                return;
            }
        }
    }

    // Handle Scheme S-expressions: (define (name ...) ...)
    if lang == Language::Scheme && kind == "list" {
        if let Some(sym) = extract_scheme_define(&node, source) {
            symbols.push(sym);
            return;
        }
    }

    // Handle Racket S-expressions: (define (name ...) ...), (module ...), (struct ...)
    if lang == Language::Racket && kind == "list" {
        if let Some(sym) = extract_racket_define(&node, source) {
            symbols.push(sym);
            return;
        }
    }

    // Handle Clojure S-expressions: (defn name ...), (ns name), (defmacro name ...), (def name ...)
    if lang == Language::Clojure && kind == "list_lit" {
        if let Some(sym) = extract_clojure_define(&node, source) {
            symbols.push(sym);
            return;
        }
    }

    // Handle Common Lisp: defun has defun_header with name; defclass via list_lit
    if lang == Language::CommonLisp && kind == "defun" {
        if let Some(sym) = extract_commonlisp_defun(&node, source) {
            symbols.push(sym);
            return;
        }
    }
    if lang == Language::CommonLisp && kind == "list_lit" {
        if let Some(sym) = extract_commonlisp_list(&node, source) {
            symbols.push(sym);
            return;
        }
    }

    // Handle Elixir calls
    if lang == Language::Elixir && kind == "call" {
        if let Some(target) = node.child_by_field_name("target") {
            if let Ok(text) = target.utf8_text(source) {
                let sym_type = match text {
                    "defmodule" => Some(SymbolType::Module),
                    "def" | "defp" | "defmacro" => Some(SymbolType::Function),
                    _ => None,
                };
                if let Some(st) = sym_type {
                    let name = extract_elixir_name(&node, source);
                    symbols.push(RawSymbol {
                        name,
                        symbol_type: st,
                        start_byte: node.start_byte(),
                        end_byte: node.end_byte(),
                        start_row: node.start_position().row,
                        end_row: node.end_position().row,
                    });
                    if text == "defmodule" {
                        if let Some(do_block) = get_elixir_do_block(&node) {
                            let mut cursor = do_block.walk();
                            collect_symbols_cursor(&mut cursor, source, lang, symbols, depth + 1);
                        }
                    }
                    return;
                }
            }
        }
    }

    if let Some(mut sym_type) = classify_node(lang, kind) {
        // Refine HCL blocks
        if lang == Language::Hcl && kind == "block" {
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                if child.kind() == "identifier" {
                    if let Ok(text) = child.utf8_text(source) {
                        sym_type = match text {
                            "resource" | "data" => SymbolType::Struct,
                            "variable" | "output" => SymbolType::TypeAlias,
                            "module" => SymbolType::Module,
                            _ => SymbolType::Struct,
                        };
                    }
                    break;
                }
            }
        }

        // Refine Kotlin and Swift class_declaration
        if (lang == Language::Kotlin || lang == Language::Swift) && kind == "class_declaration" {
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                let ckind = child.kind();
                if ckind == "enum" || ckind == "enum_class_body" {
                    sym_type = SymbolType::Enum;
                } else if ckind == "interface" {
                    sym_type = SymbolType::Interface;
                } else if ckind == "struct" {
                    sym_type = SymbolType::Struct;
                }
            }
        }

        // For Rust impl blocks, extract methods inside but also keep the whole block
        if lang == Language::Rust && kind == "impl_item" {
            extract_impl_methods(node, source, lang, symbols);
            return;
        }

        // For Python classes, extract methods as separate chunks instead of
        // keeping the entire class body as a single chunk.
        if lang == Language::Python && kind == "class_definition" {
            extract_python_class_methods(node, source, symbols);
            return;
        }

        // For Objective-C classes, extract methods as separate chunks
        if lang == Language::ObjectiveC
            && (kind == "class_interface"
                || kind == "class_implementation"
                || kind == "category_interface"
                || kind == "category_implementation")
        {
            let name = extract_name(&node, source);
            symbols.push(RawSymbol {
                name,
                symbol_type: sym_type,
                start_byte: node.start_byte(),
                end_byte: node.end_byte(),
                start_row: node.start_position().row,
                end_row: node.end_position().row,
            });
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                if let Some(child_sym) = classify_node(lang, child.kind()) {
                    let child_name = extract_name(&child, source);
                    symbols.push(RawSymbol {
                        name: child_name,
                        symbol_type: child_sym,
                        start_byte: child.start_byte(),
                        end_byte: child.end_byte(),
                        start_row: child.start_position().row,
                        end_row: child.end_position().row,
                    });
                }
                // Also check inside implementation_definition
                if child.kind() == "implementation_definition" {
                    let mut inner = child.walk();
                    for inner_child in child.children(&mut inner) {
                        if let Some(inner_sym) = classify_node(lang, inner_child.kind()) {
                            let inner_name = extract_name(&inner_child, source);
                            symbols.push(RawSymbol {
                                name: inner_name,
                                symbol_type: inner_sym,
                                start_byte: inner_child.start_byte(),
                                end_byte: inner_child.end_byte(),
                                start_row: inner_child.start_position().row,
                                end_row: inner_child.end_position().row,
                            });
                        }
                    }
                }
            }
            return;
        }

        // For C# namespace, we want to recurse inside.
        if lang == Language::CSharp
            && (kind == "namespace_declaration" || kind == "file_scoped_namespace_declaration")
        {
            let name = extract_name(&node, source);
            symbols.push(RawSymbol {
                name,
                symbol_type: sym_type,
                start_byte: node.start_byte(),
                end_byte: node.end_byte(),
                start_row: node.start_position().row,
                end_row: node.end_position().row,
            });
            let mut cursor = node.walk();
            collect_symbols_cursor(&mut cursor, source, lang, symbols, depth + 1);
            return;
        }

        // For C#, PHP, Dart classes, extract methods:
        if (lang == Language::CSharp || lang == Language::Php || lang == Language::Dart)
            && (kind == "class_declaration"
                || kind == "class_definition"
                || kind == "interface_declaration")
        {
            extract_general_class_methods(node, source, lang, symbols, sym_type);
            return;
        }

        // For Protobuf services, extract rpc methods:
        if lang == Language::Protobuf && (kind == "service" || kind == "service_definition") {
            let name = extract_name(&node, source);
            symbols.push(RawSymbol {
                name,
                symbol_type: sym_type,
                start_byte: node.start_byte(),
                end_byte: node.end_byte(),
                start_row: node.start_position().row,
                end_row: node.end_position().row,
            });

            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                // Sometime tree-sitter has a body node containing rpcs
                if child.kind() == "service_body" || child.kind() == "block" {
                    let mut inner_cursor = child.walk();
                    for inner_child in child.children(&mut inner_cursor) {
                        if inner_child.kind() == "rpc"
                            || inner_child.kind() == "rpc_definition"
                            || inner_child.kind() == "rpc_declaration"
                        {
                            if let Some(rpc_type) = classify_node(lang, inner_child.kind()) {
                                let end_byte = inner_child.end_byte();
                                let end_row = inner_child.end_position().row;
                                let rpc_name = extract_name(&inner_child, source);
                                symbols.push(RawSymbol {
                                    name: rpc_name,
                                    symbol_type: rpc_type,
                                    start_byte: inner_child.start_byte(),
                                    end_byte,
                                    start_row: inner_child.start_position().row,
                                    end_row,
                                });
                            }
                        }
                    }
                } else if child.kind() == "rpc"
                    || child.kind() == "rpc_definition"
                    || child.kind() == "rpc_declaration"
                {
                    if let Some(rpc_type) = classify_node(lang, child.kind()) {
                        let end_byte = child.end_byte();
                        let end_row = child.end_position().row;
                        let rpc_name = extract_name(&child, source);
                        symbols.push(RawSymbol {
                            name: rpc_name,
                            symbol_type: rpc_type,
                            start_byte: child.start_byte(),
                            end_byte,
                            start_row: child.start_position().row,
                            end_row,
                        });
                    }
                }
            }
            return;
        }

        let mut end_byte = node.end_byte();
        let mut end_row = node.end_position().row;

        if lang == Language::Dart && (kind == "function_signature" || kind == "method_signature") {
            if let Some(next_sibling) = node.next_sibling() {
                if next_sibling.kind() == "function_body" {
                    end_byte = next_sibling.end_byte();
                    end_row = next_sibling.end_position().row;
                }
            }
        }

        let name = extract_name(&node, source);
        symbols.push(RawSymbol {
            name,
            symbol_type: sym_type,
            start_byte: node.start_byte(),
            end_byte,
            start_row: node.start_position().row,
            end_row,
        });
        return;
    }

    // Recurse into children for wrapper nodes
    let mut cursor = node.walk();
    collect_symbols_cursor(&mut cursor, source, lang, symbols, depth + 1);
}

/// Extract a symbol from a Scheme `list` node if it starts with `define` or `define-syntax`.
///
/// Scheme AST: `list` → first child `symbol` "define" → second child is either
/// a `list` (procedure: name is its first `symbol`) or a `symbol` (variable definition).
fn extract_scheme_define(node: &tree_sitter::Node<'_>, source: &[u8]) -> Option<RawSymbol> {
    let mut cursor = node.walk();
    let children: Vec<_> = node.children(&mut cursor).collect();

    // Need at least: ( symbol <something> )
    // Find the first `symbol` child (skip parentheses)
    let first_sym = children.iter().find(|c| c.kind() == "symbol")?;
    let keyword = first_sym.utf8_text(source).ok()?;

    let sym_type = match keyword {
        "define" | "define-syntax" => SymbolType::Function,
        _ => return None,
    };

    // The name is in the next meaningful child after the keyword
    let after_keyword: Vec<_> = children
        .iter()
        .skip_while(|c| c.start_byte() <= first_sym.start_byte())
        .filter(|c| c.kind() != "(" && c.kind() != ")")
        .collect();

    let name = if let Some(next) = after_keyword.first() {
        if next.kind() == "list" {
            // (define (hello ...) ...) → name is first symbol in the inner list
            let mut inner_cursor = next.walk();
            next.children(&mut inner_cursor)
                .find(|c| c.kind() == "symbol")
                .and_then(|c| c.utf8_text(source).ok().map(|s| s.to_string()))
        } else if next.kind() == "symbol" {
            // (define x 42) → name is the symbol directly
            next.utf8_text(source).ok().map(|s| s.to_string())
        } else {
            None
        }
    } else {
        None
    };

    Some(RawSymbol {
        name,
        symbol_type: sym_type,
        start_byte: node.start_byte(),
        end_byte: node.end_byte(),
        start_row: node.start_position().row,
        end_row: node.end_position().row,
    })
}

/// Extract a symbol from a Racket `list` node for define/module/struct forms.
fn extract_racket_define(node: &tree_sitter::Node<'_>, source: &[u8]) -> Option<RawSymbol> {
    let mut cursor = node.walk();
    let children: Vec<_> = node.children(&mut cursor).collect();

    let first_sym = children.iter().find(|c| c.kind() == "symbol")?;
    let keyword = first_sym.utf8_text(source).ok()?;

    let sym_type = match keyword {
        "define" | "define-syntax" => SymbolType::Function,
        "module" | "module*" | "module+" => SymbolType::Module,
        "struct" => SymbolType::Struct,
        _ => return None,
    };

    let after_keyword: Vec<_> = children
        .iter()
        .skip_while(|c| c.start_byte() <= first_sym.start_byte())
        .filter(|c| c.kind() != "(" && c.kind() != ")")
        .collect();

    let name = if let Some(next) = after_keyword.first() {
        if next.kind() == "list" {
            let mut inner_cursor = next.walk();
            next.children(&mut inner_cursor)
                .find(|c| c.kind() == "symbol")
                .and_then(|c| c.utf8_text(source).ok().map(|s| s.to_string()))
        } else if next.kind() == "symbol" {
            next.utf8_text(source).ok().map(|s| s.to_string())
        } else {
            None
        }
    } else {
        None
    };

    Some(RawSymbol {
        name,
        symbol_type: sym_type,
        start_byte: node.start_byte(),
        end_byte: node.end_byte(),
        start_row: node.start_position().row,
        end_row: node.end_position().row,
    })
}

/// Extract a symbol from a Clojure `list_lit` node for defn/defmacro/ns/def forms.
///
/// Clojure AST: `list_lit` → first `sym_lit` child text is the keyword → second `sym_lit` is the name.
fn extract_clojure_define(node: &tree_sitter::Node<'_>, source: &[u8]) -> Option<RawSymbol> {
    let mut cursor = node.walk();
    let sym_lits: Vec<_> = node
        .children(&mut cursor)
        .filter(|c| c.kind() == "sym_lit")
        .collect();

    if sym_lits.len() < 2 {
        return None;
    }

    let keyword = sym_lits[0].utf8_text(source).ok()?;
    let sym_type = match keyword {
        "defn" | "defmacro" | "defn-" => SymbolType::Function,
        "ns" => SymbolType::Module,
        "def" | "defonce" => SymbolType::Variable,
        _ => return None,
    };

    let name = sym_lits[1].utf8_text(source).ok().map(|s| s.to_string());

    Some(RawSymbol {
        name,
        symbol_type: sym_type,
        start_byte: node.start_byte(),
        end_byte: node.end_byte(),
        start_row: node.start_position().row,
        end_row: node.end_position().row,
    })
}

/// Extract a symbol from a Common Lisp `defun` node.
///
/// CL AST: `defun` → `defun_header` child → `sym_lit` child is the name.
fn extract_commonlisp_defun(node: &tree_sitter::Node<'_>, source: &[u8]) -> Option<RawSymbol> {
    let mut cursor = node.walk();
    let name = node
        .children(&mut cursor)
        .find(|c| c.kind() == "defun_header")
        .and_then(|header| {
            let mut hcursor = header.walk();
            header
                .children(&mut hcursor)
                .find(|c| c.kind() == "sym_lit")
                .and_then(|s| s.utf8_text(source).ok().map(|t| t.to_string()))
        });

    Some(RawSymbol {
        name,
        symbol_type: SymbolType::Function,
        start_byte: node.start_byte(),
        end_byte: node.end_byte(),
        start_row: node.start_position().row,
        end_row: node.end_position().row,
    })
}

/// Extract a symbol from a Common Lisp `list_lit` node for defclass/defvar/defparameter etc.
///
/// CL AST for non-defun forms: `list_lit` → first `sym_lit` is keyword → second `sym_lit` is name.
fn extract_commonlisp_list(node: &tree_sitter::Node<'_>, source: &[u8]) -> Option<RawSymbol> {
    let mut cursor = node.walk();
    let sym_lits: Vec<_> = node
        .children(&mut cursor)
        .filter(|c| c.kind() == "sym_lit")
        .collect();

    if sym_lits.len() < 2 {
        return None;
    }

    let keyword = sym_lits[0].utf8_text(source).ok()?;
    let sym_type = match keyword {
        "defclass" => SymbolType::Class,
        "defvar" | "defparameter" | "defconstant" => SymbolType::Variable,
        "defpackage" => SymbolType::Module,
        "defmacro" | "defgeneric" | "defmethod" => SymbolType::Function,
        _ => return None,
    };

    let name = sym_lits[1].utf8_text(source).ok().map(|s| s.to_string());

    Some(RawSymbol {
        name,
        symbol_type: sym_type,
        start_byte: node.start_byte(),
        end_byte: node.end_byte(),
        start_row: node.start_position().row,
        end_row: node.end_position().row,
    })
}

fn extract_elixir_name(node: &tree_sitter::Node<'_>, source: &[u8]) -> Option<String> {
    let mut cursor = node.walk();
    let mut args = None;
    for child in node.children(&mut cursor) {
        if child.kind() == "arguments" {
            args = Some(child);
            break;
        }
    }

    if let Some(args_node) = args {
        if args_node.named_child_count() > 0 {
            if let Some(first_arg) = args_node.named_child(0) {
                if first_arg.kind() == "call" {
                    if let Some(target) = first_arg.child_by_field_name("target") {
                        return target.utf8_text(source).ok().map(|s| s.to_string());
                    }
                }
                let mut inner_cursor = first_arg.walk();
                for child in first_arg.children(&mut inner_cursor) {
                    if child.kind() == "identifier" || child.kind() == "alias" {
                        return child.utf8_text(source).ok().map(|s| s.to_string());
                    }
                }
                return first_arg.utf8_text(source).ok().map(|s| s.to_string());
            }
        }
    }
    None
}

fn get_elixir_do_block<'a>(node: &tree_sitter::Node<'a>) -> Option<tree_sitter::Node<'a>> {
    let mut cursor = node.walk();
    node.children(&mut cursor)
        .find(|child| child.kind() == "do_block")
}

/// Refine a Go type_spec into the correct SymbolType based on the type child.
fn refine_go_type_spec(node: &tree_sitter::Node<'_>, source: &[u8]) -> SymbolType {
    if let Some(type_child) = node.child_by_field_name("type") {
        match type_child.kind() {
            "struct_type" => return SymbolType::Struct,
            "interface_type" => return SymbolType::Interface,
            _ => {}
        }
    }
    // Check the text for common patterns
    let text = node.utf8_text(source).unwrap_or("");
    if text.contains("struct") {
        SymbolType::Struct
    } else if text.contains("interface") {
        SymbolType::Interface
    } else {
        SymbolType::TypeAlias
    }
}

/// Extract individual methods from a Rust `impl` block as separate symbols.
fn extract_impl_methods(
    impl_node: tree_sitter::Node<'_>,
    source: &[u8],
    lang: Language,
    symbols: &mut Vec<RawSymbol>,
) {
    let name = extract_name(&impl_node, source);
    symbols.push(RawSymbol {
        name,
        symbol_type: SymbolType::Block,
        start_byte: impl_node.start_byte(),
        end_byte: impl_node.end_byte(),
        start_row: impl_node.start_position().row,
        end_row: impl_node.end_position().row,
    });

    let mut cursor = impl_node.walk();

    for child in impl_node.children(&mut cursor) {
        if child.kind() == "declaration_list" {
            let mut inner_cursor = child.walk();
            for item in child.children(&mut inner_cursor) {
                if item.kind() == "function_item" {
                    let name = extract_name(&item, source);
                    symbols.push(RawSymbol {
                        name,
                        symbol_type: SymbolType::Method,
                        start_byte: item.start_byte(),
                        end_byte: item.end_byte(),
                        start_row: item.start_position().row,
                        end_row: item.end_position().row,
                    });
                } else if let Some(sym_type) = classify_node(lang, item.kind()) {
                    let name = extract_name(&item, source);
                    symbols.push(RawSymbol {
                        name,
                        symbol_type: sym_type,
                        start_byte: item.start_byte(),
                        end_byte: item.end_byte(),
                        start_row: item.start_position().row,
                        end_row: item.end_position().row,
                    });
                }
            }
        }
    }
}

/// Extract methods from a Python `class_definition` as separate symbols.
///
/// Similar to how Rust `impl` methods are extracted: the class body is
/// walked for `function_definition` nodes (methods) which become individual
/// [`Method`] chunks, while the class itself remains indexable as a full
/// [`Class`] symbol for definition-oriented queries.
fn extract_python_class_methods(
    class_node: tree_sitter::Node<'_>,
    source: &[u8],
    symbols: &mut Vec<RawSymbol>,
) {
    let name = extract_name(&class_node, source);
    symbols.push(RawSymbol {
        name,
        symbol_type: SymbolType::Class,
        start_byte: class_node.start_byte(),
        end_byte: class_node.end_byte(),
        start_row: class_node.start_position().row,
        end_row: class_node.end_position().row,
    });

    // The class body is a `block` child.
    let mut cursor = class_node.walk();
    for child in class_node.children(&mut cursor) {
        if child.kind() == "block" {
            let mut inner_cursor = child.walk();
            for item in child.children(&mut inner_cursor) {
                // Direct function_definition in class body = method.
                if item.kind() == "function_definition" {
                    let name = extract_name(&item, source);
                    symbols.push(RawSymbol {
                        name,
                        symbol_type: SymbolType::Method,
                        start_byte: item.start_byte(),
                        end_byte: item.end_byte(),
                        start_row: item.start_position().row,
                        end_row: item.end_position().row,
                    });
                }
                // Decorated methods: decorated_definition > function_definition
                else if item.kind() == "decorated_definition" {
                    let mut dec_cursor = item.walk();
                    for dec_child in item.children(&mut dec_cursor) {
                        if dec_child.kind() == "function_definition" {
                            let name = extract_name(&dec_child, source);
                            symbols.push(RawSymbol {
                                name,
                                symbol_type: SymbolType::Method,
                                // Use the decorated_definition range to include
                                // the decorator in the chunk.
                                start_byte: item.start_byte(),
                                end_byte: item.end_byte(),
                                start_row: item.start_position().row,
                                end_row: item.end_position().row,
                            });
                        }
                    }
                }
            }
        }
    }
}

fn extract_general_class_methods(
    class_node: tree_sitter::Node<'_>,
    source: &[u8],
    lang: Language,
    symbols: &mut Vec<RawSymbol>,
    class_sym_type: SymbolType,
) {
    // ALWAYS push the class/wrapper itself
    let name = extract_name(&class_node, source);
    symbols.push(RawSymbol {
        name,
        symbol_type: class_sym_type,
        start_byte: class_node.start_byte(),
        end_byte: class_node.end_byte(),
        start_row: class_node.start_position().row,
        end_row: class_node.end_position().row,
    });

    let mut cursor = class_node.walk();

    for child in class_node.children(&mut cursor) {
        let ckind = child.kind();
        if ckind.contains("body") || ckind.contains("block") || ckind == "declaration_list" {
            let mut inner_cursor = child.walk();
            for item in child.children(&mut inner_cursor) {
                if let Some(sym_type) = classify_node(lang, item.kind()) {
                    let mut end_byte = item.end_byte();
                    let mut end_row = item.end_position().row;

                    // Dart detached method body
                    if lang == Language::Dart && item.kind() == "method_signature" {
                        if let Some(next) = item.next_sibling() {
                            if next.kind() == "function_body" {
                                end_byte = next.end_byte();
                                end_row = next.end_position().row;
                            }
                        }
                    }

                    let name = extract_name(&item, source);
                    symbols.push(RawSymbol {
                        name,
                        symbol_type: sym_type,
                        start_byte: item.start_byte(),
                        end_byte,
                        start_row: item.start_position().row,
                        end_row,
                    });
                } else if lang == Language::Dart && item.kind() == "class_member" {
                    let mut cm_cursor = item.walk();
                    for cm_child in item.children(&mut cm_cursor) {
                        if let Some(sym_type) = classify_node(lang, cm_child.kind()) {
                            let mut end_byte = cm_child.end_byte();
                            let mut end_row = cm_child.end_position().row;

                            if cm_child.kind() == "method_signature" {
                                if let Some(next) = cm_child.next_sibling() {
                                    if next.kind() == "function_body" {
                                        end_byte = next.end_byte();
                                        end_row = next.end_position().row;
                                    }
                                }
                            }

                            let name = extract_name(&cm_child, source);
                            symbols.push(RawSymbol {
                                name,
                                symbol_type: sym_type,
                                start_byte: cm_child.start_byte(),
                                end_byte,
                                start_row: cm_child.start_position().row,
                                end_row,
                            });
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parsing::languages::tree_sitter_grammar;

    fn parse_and_extract(source: &str, lang: Language) -> Vec<RawSymbol> {
        let grammar = tree_sitter_grammar(lang).unwrap();
        let mut parser = tree_sitter::Parser::new();
        parser.set_language(&grammar).unwrap();
        let tree = parser.parse(source, None).unwrap();
        extract_symbols(&tree, source.as_bytes(), lang)
    }

    #[test]
    fn rust_extracts_functions() {
        let source = r#"
fn hello() {
    println!("hello");
}

fn world(x: i32) -> i32 {
    x + 1
}
"#;
        let symbols = parse_and_extract(source, Language::Rust);
        assert_eq!(symbols.len(), 2);
        assert_eq!(symbols[0].name.as_deref(), Some("hello"));
        assert_eq!(symbols[0].symbol_type, SymbolType::Function);
        assert_eq!(symbols[1].name.as_deref(), Some("world"));
    }

    #[test]
    fn rust_extracts_structs_and_enums() {
        let source = r#"
struct Point {
    x: f64,
    y: f64,
}

enum Color {
    Red,
    Green,
    Blue,
}
"#;
        let symbols = parse_and_extract(source, Language::Rust);
        assert_eq!(symbols.len(), 2);
        assert_eq!(symbols[0].symbol_type, SymbolType::Struct);
        assert_eq!(symbols[0].name.as_deref(), Some("Point"));
        assert_eq!(symbols[1].symbol_type, SymbolType::Enum);
        assert_eq!(symbols[1].name.as_deref(), Some("Color"));
    }

    #[test]
    fn rust_extracts_impl_methods() {
        let source = r#"
impl Point {
    fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    fn distance(&self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
        }
}
"#;
        let symbols = parse_and_extract(source, Language::Rust);
        assert_eq!(symbols.len(), 3);
        assert_eq!(symbols[0].symbol_type, SymbolType::Block);
        assert_eq!(symbols[0].name.as_deref(), Some("impl Point"));
        assert_eq!(symbols[1].symbol_type, SymbolType::Method);
        assert_eq!(symbols[1].name.as_deref(), Some("new"));
        assert_eq!(symbols[2].symbol_type, SymbolType::Method);
        assert_eq!(symbols[2].name.as_deref(), Some("distance"));
    }

    #[test]
    fn rust_extracts_trait_impl_names() {
        let source = r#"
impl Sink for StdoutSink {
    fn write(&self) {}
}
"#;
        let symbols = parse_and_extract(source, Language::Rust);
        assert_eq!(symbols[0].symbol_type, SymbolType::Block);
        assert_eq!(symbols[0].name.as_deref(), Some("impl Sink for StdoutSink"));
    }

    #[test]
    fn rust_extracts_traits() {
        let source = r#"
trait Drawable {
    fn draw(&self);
    fn area(&self) -> f64;
}
"#;
        let symbols = parse_and_extract(source, Language::Rust);
        assert_eq!(symbols.len(), 1);
        assert_eq!(symbols[0].symbol_type, SymbolType::Trait);
        assert_eq!(symbols[0].name.as_deref(), Some("Drawable"));
    }

    #[test]
    fn python_extracts_functions_and_class_methods() {
        let source = r#"
def hello():
    print("hello")

class MyClass:
    def __init__(self):
        self.x = 0

    def method(self):
        return self.x
"#;
        let symbols = parse_and_extract(source, Language::Python);
        // Should extract: hello (function), MyClass (class), __init__ (method), method (method)
        assert_eq!(symbols.len(), 4);
        assert_eq!(symbols[0].symbol_type, SymbolType::Function);
        assert_eq!(symbols[0].name.as_deref(), Some("hello"));
        assert_eq!(symbols[1].symbol_type, SymbolType::Class);
        assert_eq!(symbols[1].name.as_deref(), Some("MyClass"));
        assert_eq!(symbols[2].symbol_type, SymbolType::Method);
        assert_eq!(symbols[2].name.as_deref(), Some("__init__"));
        assert_eq!(symbols[3].symbol_type, SymbolType::Method);
        assert_eq!(symbols[3].name.as_deref(), Some("method"));
    }

    #[test]
    fn python_class_without_methods_kept_as_class() {
        let source = r#"
class Config:
    DEBUG = True
    VERSION = "1.0"
"#;
        let symbols = parse_and_extract(source, Language::Python);
        assert_eq!(symbols.len(), 1);
        assert_eq!(symbols[0].symbol_type, SymbolType::Class);
        assert_eq!(symbols[0].name.as_deref(), Some("Config"));
    }

    #[test]
    fn python_decorated_methods_extracted_separately() {
        let source = r#"
class MyClass:
    @staticmethod
    def static_method():
        pass

    @property
    def value(self):
        return self._value
"#;
        let symbols = parse_and_extract(source, Language::Python);
        assert_eq!(symbols.len(), 3);
        assert_eq!(symbols[0].symbol_type, SymbolType::Class);
        assert_eq!(symbols[0].name.as_deref(), Some("MyClass"));
        assert_eq!(symbols[1].symbol_type, SymbolType::Method);
        assert_eq!(symbols[1].name.as_deref(), Some("static_method"));
        assert_eq!(symbols[2].symbol_type, SymbolType::Method);
        assert_eq!(symbols[2].name.as_deref(), Some("value"));
    }

    #[test]
    fn typescript_extracts_functions_and_interfaces() {
        let source = r#"
function greet(name: string): string {
    return `Hello, ${name}!`;
}

interface User {
    name: string;
    age: number;
}

class UserService {
    private users: User[];

    getUser(id: number): User {
        return this.users[id];
    }
}
"#;
        let symbols = parse_and_extract(source, Language::TypeScript);
        assert!(
            symbols.len() >= 3,
            "expected >= 3 symbols, got {}",
            symbols.len()
        );

        let func = symbols.iter().find(|s| s.name.as_deref() == Some("greet"));
        assert!(func.is_some(), "should find function 'greet'");
        assert_eq!(func.unwrap().symbol_type, SymbolType::Function);

        let iface = symbols.iter().find(|s| s.name.as_deref() == Some("User"));
        assert!(iface.is_some(), "should find interface 'User'");
        assert_eq!(iface.unwrap().symbol_type, SymbolType::Interface);

        let class = symbols
            .iter()
            .find(|s| s.name.as_deref() == Some("UserService"));
        assert!(class.is_some(), "should find class 'UserService'");
        assert_eq!(class.unwrap().symbol_type, SymbolType::Class);
    }

    #[test]
    fn go_extracts_functions_and_structs() {
        let source = r#"
package main

func Hello() string {
    return "hello"
}

type Point struct {
    X float64
    Y float64
}

func (p *Point) Distance() float64 {
    return p.X * p.X + p.Y * p.Y
}
"#;
        let symbols = parse_and_extract(source, Language::Go);
        assert!(
            symbols.len() >= 3,
            "expected >= 3 symbols, got {}",
            symbols.len()
        );

        let func = symbols.iter().find(|s| s.name.as_deref() == Some("Hello"));
        assert!(func.is_some(), "should find function 'Hello'");
        assert_eq!(func.unwrap().symbol_type, SymbolType::Function);

        let struc = symbols.iter().find(|s| s.name.as_deref() == Some("Point"));
        assert!(struc.is_some(), "should find struct 'Point'");
        assert_eq!(struc.unwrap().symbol_type, SymbolType::Struct);

        let method = symbols
            .iter()
            .find(|s| s.name.as_deref() == Some("Distance"));
        assert!(method.is_some(), "should find method 'Distance'");
        assert_eq!(method.unwrap().symbol_type, SymbolType::Method);
    }

    #[test]
    fn java_extracts_class_and_methods() {
        let source = r#"
class Calculator {
    public int add(int a, int b) {
        return a + b;
    }

    public int multiply(int a, int b) {
        return a * b;
    }
}
"#;
        let symbols = parse_and_extract(source, Language::Java);
        assert!(
            symbols.len() >= 1,
            "expected >= 1 symbol, got {}",
            symbols.len()
        );

        let class = symbols
            .iter()
            .find(|s| s.name.as_deref() == Some("Calculator"));
        assert!(class.is_some(), "should find class 'Calculator'");
        assert_eq!(class.unwrap().symbol_type, SymbolType::Class);
    }

    #[test]
    fn c_extracts_functions_and_structs() {
        let source = r#"
struct Point {
    double x;
    double y;
};

double distance(struct Point* p) {
    return p->x * p->x + p->y * p->y;
}
"#;
        let symbols = parse_and_extract(source, Language::C);
        assert!(
            symbols.len() >= 1,
            "expected >= 1 symbol, got {}",
            symbols.len()
        );

        let func = symbols
            .iter()
            .find(|s| s.symbol_type == SymbolType::Function);
        assert!(func.is_some(), "should find a function");
        assert_eq!(func.unwrap().name.as_deref(), Some("distance"));
    }

    #[test]
    fn cpp_extracts_classes_and_functions() {
        let source = r#"
class Shape {
public:
    virtual double area() = 0;
};

namespace geometry {
    double pi() {
        return 3.14159;
    }
}
"#;
        let symbols = parse_and_extract(source, Language::Cpp);
        assert!(
            symbols.len() >= 1,
            "expected >= 1 symbol, got {}",
            symbols.len()
        );

        let class = symbols.iter().find(|s| s.symbol_type == SymbolType::Class);
        assert!(class.is_some(), "should find a class");

        let ns = symbols.iter().find(|s| s.symbol_type == SymbolType::Module);
        assert!(ns.is_some(), "should find a namespace");
    }

    #[test]
    fn ruby_extracts_functions_and_classes() {
        let source = r#"
module MyModule
end

class MyClass
end

def my_method
end
"#;
        let symbols = parse_and_extract(source, Language::Ruby);
        assert!(symbols.len() >= 3);
        let m = symbols.iter().find(|s| s.symbol_type == SymbolType::Module);
        assert!(m.is_some());
        assert_eq!(m.unwrap().name.as_deref(), Some("MyModule"));

        let c = symbols.iter().find(|s| s.symbol_type == SymbolType::Class);
        assert!(c.is_some());
        assert_eq!(c.unwrap().name.as_deref(), Some("MyClass"));

        let f = symbols
            .iter()
            .find(|s| s.name.as_deref() == Some("my_method"));
        assert!(f.is_some());
        assert_eq!(f.unwrap().symbol_type, SymbolType::Function);
    }

    #[test]
    fn bash_extracts_functions() {
        let source = r#"
function foo() {
    echo "foo"
}
bar() {
    echo "bar"
}
"#;
        let symbols = parse_and_extract(source, Language::Bash);
        assert_eq!(symbols.len(), 2);
        assert_eq!(symbols[0].symbol_type, SymbolType::Function);
        assert_eq!(symbols[0].name.as_deref(), Some("foo"));
        assert_eq!(symbols[1].symbol_type, SymbolType::Function);
        assert_eq!(symbols[1].name.as_deref(), Some("bar"));
    }

    #[test]
    fn kotlin_extracts_types_and_functions() {
        let source = r#"
fun foo() {}
class Bar {}
interface Baz {}
object Qux {}
enum class Quux {}
"#;
        let symbols = parse_and_extract(source, Language::Kotlin);
        assert_eq!(symbols.len(), 5);

        let fun = symbols
            .iter()
            .find(|s| s.name.as_deref() == Some("foo"))
            .unwrap();
        assert_eq!(fun.symbol_type, SymbolType::Function);

        let cls = symbols
            .iter()
            .find(|s| s.name.as_deref() == Some("Bar"))
            .unwrap();
        assert_eq!(cls.symbol_type, SymbolType::Class);

        let iface = symbols
            .iter()
            .find(|s| s.name.as_deref() == Some("Baz"))
            .unwrap();
        assert_eq!(iface.symbol_type, SymbolType::Interface);

        let obj = symbols
            .iter()
            .find(|s| s.name.as_deref() == Some("Qux"))
            .unwrap();
        assert_eq!(obj.symbol_type, SymbolType::Class);

        let enm = symbols
            .iter()
            .find(|s| s.name.as_deref() == Some("Quux"))
            .unwrap();
        assert_eq!(enm.symbol_type, SymbolType::Enum);
    }

    #[test]
    fn swift_extracts_types_and_functions() {
        let source = r#"
func foo() {}
class Bar {}
struct Baz {}
enum Qux {}
protocol Quux {}
"#;
        let symbols = parse_and_extract(source, Language::Swift);
        assert_eq!(symbols.len(), 5);

        let fun = symbols
            .iter()
            .find(|s| s.name.as_deref() == Some("foo"))
            .unwrap();
        assert_eq!(fun.symbol_type, SymbolType::Function);

        let cls = symbols
            .iter()
            .find(|s| s.name.as_deref() == Some("Bar"))
            .unwrap();
        assert_eq!(cls.symbol_type, SymbolType::Class);

        let strc = symbols
            .iter()
            .find(|s| s.name.as_deref() == Some("Baz"))
            .unwrap();
        assert_eq!(strc.symbol_type, SymbolType::Struct);

        let enm = symbols
            .iter()
            .find(|s| s.name.as_deref() == Some("Qux"))
            .unwrap();
        assert_eq!(enm.symbol_type, SymbolType::Enum);

        let proto = symbols
            .iter()
            .find(|s| s.name.as_deref() == Some("Quux"))
            .unwrap();
        assert_eq!(proto.symbol_type, SymbolType::Interface);
    }

    #[test]
    fn zig_extracts_functions_and_structs() {
        let source = r#"
fn foo() void {}
const Bar = struct {};
"#;
        let symbols = parse_and_extract(source, Language::Zig);
        assert_eq!(symbols.len(), 2);

        let fun = symbols
            .iter()
            .find(|s| s.name.as_deref() == Some("foo"))
            .unwrap();
        assert_eq!(fun.symbol_type, SymbolType::Function);

        let strc = symbols
            .iter()
            .find(|s| s.name.as_deref() == Some("Bar"))
            .unwrap();
        assert_eq!(strc.symbol_type, SymbolType::Struct);
    }

    #[test]
    fn lua_extracts_functions() {
        let source = r#"
function foo() end
local function bar() end
"#;
        let symbols = parse_and_extract(source, Language::Lua);
        assert_eq!(symbols.len(), 2);

        let foo = symbols
            .iter()
            .find(|s| s.name.as_deref() == Some("foo"))
            .unwrap();
        assert_eq!(foo.symbol_type, SymbolType::Function);

        let bar = symbols
            .iter()
            .find(|s| s.name.as_deref() == Some("bar"))
            .unwrap();
        assert_eq!(bar.symbol_type, SymbolType::Function);
    }

    #[test]
    fn scala_extracts_types_and_functions() {
        let source = r#"
def main(args: Array[String]): Unit = {}
class User {}
trait Logger {}
"#;
        let symbols = parse_and_extract(source, Language::Scala);
        println!("Scala symbols: {:#?}", symbols);
        assert!(symbols
            .iter()
            .any(|s| s.symbol_type == SymbolType::Function && s.name.as_deref() == Some("main")));
        assert!(
            symbols
                .iter()
                .any(|s| s.symbol_type == SymbolType::Class && s.name.as_deref() == Some("User"))
        );
        assert!(
            symbols
                .iter()
                .any(|s| s.symbol_type == SymbolType::Trait && s.name.as_deref() == Some("Logger"))
        );
    }

    #[test]
    fn csharp_extracts_types_and_methods() {
        let source = r#"
namespace MyApp {
    class Calculator {
        public int Add(int a, int b) { return a + b; }
    }
    interface IWorker {}
    struct Point {}
    enum Status { Ok, Error }
}
"#;
        let symbols = parse_and_extract(source, Language::CSharp);
        println!("CSharp symbols: {:#?}", symbols);
        assert!(symbols.iter().any(|s| s.symbol_type == SymbolType::Module));
        assert!(symbols.iter().any(|s| s.symbol_type == SymbolType::Class));
        assert!(symbols.iter().any(|s| s.symbol_type == SymbolType::Method));
        assert!(
            symbols
                .iter()
                .any(|s| s.symbol_type == SymbolType::Interface)
        );
        assert!(symbols.iter().any(|s| s.symbol_type == SymbolType::Struct));
        assert!(symbols.iter().any(|s| s.symbol_type == SymbolType::Enum));
    }

    #[test]
    fn php_extracts_classes_and_functions() {
        let source = r#"
<?php
function foo() {}
class Bar {
    public function baz() {}
}
interface Qux {}
"#;
        let symbols = parse_and_extract(source, Language::Php);
        println!("PHP symbols: {:#?}", symbols);
        assert!(
            symbols
                .iter()
                .any(|s| s.symbol_type == SymbolType::Function)
        );
        assert!(symbols.iter().any(|s| s.symbol_type == SymbolType::Class));
        assert!(symbols.iter().any(|s| s.symbol_type == SymbolType::Method));
        assert!(
            symbols
                .iter()
                .any(|s| s.symbol_type == SymbolType::Interface)
        );
    }

    #[test]
    fn haskell_extracts_types_and_functions() {
        let source = r#"
data Point = Point Float Float
type Age = Int
add :: Int -> Int -> Int
add x y = x + y
"#;
        let symbols = parse_and_extract(source, Language::Haskell);
        println!("Haskell symbols: {:#?}", symbols);
        assert!(symbols.iter().any(|s| s.symbol_type == SymbolType::Struct));
        assert!(
            symbols
                .iter()
                .any(|s| s.symbol_type == SymbolType::TypeAlias)
        );
        assert!(
            symbols
                .iter()
                .any(|s| s.symbol_type == SymbolType::Function)
        );
    }

    #[test]
    fn elixir_extracts_modules_and_functions() {
        let source = r#"
defmodule Math do
  def add(a, b) do
    a + b
  end
  defp sub(a, b), do: a - b
  defmacro mul(a, b) do
  end
end
"#;
        let symbols = parse_and_extract(source, Language::Elixir);
        println!("Elixir symbols: {:#?}", symbols);
        assert!(
            symbols
                .iter()
                .any(|s| s.symbol_type == SymbolType::Module && s.name.as_deref() == Some("Math"))
        );
        assert!(
            symbols
                .iter()
                .any(|s| s.symbol_type == SymbolType::Function && s.name.as_deref() == Some("add"))
        );
        assert!(
            symbols
                .iter()
                .any(|s| s.symbol_type == SymbolType::Function && s.name.as_deref() == Some("sub"))
        );
        assert!(
            symbols
                .iter()
                .any(|s| s.symbol_type == SymbolType::Function && s.name.as_deref() == Some("mul"))
        );
    }

    #[test]
    fn hcl_extracts_blocks() {
        let source = r#"
resource "aws_instance" "web" {
  ami = "ami-123"
}
module "vpc" {}
data "aws_ami" "ubuntu" {}
variable "image_id" {}
output "instance_ip" {}
"#;
        let symbols = parse_and_extract(source, Language::Hcl);
        println!("HCL symbols: {:#?}", symbols);
        assert!(
            symbols.iter().any(|s| s.symbol_type == SymbolType::Struct
                && s.name.as_deref() == Some("aws_instance"))
        );
        assert!(
            symbols
                .iter()
                .any(|s| s.symbol_type == SymbolType::Module && s.name.as_deref() == Some("vpc"))
        );
        assert!(
            symbols.iter().any(
                |s| s.symbol_type == SymbolType::Struct && s.name.as_deref() == Some("aws_ami")
            )
        );
        assert!(symbols.iter().any(
            |s| s.symbol_type == SymbolType::TypeAlias && s.name.as_deref() == Some("image_id")
        ));
        assert!(
            symbols
                .iter()
                .any(|s| s.symbol_type == SymbolType::TypeAlias
                    && s.name.as_deref() == Some("instance_ip"))
        );
    }

    #[test]
    fn sql_extracts_statements() {
        let source = r#"
CREATE TABLE users (id INT);
CREATE FUNCTION get_user() RETURNS INT AS $$ SELECT 1 $$ LANGUAGE SQL;
CREATE VIEW active_users AS SELECT * FROM users;
CREATE PROCEDURE my_proc() LANGUAGE SQL AS $$ $$;
"#;
        let symbols = parse_and_extract(source, Language::Sql);
        println!("SQL symbols: {:#?}", symbols);
        assert!(!symbols.is_empty());
    }

    #[test]
    fn protobuf_extracts_types() {
        let source = r#"
syntax = "proto3";
message User { int32 id = 1; }
enum Status { ACTIVE = 0; }
service AuthService { rpc Login(User) returns (User); }
"#;
        let symbols = parse_and_extract(source, Language::Protobuf);
        println!("Protobuf symbols: {:#?}", symbols);
        assert!(!symbols.is_empty());
        assert!(
            symbols
                .iter()
                .any(|s| s.symbol_type == SymbolType::Struct && s.name.as_deref() == Some("User"))
        );
        assert!(
            symbols
                .iter()
                .any(|s| s.symbol_type == SymbolType::Enum && s.name.as_deref() == Some("Status"))
        );
        assert!(symbols.iter().any(
            |s| s.symbol_type == SymbolType::Class && s.name.as_deref() == Some("AuthService")
        ));
        assert!(
            symbols
                .iter()
                .any(|s| s.symbol_type == SymbolType::Method && s.name.as_deref() == Some("Login"))
        );
    }

    // ── Tier 1B symbol extraction tests ─────────────────────────

    #[test]
    fn html_extracts_elements() {
        let source = r#"
<html>
<head><title>Test</title></head>
<body>
  <div class="container">
    <p>Hello</p>
  </div>
  <script>console.log("hi")</script>
  <style>body { color: red; }</style>
</body>
</html>
"#;
        let symbols = parse_and_extract(source, Language::Html);
        assert!(!symbols.is_empty(), "HTML should extract some symbols");
        assert!(
            symbols.iter().any(|s| s.symbol_type == SymbolType::Block),
            "HTML should extract block symbols"
        );
    }

    #[test]
    fn css_extracts_rules() {
        let source = r#"
body {
    color: red;
    font-size: 16px;
}

.container {
    max-width: 1200px;
}

@media (max-width: 768px) {
    body { font-size: 14px; }
}
"#;
        let symbols = parse_and_extract(source, Language::Css);
        assert!(!symbols.is_empty(), "CSS should extract some symbols");
        assert!(
            symbols.iter().any(|s| s.symbol_type == SymbolType::Block),
            "CSS should extract rule_set blocks"
        );
    }

    #[test]
    fn scss_extracts_mixins_and_rules() {
        let source = r#"
$primary: #333;

@mixin flex-center {
    display: flex;
    align-items: center;
}

.container {
    @include flex-center;
    color: $primary;
}
"#;
        let symbols = parse_and_extract(source, Language::Scss);
        assert!(!symbols.is_empty(), "SCSS should extract some symbols");
        assert!(
            symbols
                .iter()
                .any(|s| s.symbol_type == SymbolType::Function),
            "SCSS should extract mixin as function"
        );
        assert!(
            symbols.iter().any(|s| s.symbol_type == SymbolType::Block),
            "SCSS should extract rule_set blocks"
        );
    }

    #[test]
    fn vue_extracts_sections() {
        let source = r#"
<template>
  <div>{{ message }}</div>
</template>

<script>
export default {
  data() { return { message: "Hello" } }
}
</script>

<style>
.container { color: red; }
</style>
"#;
        let symbols = parse_and_extract(source, Language::Vue);
        assert!(!symbols.is_empty(), "Vue should extract some symbols");
        assert!(
            symbols.iter().any(|s| s.symbol_type == SymbolType::Block),
            "Vue should extract template/script/style blocks"
        );
    }

    #[test]
    fn graphql_extracts_types_and_queries() {
        let source = r#"
type User {
    id: ID!
    name: String!
    email: String
}

enum Role {
    ADMIN
    USER
    GUEST
}

interface Node {
    id: ID!
}

input CreateUserInput {
    name: String!
    email: String!
}

query GetUser($id: ID!) {
    user(id: $id) {
        name
        email
    }
}
"#;
        let symbols = parse_and_extract(source, Language::GraphQl);
        assert!(!symbols.is_empty(), "GraphQL should extract symbols");
        assert!(
            symbols.iter().any(|s| s.symbol_type == SymbolType::Struct),
            "GraphQL should extract type as struct"
        );
        assert!(
            symbols.iter().any(|s| s.symbol_type == SymbolType::Enum),
            "GraphQL should extract enum"
        );
        assert!(
            symbols
                .iter()
                .any(|s| s.symbol_type == SymbolType::Interface),
            "GraphQL should extract interface"
        );
    }

    #[test]
    fn cmake_extracts_functions_and_commands() {
        let source = r#"
cmake_minimum_required(VERSION 3.10)
project(MyProject)

function(my_helper ARG)
    message(STATUS "${ARG}")
endfunction()

macro(my_macro)
    set(MY_VAR "value")
endmacro()

add_executable(main src/main.cpp)
"#;
        let symbols = parse_and_extract(source, Language::CMake);
        assert!(!symbols.is_empty(), "CMake should extract symbols");
    }

    #[test]
    fn dockerfile_extracts_instructions() {
        let source = r#"FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
ARG BUILD_VERSION=1.0

RUN apt-get update && apt-get install -y curl

COPY . /app
WORKDIR /app

CMD ["./app"]
"#;
        let symbols = parse_and_extract(source, Language::Dockerfile);
        assert!(!symbols.is_empty(), "Dockerfile should extract symbols");
        assert!(
            symbols.iter().any(|s| s.symbol_type == SymbolType::Block),
            "Dockerfile should extract FROM as block"
        );
        assert!(
            symbols
                .iter()
                .any(|s| s.symbol_type == SymbolType::Constant),
            "Dockerfile should extract ENV/ARG as constant"
        );
    }

    #[test]
    fn xml_extracts_elements() {
        let source = r#"<?xml version="1.0" encoding="UTF-8"?>
<project>
    <name>MyProject</name>
    <dependencies>
        <dependency>
            <groupId>com.example</groupId>
            <artifactId>lib</artifactId>
        </dependency>
    </dependencies>
</project>
"#;
        let symbols = parse_and_extract(source, Language::Xml);
        assert!(!symbols.is_empty(), "XML should extract some symbols");
        assert!(
            symbols.iter().any(|s| s.symbol_type == SymbolType::Block),
            "XML should extract elements as blocks"
        );
    }

    // ── Tier 2A symbol extraction tests ─────────────────────────

    #[test]
    fn objectivec_extracts_classes_and_methods() {
        let source = r#"
@interface Calculator : NSObject
- (int)add:(int)a to:(int)b;
@end

@implementation Calculator
- (int)add:(int)a to:(int)b {
    return a + b;
}
@end
"#;
        let symbols = parse_and_extract(source, Language::ObjectiveC);
        assert!(!symbols.is_empty(), "ObjC should extract symbols");
        assert!(
            symbols.iter().any(|s| s.symbol_type == SymbolType::Class),
            "ObjC should extract class interface/implementation"
        );
        // Methods are extracted inside class body
        assert!(
            symbols.iter().any(|s| s.symbol_type == SymbolType::Method),
            "ObjC should extract methods"
        );
    }

    #[test]
    fn perl_extracts_functions() {
        let source = r#"
package MyModule;

sub hello {
    print "hello\n";
}

sub world {
    my ($name) = @_;
    print "hello $name\n";
}

1;
"#;
        let symbols = parse_and_extract(source, Language::Perl);
        println!("Perl symbols: {:#?}", symbols);
        assert!(!symbols.is_empty(), "Perl should extract symbols");
        assert!(
            symbols
                .iter()
                .any(|s| s.symbol_type == SymbolType::Function),
            "Perl should extract functions"
        );
    }

    #[test]
    fn julia_extracts_functions_and_structs() {
        let source = r#"
function hello()
    println("hello")
end

struct Point
    x::Float64
    y::Float64
end

module MyModule
end
"#;
        let symbols = parse_and_extract(source, Language::Julia);
        println!("Julia symbols: {:#?}", symbols);
        assert!(!symbols.is_empty(), "Julia should extract symbols");
        assert!(
            symbols
                .iter()
                .any(|s| s.symbol_type == SymbolType::Function),
            "Julia should extract functions"
        );
        assert!(
            symbols.iter().any(|s| s.symbol_type == SymbolType::Struct),
            "Julia should extract structs"
        );
    }

    #[test]
    fn nix_extracts_bindings() {
        let source = r#"
{
  hello = "world";
  foo = x: x + 1;
}
"#;
        let symbols = parse_and_extract(source, Language::Nix);
        println!("Nix symbols: {:#?}", symbols);
        assert!(!symbols.is_empty(), "Nix should extract symbols");
    }

    #[test]
    fn ocaml_extracts_functions_and_types() {
        let source = r#"
let hello () = print_endline "hello"

type point = { x: float; y: float }

module MyModule = struct end
"#;
        let symbols = parse_and_extract(source, Language::OCaml);
        println!("OCaml symbols: {:#?}", symbols);
        assert!(!symbols.is_empty(), "OCaml should extract symbols");
        assert!(
            symbols
                .iter()
                .any(|s| s.symbol_type == SymbolType::Function),
            "OCaml should extract functions"
        );
    }

    #[test]
    fn groovy_extracts_classes_and_functions() {
        let source = r#"
class Calculator {
    int add(int a, int b) {
        return a + b
    }
}

def hello() {
    println "hello"
}
"#;
        let symbols = parse_and_extract(source, Language::Groovy);
        println!("Groovy symbols: {:#?}", symbols);
        assert!(!symbols.is_empty(), "Groovy should extract symbols");
    }

    #[test]
    fn clojure_extracts_functions() {
        let source = r#"
(ns myapp.core)

(defn hello []
  (println "hello"))

(defn add [a b]
  (+ a b))
"#;
        let symbols = parse_and_extract(source, Language::Clojure);
        println!("Clojure symbols: {:#?}", symbols);
        assert!(
            symbols.len() >= 3,
            "Clojure should extract at least 3 symbols (ns + 2 defn), got {}",
            symbols.len()
        );
        assert!(
            symbols
                .iter()
                .any(|s| s.name.as_deref() == Some("myapp.core")
                    && s.symbol_type == SymbolType::Module),
            "Clojure should extract ns 'myapp.core'"
        );
        assert!(
            symbols.iter().any(
                |s| s.name.as_deref() == Some("hello") && s.symbol_type == SymbolType::Function
            ),
            "Clojure should extract function 'hello'"
        );
        assert!(
            symbols
                .iter()
                .any(|s| s.name.as_deref() == Some("add") && s.symbol_type == SymbolType::Function),
            "Clojure should extract function 'add'"
        );
    }

    #[test]
    fn clojure_extracts_defmacro() {
        let source = "(defmacro my-when [test & body]\n  `(if ~test (do ~@body)))\n";
        let symbols = parse_and_extract(source, Language::Clojure);
        println!("Clojure macro symbols: {:#?}", symbols);
        assert!(
            symbols
                .iter()
                .any(|s| s.name.as_deref() == Some("my-when")
                    && s.symbol_type == SymbolType::Function),
            "Clojure should extract defmacro 'my-when'"
        );
    }

    #[test]
    fn clojure_extracts_def() {
        let source = "(def pi 3.14159)\n";
        let symbols = parse_and_extract(source, Language::Clojure);
        println!("Clojure def symbols: {:#?}", symbols);
        assert!(
            symbols
                .iter()
                .any(|s| s.name.as_deref() == Some("pi") && s.symbol_type == SymbolType::Variable),
            "Clojure should extract def 'pi'"
        );
    }

    #[test]
    fn commonlisp_extracts_functions() {
        let source = r#"
(defun hello ()
  (format t "hello~%"))

(defun add (a b)
  (+ a b))

(defclass point ()
  ((x :initarg :x)
   (y :initarg :y)))
"#;
        let symbols = parse_and_extract(source, Language::CommonLisp);
        println!("Common Lisp symbols: {:#?}", symbols);
        assert!(
            symbols.len() >= 3,
            "CL should extract at least 3 symbols (2 defun + defclass), got {}",
            symbols.len()
        );
        assert!(
            symbols.iter().any(
                |s| s.name.as_deref() == Some("hello") && s.symbol_type == SymbolType::Function
            ),
            "CL should extract function 'hello'"
        );
        assert!(
            symbols
                .iter()
                .any(|s| s.name.as_deref() == Some("add") && s.symbol_type == SymbolType::Function),
            "CL should extract function 'add'"
        );
        assert!(
            symbols
                .iter()
                .any(|s| s.name.as_deref() == Some("point") && s.symbol_type == SymbolType::Class),
            "CL should extract class 'point'"
        );
    }

    #[test]
    fn commonlisp_extracts_defvar() {
        let source = "(defvar *my-var* 42)\n";
        let symbols = parse_and_extract(source, Language::CommonLisp);
        println!("CL defvar symbols: {:#?}", symbols);
        assert!(
            symbols
                .iter()
                .any(|s| s.name.as_deref() == Some("*my-var*")
                    && s.symbol_type == SymbolType::Variable),
            "CL should extract defvar '*my-var*'"
        );
    }

    #[test]
    fn erlang_extracts_functions() {
        let source = r#"
-module(hello).
-export([hello/0, add/2]).

hello() -> ok.

add(A, B) -> A + B.
"#;
        let symbols = parse_and_extract(source, Language::Erlang);
        println!("Erlang symbols: {:#?}", symbols);
        assert!(!symbols.is_empty(), "Erlang should extract symbols");
        assert!(
            symbols
                .iter()
                .any(|s| s.symbol_type == SymbolType::Function),
            "Erlang should extract functions"
        );
    }

    #[test]
    fn fsharp_extracts_functions() {
        let source = r#"
let hello () = printfn "hello"

let add a b = a + b

type Point = { X: float; Y: float }
"#;
        let symbols = parse_and_extract(source, Language::FSharp);
        println!("F# symbols: {:#?}", symbols);
        assert!(!symbols.is_empty(), "F# should extract symbols");
    }

    #[test]
    fn fortran_extracts_functions_and_subroutines() {
        let source = r#"
program hello
  print *, 'hello'
end program hello

subroutine greet(name)
  character(*), intent(in) :: name
  print *, 'Hello ', name
end subroutine greet

function add(a, b) result(c)
  integer, intent(in) :: a, b
  integer :: c
  c = a + b
end function add
"#;
        let symbols = parse_and_extract(source, Language::Fortran);
        println!("Fortran symbols: {:#?}", symbols);
        assert!(!symbols.is_empty(), "Fortran should extract symbols");
    }

    #[test]
    fn powershell_extracts_functions_and_classes() {
        let source = r#"
function Get-Hello {
    Write-Host "hello"
}

function Add-Numbers {
    param($a, $b)
    return $a + $b
}

class Calculator {
    [int] Add([int]$a, [int]$b) {
        return $a + $b
    }
}
"#;
        let symbols = parse_and_extract(source, Language::PowerShell);
        println!("PowerShell symbols: {:#?}", symbols);
        assert!(!symbols.is_empty(), "PowerShell should extract symbols");
        assert!(
            symbols
                .iter()
                .any(|s| s.symbol_type == SymbolType::Function),
            "PowerShell should extract functions"
        );
    }

    #[test]
    fn r_extracts_functions() {
        let source = r#"
hello <- function() {
  print("hello")
}

add <- function(a, b) {
  a + b
}
"#;
        let symbols = parse_and_extract(source, Language::R);
        assert!(!symbols.is_empty(), "R should extract symbols");
        assert!(
            symbols
                .iter()
                .any(|s| s.symbol_type == SymbolType::Function),
            "R should extract functions"
        );
        assert!(
            symbols.iter().any(|s| s.name.as_deref() == Some("hello")),
            "R should extract function name 'hello'"
        );
    }

    // ── Tier 2A batch 2 symbol extraction tests ─────────────────

    #[test]
    fn matlab_extracts_functions() {
        let source = r#"
function y = square(x)
  y = x^2;
end

function result = add(a, b)
  result = a + b;
end
"#;
        let symbols = parse_and_extract(source, Language::Matlab);
        println!("MATLAB symbols: {:#?}", symbols);
        assert!(!symbols.is_empty(), "MATLAB should extract symbols");
        assert!(
            symbols
                .iter()
                .any(|s| s.symbol_type == SymbolType::Function),
            "MATLAB should extract functions"
        );
    }

    #[test]
    fn dlang_extracts_functions_and_classes() {
        let source = r#"
void main() {
    writeln("hello");
}

class Foo {
    int bar() { return 42; }
}

struct Point {
    float x, y;
}
"#;
        let symbols = parse_and_extract(source, Language::DLang);
        println!("D symbols: {:#?}", symbols);
        assert!(!symbols.is_empty(), "D should extract symbols");
        assert!(
            symbols
                .iter()
                .any(|s| s.symbol_type == SymbolType::Function),
            "D should extract functions"
        );
    }

    #[test]
    fn fish_extracts_functions() {
        let source = r#"
function hello
  echo hello
end

function greet -a name
  echo "Hello, $name"
end
"#;
        let symbols = parse_and_extract(source, Language::Fish);
        println!("Fish symbols: {:#?}", symbols);
        assert!(!symbols.is_empty(), "Fish should extract symbols");
        assert!(
            symbols
                .iter()
                .any(|s| s.symbol_type == SymbolType::Function),
            "Fish should extract functions"
        );
    }

    #[test]
    fn zsh_extracts_functions() {
        let source = r#"
function hello() {
  echo hello
}

greet() {
  echo "Hello, $1"
}
"#;
        let symbols = parse_and_extract(source, Language::Zsh);
        println!("Zsh symbols: {:#?}", symbols);
        assert!(!symbols.is_empty(), "Zsh should extract symbols");
        assert!(
            symbols
                .iter()
                .any(|s| s.symbol_type == SymbolType::Function),
            "Zsh should extract functions"
        );
    }

    #[test]
    fn luau_extracts_functions() {
        let source = r#"
local function hello()
  print("hello")
end

function greet(name: string)
  print("Hello, " .. name)
end
"#;
        let symbols = parse_and_extract(source, Language::Luau);
        println!("Luau symbols: {:#?}", symbols);
        assert!(!symbols.is_empty(), "Luau should extract symbols");
        assert!(
            symbols
                .iter()
                .any(|s| s.symbol_type == SymbolType::Function),
            "Luau should extract functions"
        );
    }

    #[test]
    fn scheme_extracts_definitions() {
        let source = r#"
(define (hello)
  (display "hello"))

(define (add a b)
  (+ a b))
"#;
        let symbols = parse_and_extract(source, Language::Scheme);
        println!("Scheme symbols: {:#?}", symbols);
        assert!(
            symbols.len() >= 2,
            "Scheme should extract at least 2 symbols, got {}",
            symbols.len()
        );
        assert!(
            symbols.iter().any(
                |s| s.name.as_deref() == Some("hello") && s.symbol_type == SymbolType::Function
            ),
            "Scheme should extract function 'hello'"
        );
        assert!(
            symbols
                .iter()
                .any(|s| s.name.as_deref() == Some("add") && s.symbol_type == SymbolType::Function),
            "Scheme should extract function 'add'"
        );
    }

    #[test]
    fn scheme_extracts_variable_define() {
        let source = "(define x 42)\n";
        let symbols = parse_and_extract(source, Language::Scheme);
        println!("Scheme variable symbols: {:#?}", symbols);
        assert!(
            symbols.iter().any(|s| s.name.as_deref() == Some("x")),
            "Scheme should extract variable 'x'"
        );
    }

    #[test]
    fn scheme_extracts_define_syntax() {
        let source = "(define-syntax my-macro\n  (syntax-rules () ()))\n";
        let symbols = parse_and_extract(source, Language::Scheme);
        println!("Scheme define-syntax symbols: {:#?}", symbols);
        assert!(
            symbols
                .iter()
                .any(|s| s.name.as_deref() == Some("my-macro")),
            "Scheme should extract define-syntax 'my-macro'"
        );
    }

    #[test]
    fn racket_extracts_definitions() {
        let source = r#"
#lang racket

(define (hello)
  (displayln "hello"))

(define (add a b)
  (+ a b))
"#;
        let symbols = parse_and_extract(source, Language::Racket);
        println!("Racket symbols: {:#?}", symbols);
        assert!(
            symbols.len() >= 2,
            "Racket should extract at least 2 symbols, got {}",
            symbols.len()
        );
        assert!(
            symbols.iter().any(
                |s| s.name.as_deref() == Some("hello") && s.symbol_type == SymbolType::Function
            ),
            "Racket should extract function 'hello'"
        );
        assert!(
            symbols
                .iter()
                .any(|s| s.name.as_deref() == Some("add") && s.symbol_type == SymbolType::Function),
            "Racket should extract function 'add'"
        );
    }

    #[test]
    fn racket_extracts_struct() {
        let source = "#lang racket\n(struct point (x y))\n";
        let symbols = parse_and_extract(source, Language::Racket);
        println!("Racket struct symbols: {:#?}", symbols);
        assert!(
            symbols
                .iter()
                .any(|s| s.name.as_deref() == Some("point") && s.symbol_type == SymbolType::Struct),
            "Racket should extract struct 'point'"
        );
    }

    #[test]
    fn elm_extracts_functions_and_types() {
        let source = r#"
module Main exposing (main)

type alias Model =
    { count : Int
    }

type Msg
    = Increment
    | Decrement

main =
    text "hello"
"#;
        let symbols = parse_and_extract(source, Language::Elm);
        println!("Elm symbols: {:#?}", symbols);
        assert!(!symbols.is_empty(), "Elm should extract symbols");
    }

    #[test]
    fn glsl_extracts_functions() {
        let source = r#"
struct Light {
    vec3 position;
    vec3 color;
};

void main() {
    gl_FragColor = vec4(1.0);
}
"#;
        let symbols = parse_and_extract(source, Language::Glsl);
        println!("GLSL symbols: {:#?}", symbols);
        assert!(!symbols.is_empty(), "GLSL should extract symbols");
        assert!(
            symbols
                .iter()
                .any(|s| s.symbol_type == SymbolType::Function),
            "GLSL should extract functions"
        );
    }

    #[test]
    fn hlsl_extracts_functions() {
        let source = r#"
struct VS_OUTPUT {
    float4 pos : SV_Position;
    float2 uv : TEXCOORD0;
};

float4 main(float4 pos : SV_Position) : SV_Target {
    return float4(1, 0, 0, 1);
}
"#;
        let symbols = parse_and_extract(source, Language::Hlsl);
        println!("HLSL symbols: {:#?}", symbols);
        assert!(!symbols.is_empty(), "HLSL should extract symbols");
        assert!(
            symbols
                .iter()
                .any(|s| s.symbol_type == SymbolType::Function),
            "HLSL should extract functions"
        );
    }

    // ── Tier 2B symbol extraction tests ─────────────────

    #[test]
    fn svelte_extracts_blocks() {
        let source = r#"
<script>
  let count = 0;
  function increment() { count += 1; }
</script>

<button on:click={increment}>
  Count: {count}
</button>

<style>
  button { color: red; }
</style>
"#;
        let symbols = parse_and_extract(source, Language::Svelte);
        println!("Svelte symbols: {:#?}", symbols);
        assert!(!symbols.is_empty(), "Svelte should extract symbols");
        assert!(
            symbols.iter().any(|s| s.symbol_type == SymbolType::Block),
            "Svelte should extract blocks"
        );
    }

    #[test]
    fn astro_extracts_frontmatter_and_elements() {
        let source = r#"---
const title = "Hello";
const items = [1, 2, 3];
---
<html>
<head><title>{title}</title></head>
<body>
  <h1>{title}</h1>
</body>
</html>
"#;
        let symbols = parse_and_extract(source, Language::Astro);
        println!("Astro symbols: {:#?}", symbols);
        assert!(!symbols.is_empty(), "Astro should extract symbols");
    }

    #[test]
    fn makefile_extracts_rules_and_variables() {
        let source = r#"
CC = gcc
CFLAGS = -Wall

all: build

build:
	$(CC) $(CFLAGS) -o main main.c

clean:
	rm -f main
"#;
        let symbols = parse_and_extract(source, Language::Makefile);
        println!("Makefile symbols: {:#?}", symbols);
        assert!(!symbols.is_empty(), "Makefile should extract symbols");
        assert!(
            symbols
                .iter()
                .any(|s| s.symbol_type == SymbolType::Function),
            "Makefile should extract rules as functions"
        );
    }

    #[test]
    fn ini_extracts_sections() {
        let source = r#"
[database]
host = localhost
port = 5432

[server]
bind = 0.0.0.0
"#;
        let symbols = parse_and_extract(source, Language::Ini);
        println!("INI symbols: {:#?}", symbols);
        assert!(!symbols.is_empty(), "INI should extract symbols");
        assert!(
            symbols.iter().any(|s| s.symbol_type == SymbolType::Block),
            "INI should extract sections as blocks"
        );
    }

    #[test]
    fn nginx_extracts_blocks() {
        let source = r#"
server {
    listen 80;
    server_name example.com;

    location / {
        proxy_pass http://backend;
    }
}
"#;
        let symbols = parse_and_extract(source, Language::Nginx);
        println!("Nginx symbols: {:#?}", symbols);
        assert!(!symbols.is_empty(), "Nginx should extract symbols");
    }

    #[test]
    fn prisma_extracts_models_and_enums() {
        let source = r#"
generator client {
  provider = "prisma-client-js"
}

datasource db {
  provider = "postgresql"
  url      = env("DATABASE_URL")
}

model User {
  id    Int     @id @default(autoincrement())
  email String  @unique
  name  String?
  posts Post[]
}

model Post {
  id       Int    @id @default(autoincrement())
  title    String
  author   User   @relation(fields: [authorId], references: [id])
  authorId Int
}

enum Role {
  USER
  ADMIN
}
"#;
        let symbols = parse_and_extract(source, Language::Prisma);
        println!("Prisma symbols: {:#?}", symbols);
        assert!(!symbols.is_empty(), "Prisma should extract symbols");
        assert!(
            symbols.iter().any(|s| s.symbol_type == SymbolType::Struct),
            "Prisma should extract models as structs"
        );
    }
}
