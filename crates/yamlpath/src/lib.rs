//! Comment and format-preserving YAML path routes.
//!
//! This is **not** "XPath but for YAML". If you need a generic object
//! route language that **doesn't** capture exact parse spans or comments,
//! then you probably want an implementation of [JSONPath] or something
//! like [jq].
//!
//! [JSONPath]: https://en.wikipedia.org/wiki/JSONPath
//! [jq]: https://jqlang.github.io/jq/

#![deny(rustdoc::broken_intra_doc_links)]
#![deny(missing_docs)]
#![allow(clippy::redundant_field_names)]
#![forbid(unsafe_code)]

use std::{
    borrow::Cow,
    collections::{BTreeMap, HashMap},
    ops::{Deref, RangeBounds},
};

use line_index::LineIndex;
use serde::Serialize;
use thiserror::Error;
use tree_sitter::{Language, Node, Parser};
use tree_sitter_iter::TreeIter;

/// Possible errors when performing YAML path routes.
#[derive(Error, Debug)]
pub enum QueryError {
    /// The tree-sitter backend couldn't accept the YAML grammar.
    #[error("malformed or unsupported tree-sitter grammar")]
    InvalidLanguage(#[from] tree_sitter::LanguageError),
    /// The user's input YAML is malformed.
    #[error("input is not valid YAML")]
    InvalidInput,
    /// The route expects a key at a given point, but the input isn't a mapping.
    #[error("expected mapping containing key `{0}`")]
    ExpectedMapping(String),
    /// The route expects a list index at a given point, but the input isn't a list.
    #[error("expected list for index `[{0}]`")]
    ExpectedList(usize),
    /// The route expects the given key in a mapping, but the mapping doesn't have that key.
    #[error("mapping has no key `{0}`")]
    ExhaustedMapping(String),
    /// The route expects the given list index, but the list isn't the right size.
    #[error("index `[{0}]` exceeds list size ({1})")]
    ExhaustedList(usize, usize),
    /// The YAML syntax tree wasn't structured the way we expect.
    #[error("unexpected node: `{0}`")]
    UnexpectedNode(String),
    /// The YAML syntax tree is missing an expected named child node.
    #[error("syntax node `{0}` is missing named child `{1}`")]
    MissingChild(String, String),
    /// The YAML syntax tree is missing an expected named child node with
    /// the given field name.
    #[error("syntax node `{0}` is missing child field `{1}`")]
    MissingChildField(String, &'static str),
    /// Any other route error that doesn't fit cleanly above.
    #[error("route error: {0}")]
    Other(String),
}

/// A route into some YAML document.
///
/// Internally, a route is zero or more "component" selectors, each of which
/// is either a mapping key or list index to descend through. An empty
/// route corresponds to the top-most document feature.
///
/// For example, with the following YAML document:
///
/// ```yaml
/// foo:
///   bar:
///     baz:
///       - [a, b, c]
///       - [d, e, f]
/// ```
///
/// The sub-list member `e` would be identified via the path
/// `foo`, `bar`, `baz`, `1`, `1`.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Route<'a> {
    /// The individual top-down components of this route.
    route: Vec<Component<'a>>,
}

impl<'a> Route<'a> {
    /// Returns whether this route is empty.
    pub fn is_empty(&self) -> bool {
        self.route.is_empty()
    }

    /// Create a new route from this route, with the given component
    /// added to the end.
    pub fn with_key(&self, component: impl Into<Component<'a>>) -> Self {
        let mut components = self.route.clone();
        components.push(component.into());

        Self::from(components)
    }

    /// Create a new route from this route, with the given components
    /// added to the end.
    pub fn with_keys(&self, components: impl IntoIterator<Item = Component<'a>>) -> Self {
        let mut new_route = self.route.clone();
        new_route.extend(components);

        Self::from(new_route)
    }

    /// Returns a route for the "parent" path of the route's current path,
    /// or `None` the current route has no parent.
    pub fn parent(&self) -> Option<Self> {
        if self.is_empty() {
            None
        } else {
            let mut route = self.route.clone();
            route.truncate(self.route.len() - 1);
            Some(Self::from(route))
        }
    }
}

/// Convenience builder for constructing a `Route`.
#[macro_export]
macro_rules! route {
    ($($key:expr),* $(,)?) => {
        $crate::Route::from(
            vec![$($crate::Component::from($key)),*]
        )
    };
    () => {
        $crate::Route::default()
    };
}

impl<'a> From<Vec<Component<'a>>> for Route<'a> {
    fn from(route: Vec<Component<'a>>) -> Self {
        Self { route }
    }
}

/// A single `Route` component.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub enum Component<'a> {
    /// A YAML key.
    Key(Cow<'a, str>),

    /// An index into a YAML array.
    Index(usize),
}

impl From<usize> for Component<'_> {
    fn from(index: usize) -> Self {
        Component::Index(index)
    }
}

impl<'a> From<&'a str> for Component<'a> {
    fn from(key: &'a str) -> Self {
        Component::Key(key.into())
    }
}

impl From<String> for Component<'_> {
    fn from(key: String) -> Self {
        Component::Key(key.into())
    }
}

/// Represents the concrete location of some YAML syntax.
#[derive(Debug)]
pub struct Location {
    /// The byte span at which the route's result appears.
    pub byte_span: (usize, usize),
    /// The "point" (i.e., line/column) span at which the route's result appears.
    pub point_span: ((usize, usize), (usize, usize)),
}

impl From<Node<'_>> for Location {
    fn from(node: Node<'_>) -> Self {
        let start_point = node.start_position();
        let end_point = node.end_position();

        Self {
            byte_span: (node.start_byte(), node.end_byte()),
            point_span: (
                (start_point.row, start_point.column),
                (end_point.row, end_point.column),
            ),
        }
    }
}

/// Unique identifier for a CST node. Provides O(1) parent access without lifetime issues.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct NodeId(usize);

/// Storage for all CST nodes. Single source of truth for the entire tree.
#[derive(Clone)]
struct CstStore {
    /// All nodes stored in a contiguous Vec for cache-friendly access.
    nodes: Vec<YamlNodeData>,
    /// The root node of the document.
    root_id: NodeId,
}

/// A node in the CST with parent tracking via NodeId.
#[derive(Clone)]
struct YamlNodeData {
    /// Unique identifier for this node.
    id: NodeId,
    /// Parent node ID (None only for root).
    parent_id: Option<NodeId>,
    /// The node's type and data.
    kind: YamlNodeKind,
}

#[derive(Clone)]
enum YamlNodeKind {
    Mapping(MappingData),
    Sequence(SequenceData),
    Scalar(ScalarData),
}

/// A mapping node (object/dict) in the reduced CST.
#[derive(Clone)]
struct MappingData {
    /// Map from key string to entry (key location + optional value node).
    /// Uses Cow for zero-copy when keys don't need escaping.
    entries: HashMap<Cow<'static, str>, MappingEntry>,

    /// The exact span of this mapping in the source.
    exact_span: SpanInfo,
}

#[derive(Clone)]
struct MappingEntry {
    /// Span of the key itself.
    key_span: SpanInfo,

    /// The value node ID (None for absent values like `foo:` or `{ foo }`).
    value_id: Option<NodeId>,

    /// Span of the entire mapping pair (key: value).
    pair_span: SpanInfo,
}

/// A sequence node (array/list) in the reduced CST.
#[derive(Clone)]
struct SequenceData {
    /// The item node IDs in this sequence.
    item_ids: Vec<NodeId>,

    /// The exact span of this sequence in the source.
    exact_span: SpanInfo,
}

/// A scalar node (leaf value) in the reduced CST.
#[derive(Clone)]
struct ScalarData {
    /// The exact span of this scalar in the source.
    exact_span: SpanInfo,

    /// The span including any block scalar indicator (|, >).
    pretty_span: Option<SpanInfo>,

    /// The scalar kind (plain, single-quote, double-quote, or block).
    /// Needed to properly extract the value (e.g., unquoting).
    kind: ScalarKind,
}

#[derive(Clone, Debug)]
enum ScalarKind {
    Plain,         // foo, 123, true
    SingleQuote,   // 'foo bar'
    DoubleQuote,   // "foo\nbar"
    Block,         // | or > multiline
}

/// Span information: byte range + line/col range.
#[derive(Clone, Debug)]
struct SpanInfo {
    byte_range: std::ops::Range<usize>,
    point_range: std::ops::Range<tree_sitter::Point>,
}

/// Describes the feature's kind, i.e. whether it's a mapping, sequence,
/// or scalar value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FeatureKind {
    /// A mapping (object/dict), either block or flow style.
    Mapping,
    /// A sequence (array/list), either block or flow style.
    Sequence,
    /// Any sort of scalar value.
    Scalar,
}

/// Represents the result of a successful route.
#[derive(Debug)]
pub struct Feature<'tree> {
    /// The tree-sitter node that this feature was extracted from.
    _node: Node<'tree>,

    /// The exact location of the route result.
    pub location: Location,

    /// The "context" location for the route result.
    /// This is typically the surrounding mapping or list structure.
    pub context: Option<Location>,
}

impl Feature<'_> {
    /// Return this feature's parent feature, if it has one.
    pub fn parent(&self) -> Option<Feature<'_>> {
        self._node.parent().map(Feature::from)
    }

    /// Return this feature's [`FeatureKind`].
    pub fn kind(&self) -> FeatureKind {
        // TODO: Use node kind IDs instead of string matching.

        // Our feature's underlying node is often a
        // `block_node` or `flow_node`, which is a container
        // for the real kind of node we're interested in.
        let node = match self._node.kind() {
            "block_node" | "flow_node" => self
                ._node
                .child(0)
                .expect("internal error: expected child of block_node/flow_node"),
            _ => self._node,
        };

        match node.kind() {
            "block_mapping" | "flow_mapping" => FeatureKind::Mapping,
            "block_sequence" | "flow_sequence" => FeatureKind::Sequence,
            "plain_scalar" | "single_quote_scalar" | "double_quote_scalar" | "block_scalar" => {
                FeatureKind::Scalar
            }
            kind => unreachable!("unexpected feature kind: {kind}"),
        }
    }

    /// Returns whether this feature spans multiple lines.
    pub fn is_multiline(&self) -> bool {
        self.location.point_span.0.0 != self.location.point_span.1.0
    }
}

impl RangeBounds<usize> for &Feature<'_> {
    fn start_bound(&self) -> std::ops::Bound<&usize> {
        std::ops::Bound::Included(&self.location.byte_span.0)
    }

    fn end_bound(&self) -> std::ops::Bound<&usize> {
        std::ops::Bound::Excluded(&self.location.byte_span.1)
    }
}

impl<'tree> From<Node<'tree>> for Feature<'tree> {
    fn from(node: Node<'tree>) -> Self {
        Feature {
            _node: node,
            location: Location::from(node),
            context: node.parent().map(Location::from),
        }
    }
}

/// Configures how features are extracted from a YAML document
/// during queries.
#[derive(Copy, Clone, Debug)]
enum QueryMode {
    /// Make extracted features as "pretty" as possible, e.g. by
    /// including components that humans subjectively consider relevant.
    ///
    /// For example, querying `foo: bar` for `foo` will return
    /// `foo: bar` instead of just `bar`.
    Pretty,
    /// For routes that terminate in a key, make the extracted
    /// feature only that key, rather than both the key and value ("pretty"),
    /// or just the value ("exact").
    ///
    /// For example, querying `foo: bar` for `foo` will return `foo`.
    KeyOnly,
    /// Make extracted features as "exact" as possible, e.g. by
    /// including only the exact span of the route result.
    Exact,
}

/// A holder type so that we can associate both source and node references
/// with the same lifetime for [`self_cell`].
#[derive(Clone)]
struct SourceTree {
    source: String,
    tree: tree_sitter::Tree,
}

impl Deref for SourceTree {
    type Target = tree_sitter::Tree;

    fn deref(&self) -> &Self::Target {
        &self.tree
    }
}

type AnchorMap<'tree> = HashMap<&'tree str, BTreeMap<usize, Node<'tree>>>;

self_cell::self_cell!(
    /// A wrapper for a [`SourceTree`] that also contains a computed
    /// anchor map.
    struct Tree {
        owner: SourceTree,

        #[covariant]
        dependent: AnchorMap,
    }
);

impl Tree {
    fn build(inner: SourceTree) -> Result<Self, QueryError> {
        Tree::try_new(SourceTree::clone(&inner), |tree| {
            let mut anchor_map: AnchorMap = HashMap::new();

            for anchor in TreeIter::new(tree).filter(|n| n.kind() == "anchor") {
                // NOTE(ww): We could poke into the `anchor_name` child
                // instead of slicing, but this is simpler.
                let anchor_name = &anchor
                    .utf8_text(tree.source.as_bytes())
                    .expect("impossible: anchor name should be UTF-8 by construction")[1..];

                // NOTE(ww): We insert the anchor's next non-comment
                // sibling as the anchor's target. This makes things
                // a bit simpler when descending later, plus it produces
                // more useful spans, since neither the anchor node
                // nor its parent are useful in the aliased context.
                let parent = anchor.parent().ok_or_else(|| {
                    QueryError::UnexpectedNode("anchor node has no parent".into())
                })?;

                let mut cursor = parent.walk();
                let sibling = parent
                    .named_children(&mut cursor)
                    .find(|child| child.kind() != "anchor" && child.kind() != "comment")
                    .ok_or_else(|| {
                        QueryError::UnexpectedNode("anchor has no non-comment sibling".into())
                    })?;

                // Store anchor with its position; duplicates are allowed and
                // resolved by position when aliases are encountered.
                anchor_map
                    .entry(anchor_name)
                    .or_default()
                    .insert(anchor.start_byte(), sibling);
            }

            Ok(anchor_map)
        })
    }
}

impl Clone for Tree {
    fn clone(&self) -> Self {
        // Cloning is mildly annoying: we can clone the tree itself,
        // but we need to reconstruct the anchor map from scratch since
        // it borrows from the tree.
        // TODO: Can we do better here?
        // Unwrap safety: we're cloning from an existing valid owner.
        Self::build(self.borrow_owner().clone())
            .expect("impossible: cloning a Tree preserves invariants")
    }
}

impl Deref for Tree {
    type Target = tree_sitter::Tree;

    fn deref(&self) -> &Self::Target {
        &self.borrow_owner().tree
    }
}

/// Builder for constructing the CST with automatic parent tracking.
struct CstBuilder {
    nodes: Vec<YamlNodeData>,
    next_id: usize,
}

impl CstBuilder {
    fn new() -> Self {
        Self {
            nodes: Vec::new(),
            next_id: 0,
        }
    }

    fn alloc_id(&mut self) -> NodeId {
        let id = NodeId(self.next_id);
        self.next_id += 1;
        id
    }

    fn add_node(&mut self, id: NodeId, parent_id: Option<NodeId>, kind: YamlNodeKind) {
        self.nodes.push(YamlNodeData {
            id,
            parent_id,
            kind,
        });
    }

    fn finish(self, root_id: NodeId) -> CstStore {
        CstStore {
            nodes: self.nodes,
            root_id,
        }
    }
}

/// Build the reduced CST from a tree-sitter node.
fn build_node(
    builder: &mut CstBuilder,
    node: Node,
    parent_id: Option<NodeId>,
    source: &str,
    anchor_map: &AnchorMap,
) -> Result<NodeId, QueryError> {
    // Skip wrapper nodes (document, block_node, flow_node)
    let mut current = node;
    while matches!(current.kind(), "document" | "block_node" | "flow_node") {
        if let Some(child) = current.named_child(0) {
            current = child;
        } else {
            break;
        }
    }

    // Handle anchors: skip to the anchored content
    if current.kind() == "anchor" {
        if let Some(content) = current.next_named_sibling() {
            current = content;
        }
    }

    // Handle aliases: resolve to the anchored content
    if current.kind() == "alias" {
        let alias_name = current
            .child_by_field_name("name")
            .and_then(|n| n.utf8_text(source.as_bytes()).ok())
            .ok_or_else(|| QueryError::Other("alias missing name".into()))?;

        // Find the most recent anchor with this name
        let target_node = anchor_map
            .get(alias_name)
            .and_then(|positions| positions.range(..current.start_byte()).next_back())
            .map(|(_, node)| *node)
            .ok_or_else(|| QueryError::Other(format!("undefined alias: {}", alias_name)))?;

        // Recursively build the aliased node with the ALIAS's parent
        // Note: This creates a new subtree for each alias (no structural sharing)
        return build_node(builder, target_node, parent_id, source, anchor_map);
    }

    // Allocate ID for this node
    let node_id = builder.alloc_id();

    // Build node kind based on type
    let kind = match current.kind() {
        "block_mapping" | "flow_mapping" => {
            build_mapping_node(builder, current, node_id, source, anchor_map)?
        }
        "block_sequence" | "flow_sequence" => {
            build_sequence_node(builder, current, node_id, source, anchor_map)?
        }
        _ => {
            // Scalar node
            YamlNodeKind::Scalar(ScalarData {
                exact_span: span_from_node(&current),
                pretty_span: compute_pretty_span(&current),
                kind: infer_scalar_kind(&current),
            })
        }
    };

    // Store node with parent link
    builder.add_node(node_id, parent_id, kind);

    Ok(node_id)
}

/// Build a mapping node from a tree-sitter block_mapping or flow_mapping.
fn build_mapping_node(
    builder: &mut CstBuilder,
    node: Node,
    parent_id: NodeId,  // This mapping's ID becomes parent for children
    source: &str,
    anchor_map: &AnchorMap,
) -> Result<YamlNodeKind, QueryError> {
    let mut entries = HashMap::new();

    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        // Extract key from flow_pair, block_mapping_pair, or bare flow_node
        let (key_node, value_node_opt, pair_span) = match child.kind() {
            "flow_pair" | "block_mapping_pair" => {
                let key = child
                    .child_by_field_name("key")
                    .ok_or_else(|| QueryError::MissingChildField(child.kind().into(), "key"))?;
                let value = child.child_by_field_name("value");
                (key, value, span_from_node(&child))
            }
            "flow_node" => {
                // Bare key in flow mapping like { foo }
                (child, None, span_from_node(&child))
            }
            _ => continue,
        };

        // Extract key string (handle quotes, anchors)
        let key_str = extract_key_string(&key_node, source)?;

        // Build value node with THIS node as parent
        let value_id = if let Some(val_node) = value_node_opt {
            Some(build_node(builder, val_node, Some(parent_id), source, anchor_map)?)
        } else {
            None
        };

        entries.insert(
            key_str,
            MappingEntry {
                key_span: span_from_node(&key_node),
                value_id,
                pair_span,
            },
        );
    }

    Ok(YamlNodeKind::Mapping(MappingData {
        entries,
        exact_span: span_from_node(&node),
    }))
}

/// Build a sequence node from a tree-sitter block_sequence or flow_sequence.
fn build_sequence_node(
    builder: &mut CstBuilder,
    node: Node,
    parent_id: NodeId,
    source: &str,
    anchor_map: &AnchorMap,
) -> Result<YamlNodeKind, QueryError> {
    let mut item_ids = Vec::new();

    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        // Handle block_sequence_item wrappers
        let content = if child.kind() == "block_sequence_item" {
            child.named_child(0).unwrap_or(child)
        } else {
            child
        };

        let item_id = build_node(builder, content, Some(parent_id), source, anchor_map)?;
        item_ids.push(item_id);
    }

    Ok(YamlNodeKind::Sequence(SequenceData {
        item_ids,
        exact_span: span_from_node(&node),
    }))
}

/// Extract a key string from a key node, handling quotes and anchors.
fn extract_key_string(node: &Node, source: &str) -> Result<Cow<'static, str>, QueryError> {
    // Skip anchor nodes to get to the actual key
    let mut cursor = node.walk();
    let scalar = node
        .named_children(&mut cursor)
        .find(|n| n.kind() != "anchor")
        .unwrap_or(*node);

    let text = scalar
        .utf8_text(source.as_bytes())
        .map_err(|_| QueryError::Other("invalid UTF-8 in key".into()))?;

    // Unquote if necessary
    let unquoted = match scalar.kind() {
        "single_quote_scalar" | "double_quote_scalar" => {
            let mut chars = text.chars();
            chars.next(); // Skip opening quote
            chars.next_back(); // Skip closing quote
            Cow::Owned(chars.as_str().to_string())
        }
        _ => Cow::Owned(text.to_string()), // TODO: Use Cow::Borrowed when possible
    };

    Ok(unquoted)
}

/// Create a SpanInfo from a tree-sitter node.
fn span_from_node(node: &Node) -> SpanInfo {
    SpanInfo {
        byte_range: node.start_byte()..node.end_byte(),
        point_range: node.start_position()..node.end_position(),
    }
}

/// Compute the pretty span for a scalar (includes block scalar indicators).
fn compute_pretty_span(node: &Node) -> Option<SpanInfo> {
    // If this is a block scalar, include the indicator (|, >, |-, etc.)
    if node.kind() == "block_scalar" {
        Some(span_from_node(node))
    } else {
        None
    }
}

/// Infer the scalar kind from a tree-sitter node.
fn infer_scalar_kind(node: &Node) -> ScalarKind {
    match node.kind() {
        "single_quote_scalar" => ScalarKind::SingleQuote,
        "double_quote_scalar" => ScalarKind::DoubleQuote,
        "block_scalar" => ScalarKind::Block,
        _ => ScalarKind::Plain,
    }
}

/// Represents a queryable YAML document.
#[derive(Clone)]
pub struct Document {
    /// The underlying tree-sitter parse tree and source.
    /// Kept for comment extraction and span computation.
    tree: Tree,

    /// The reduced CST storage (all nodes in a Vec).
    cst: CstStore,

    /// Line/column index for the source.
    line_index: LineIndex,

    // NOTE: These field IDs are kept temporarily for backwards compatibility
    // with the old query_node implementation. They will be removed once
    // all query methods are migrated to use the CST.
    document_id: u16,
    block_node_id: u16,
    flow_node_id: u16,
    block_sequence_id: u16,
    flow_sequence_id: u16,
    block_mapping_id: u16,
    flow_mapping_id: u16,
    block_mapping_pair_id: u16,
    flow_pair_id: u16,
    block_sequence_item_id: u16,
    comment_id: u16,
    anchor_id: u16,
    alias_id: u16,
    block_scalar_id: u16,
}

impl Document {
    /// Construct a new `Document` from the given YAML.
    pub fn new(source: impl Into<String>) -> Result<Self, QueryError> {
        let source = source.into();

        let mut parser = Parser::new();
        let language: Language = tree_sitter_yaml::LANGUAGE.into();
        parser.set_language(&language)?;

        // NOTE: Infallible, assuming `language` is correctly constructed above.
        let tree = parser
            .parse(&source, None)
            .expect("impossible: tree-sitter parsing should never fail");

        if tree.root_node().has_error() {
            return Err(QueryError::InvalidInput);
        }

        let line_index = LineIndex::new(&source);

        let source_tree = SourceTree {
            source: source,
            tree,
        };

        let tree_struct = Tree::build(source_tree)?;

        // Build the reduced CST with NodeId-based parent tracking
        let mut builder = CstBuilder::new();
        let root_node = tree_struct.borrow_owner().tree.root_node();
        let root_id = build_node(
            &mut builder,
            root_node,
            None,  // No parent for root
            tree_struct.borrow_owner().source.as_str(),
            tree_struct.borrow_dependent(),
        )?;
        let cst = builder.finish(root_id);

        Ok(Self {
            tree: tree_struct,
            cst,
            line_index,
            document_id: language.id_for_node_kind("document", true),
            block_node_id: language.id_for_node_kind("block_node", true),
            flow_node_id: language.id_for_node_kind("flow_node", true),
            block_sequence_id: language.id_for_node_kind("block_sequence", true),
            flow_sequence_id: language.id_for_node_kind("flow_sequence", true),
            block_mapping_id: language.id_for_node_kind("block_mapping", true),
            flow_mapping_id: language.id_for_node_kind("flow_mapping", true),
            block_mapping_pair_id: language.id_for_node_kind("block_mapping_pair", true),
            flow_pair_id: language.id_for_node_kind("flow_pair", true),
            block_sequence_item_id: language.id_for_node_kind("block_sequence_item", true),
            comment_id: language.id_for_node_kind("comment", true),
            anchor_id: language.id_for_node_kind("anchor", true),
            alias_id: language.id_for_node_kind("alias", true),
            block_scalar_id: language.id_for_node_kind("block_scalar", true),
        })
    }

    /// Returns a [`LineIndex`] for this document, which can be used
    /// to efficiently map between byte offsets and line coordinates.
    pub fn line_index(&self) -> &LineIndex {
        &self.line_index
    }

    /// Return a view of the original YAML source that this document was
    /// loaded from.
    pub fn source(&self) -> &str {
        &self.tree.borrow_owner().source
    }

    /// Resolve an anchor by name, returning the target node that was active
    /// at the given position. For duplicate anchors, this returns the most
    /// recent definition that appears before `position`.
    fn resolve_anchor(&self, name: &str, position: usize) -> Option<Node<'_>> {
        self.tree
            .borrow_dependent()
            .get(name)?
            .range(..position)
            .next_back()
            .map(|(_, node)| *node)
    }

    /// Returns a [`Feature`] for the topmost semantic object in this document.
    ///
    /// This is typically useful as a "fallback" feature, e.g. for positioning
    /// relative to the "top" of the document.
    pub fn top_feature(&self) -> Result<Feature<'_>, QueryError> {
        let top_node = self.top_object()?;
        Ok(top_node.into())
    }

    /// Returns whether the given range is spanned by a comment node.
    ///
    /// The comment node must fully span the range; a range that ends
    /// after the comment or starts before it will not be considered
    /// spanned.
    pub fn range_spanned_by_comment(&self, start: usize, end: usize) -> bool {
        let root = self.tree.root_node();

        match root.named_descendant_for_byte_range(start, end) {
            Some(child) => child.kind_id() == self.comment_id,
            None => false,
        }
    }

    /// Returns whether the given offset is within a comment node's span.
    pub fn offset_inside_comment(&self, offset: usize) -> bool {
        self.range_spanned_by_comment(offset, offset)
    }

    /// Perform a route on the current document, returning `true`
    /// if the route succeeds (i.e. references an existing feature).
    ///
    /// All errors become `false`.
    pub fn query_exists(&self, route: &Route) -> bool {
        self.query_node(route, QueryMode::Exact).is_ok()
    }

    /// Perform a route on the current document, returning a `Feature`
    /// if the route succeeds.
    ///
    /// The feature is extracted in "pretty" mode, meaning that it'll
    /// contain a subjectively relevant "pretty" span rather than the
    /// exact span of the route result.
    ///
    /// For example, querying `foo: bar` for `foo` will return
    /// `foo: bar` instead of just `bar`.
    pub fn query_pretty(&self, route: &Route) -> Result<Feature<'_>, QueryError> {
        self.query_node(route, QueryMode::Pretty).map(|n| n.into())
    }

    /// Perform a route on the current document, returning a `Feature`
    /// if the route succeeds. Returns `None` if the route
    /// succeeds, but matches an absent value (e.g. `foo:`).
    ///
    /// The feature is extracted in "exact" mode, meaning that it'll
    /// contain the exact span of the route result.
    ///
    /// For example, querying `foo: bar` for `foo` will return
    /// just `bar` instead of `foo: bar`.
    pub fn query_exact(&self, route: &Route) -> Result<Option<Feature<'_>>, QueryError> {
        let node = self.query_node(route, QueryMode::Exact)?;

        if node.kind_id() == self.block_mapping_pair_id || node.kind_id() == self.flow_pair_id {
            // If the route matches a mapping pair, we return None,
            // since this indicates an absent value.
            Ok(None)
        } else {
            // Otherwise, we return the node as a feature.
            Ok(Some(node.into()))
        }
    }

    /// Perform a route on the current document, returning a `Feature`
    /// if the route succeeds.
    ///
    /// The feature is extracted in "key only" mode, meaning that it'll
    /// contain only the key of a mapping, rather than the
    /// key and value ("pretty") or just the value ("exact").
    ///
    /// For example, querying `foo: bar` for `foo` will return
    /// just `foo` instead of `foo: bar` or `bar`.
    pub fn query_key_only(&self, route: &Route) -> Result<Feature<'_>, QueryError> {
        if !matches!(route.route.last(), Some(Component::Key(_))) {
            return Err(QueryError::Other(
                "route must end with a key component for key-only routes".into(),
            ));
        }

        self.query_node(route, QueryMode::KeyOnly).map(|n| n.into())
    }

    /// Returns a string slice of the original document corresponding to
    /// the given [`Feature`].
    ///
    /// This function returns a slice corresponding to the [`Feature`]'s exact
    /// span, meaning that leading whitespace for the start point is not
    /// necessarily captured. See [`Self::extract_with_leading_whitespace`]
    /// for feature extraction with rudimentary whitespace handling.
    ///
    /// Panics if the feature's span is invalid.
    pub fn extract(&self, feature: &Feature) -> &str {
        &self.source()[feature.location.byte_span.0..feature.location.byte_span.1]
    }

    /// Returns a string slice of the original document corresponding to the given
    /// [`Feature`], along with any leading (indentation-semantic) whitespace.
    ///
    /// **Important**: The returned string here can be longer than the span
    /// identified in the [`Feature`]. In particular, this API will return a
    /// longer string if it identifies leading non-newline whitespace
    /// ahead of the captured [`Feature`], since this indicates indentation
    /// not encapsulated by the feature itself.
    ///
    /// Panics if the feature's span is invalid.
    pub fn extract_with_leading_whitespace<'a>(&'a self, feature: &Feature) -> &'a str {
        let mut start_idx = feature.location.byte_span.0;
        let pre_slice = &self.source()[0..start_idx];
        if let Some(last_newline) = pre_slice.rfind('\n') {
            // If everything between the last newline and the start_index
            // is ASCII spaces, then we include it.
            if self.source()[last_newline + 1..start_idx]
                .bytes()
                .all(|b| b == b' ')
            {
                start_idx = last_newline + 1
            }
        }

        &self.source()[start_idx..feature.location.byte_span.1]
    }

    /// Given a [`Feature`], return all comments that span the same range
    /// as the feature does.
    pub fn feature_comments<'tree>(&'tree self, feature: &Feature<'tree>) -> Vec<Feature<'tree>> {
        // To extract all comments for a feature, we trawl the entire tree's
        // nodes and extract all comment nodes in the line range for the
        // feature.
        // This isn't the fastest way to do things, since we end up
        // visiting a lot of (top-level) nodes that aren't in the feature's
        // range.
        // The alternative to this approach would be to find the feature's
        // spanning parent and only trawl that subset of the tree; the main
        // annoyance with doing things that way is the AST can look like this:
        //
        // top
        // |
        // |------ parent
        // |       |
        // |       |____ child
        // |
        // |______ comment
        //
        // With this AST the spanning parent is 'parent', but the 'comment'
        // node is actually *adjacent* to 'parent' rather than enclosed in it.

        let start_line = feature.location.point_span.0.0;
        let end_line = feature.location.point_span.1.0;

        fn trawl<'tree>(
            node: &Node<'tree>,
            comment_id: u16,
            start_line: usize,
            end_line: usize,
        ) -> Vec<Feature<'tree>> {
            let mut comments = vec![];
            let mut cur = node.walk();

            // If this node ends before our span or starts after it, there's
            // no point in recursing through it.
            if node.end_position().row < start_line || node.start_position().row > end_line {
                return comments;
            }

            // Find any comments among the current children.
            comments.extend(
                node.named_children(&mut cur)
                    .filter(|c| {
                        c.kind_id() == comment_id
                            && c.start_position().row >= start_line
                            && c.end_position().row <= end_line
                    })
                    .map(|c| c.into()),
            );

            for child in node.children(&mut cur) {
                comments.extend(trawl(&child, comment_id, start_line, end_line));
            }

            comments
        }

        trawl(
            &self.tree.root_node(),
            self.comment_id,
            start_line,
            end_line,
        )
    }

    /// Returns whether this document contains any YAML anchors.
    pub fn has_anchors(&self) -> bool {
        !self.tree.borrow_dependent().is_empty()
    }

    /// Returns the topmost semantic object in the YAML document,
    /// i.e. the node corresponding to the first block or flow feature.
    fn top_object(&self) -> Result<Node<'_>, QueryError> {
        // All tree-sitter-yaml trees start with a `stream` node.
        let stream = self.tree.root_node();

        // The `document` child is the "body" of the YAML document; it
        // might not be the first node in the `stream` if there are comments.
        let mut cur = stream.walk();
        let document = stream
            .named_children(&mut cur)
            .find(|c| c.kind_id() == self.document_id)
            .ok_or_else(|| QueryError::MissingChild(stream.kind().into(), "document".into()))?;

        // The document might have a directives section, which we need to
        // skip over. We do this by finding the top-level `block_node`
        // or `flow_node`, of which one will be present depending on how
        // the top-level document value is expressed.
        let top_node = document
            .named_children(&mut cur)
            .find(|c| c.kind_id() == self.block_node_id || c.kind_id() == self.flow_node_id)
            .ok_or_else(|| QueryError::Other("document has no block_node or flow_node".into()))?;

        Ok(top_node)
    }

    fn query_node(&self, route: &Route, mode: QueryMode) -> Result<Node<'_>, QueryError> {
        let mut focus_node = self.top_object()?;
        for component in &route.route {
            match self.descend(&focus_node, component) {
                Ok(next) => focus_node = next,
                Err(e) => return Err(e),
            }
        }

        // Our focus node might be an alias, in which case we need to
        // do one last leap to get our "real" final focus node.
        // TODO(ww): What about nested aliases?
        focus_node = match focus_node.child(0) {
            Some(child) if child.kind_id() == self.alias_id => {
                let alias_name = child
                    .utf8_text(self.source().as_bytes())
                    .expect("impossible: alias name should be UTF-8 by construction");
                self.resolve_anchor(&alias_name[1..], child.start_byte())
                    .ok_or_else(|| QueryError::Other(format!("unknown alias: {}", alias_name)))?
            }
            // Our focus node might have an anchor prefix (e.g. `[&x v, *x]`),
            // in which case we skip to the non-anchor sibling.
            Some(child) if child.kind_id() == self.anchor_id => {
                let mut cursor = focus_node.walk();
                focus_node
                    .named_children(&mut cursor)
                    .find(|n| n.kind_id() != self.anchor_id)
                    .unwrap_or(focus_node)
            }
            _ => focus_node,
        };

        focus_node = match mode {
            QueryMode::Pretty => {
                // If we're in "pretty" mode, we want to return the
                // block/flow pair node that contains the key.
                // This results in a (subjectively) more intuitive extracted feature,
                // since `foo: bar` gets extracted for `foo` instead of just `bar`.
                //
                // NOTE: We might already be on the block/flow pair if we terminated
                // with an absent value, in which case we don't need to do this cleanup.
                if matches!(route.route.last(), Some(Component::Key(_)))
                    && focus_node.kind_id() != self.block_mapping_pair_id
                    && focus_node.kind_id() != self.flow_pair_id
                {
                    focus_node.parent().expect("missing parent of focus node")
                } else {
                    focus_node
                }
            }
            QueryMode::KeyOnly => {
                // If we're in "key only" mode, we need to walk back up to
                // the parent block/flow pair node that contains the key,
                // and isolate on the key child instead.

                let parent_node = if focus_node.kind_id() == self.block_mapping_pair_id
                    || focus_node.kind_id() == self.flow_pair_id
                {
                    // If we're already on block/flow pair, then we're already
                    // the key's parent.
                    focus_node
                } else if focus_node.kind_id() == self.block_scalar_id {
                    // We might be on the internal `block_scalar` node, if
                    // we got here via an alias. We need to go up two levels
                    // to get to the mapping pair.
                    focus_node
                        .parent()
                        .expect("missing parent of focus node")
                        .parent()
                        .expect("missing grandparent of focus node")
                } else {
                    // Otherwise, we expect to be on the `block_node`
                    // or `flow_node`, so we go up one level.
                    focus_node.parent().expect("missing parent of focus node")
                };

                if parent_node.kind_id() == self.flow_mapping_id {
                    // Handle the annoying `foo: { key }` case, where our "parent"
                    // is actually a `flow_mapping` instead of a proper block/flow pair.
                    // To handle this, we get the first `flow_node` child of the
                    // flow_mapping, which is the "key".
                    let mut cur = parent_node.walk();
                    parent_node
                        .named_children(&mut cur)
                        .find(|n| n.kind_id() == self.flow_node_id)
                        .ok_or_else(|| {
                            QueryError::MissingChildField(parent_node.kind().into(), "flow_node")
                        })?
                } else {
                    parent_node.child_by_field_name("key").ok_or_else(|| {
                        QueryError::MissingChildField(parent_node.kind().into(), "key")
                    })?
                }
            }
            // Nothing special to do in exact mode.
            QueryMode::Exact => focus_node,
        };

        // If we're extracting "pretty" features, we clean up the final
        // node a bit to have it point to the parent `block_mapping_pair`.
        // This results in a (subjectively) more intuitive extracted feature,
        // since `foo: bar` gets extracted for `foo` instead of just `bar`.
        //
        // NOTE: We might already be on the block_mapping_pair if we terminated
        // with an absent value, in which case we don't need to do this cleanup.
        if matches!(mode, QueryMode::Pretty)
            && matches!(route.route.last(), Some(Component::Key(_)))
            && focus_node.kind_id() != self.block_mapping_pair_id
        {
            focus_node = focus_node.parent().expect("missing parent of focus node")
        }

        Ok(focus_node)
    }

    fn descend<'b>(
        &'b self,
        node: &Node<'b>,
        component: &Component,
    ) -> Result<Node<'b>, QueryError> {
        // The cursor is assumed to start on a block_node or flow_node,
        // which has a child containing the inner scalar/vector/alias
        // type we're descending through.
        //
        // To get to that child, we might have to skip over any
        // anchor nodes that we're not actually aliasing through
        // in this descent step.
        //
        // For example, for a YAML snippet like:
        //
        // ```yaml
        // foo: &foo
        //   bar: baz
        // ```
        //
        // ...the relevant part of the CST looks roughly like:
        //
        // ```
        // block_node         <- `node` points here
        // |--- anchor        <- we need to skip this
        // |--- block_mapping <- we want `child` to point here
        // ```
        let mut child = {
            let mut cursor = node.walk();
            node.named_children(&mut cursor)
                .find(|n| n.kind_id() != self.anchor_id)
                .ok_or_else(|| {
                    QueryError::Other(format!(
                        "node of kind {} has no non-anchor child",
                        node.kind()
                    ))
                })?
        };

        // We might be on an alias node, in which case we need to
        // jump to the alias's target via the anchor map.
        if child.kind_id() == self.alias_id {
            let alias_name = node
                .utf8_text(self.source().as_bytes())
                .expect("impossible: alias name should be UTF-8 by construction");

            child = self
                .resolve_anchor(&alias_name[1..], node.start_byte())
                .ok_or_else(|| QueryError::Other(format!("unknown alias: {}", alias_name)))?;
        }

        // We expect the child to be a sequence or mapping of either
        // flow or block type.
        if child.kind_id() == self.block_mapping_id || child.kind_id() == self.flow_mapping_id {
            match component {
                Component::Key(key) => self.descend_mapping(&child, key),
                Component::Index(idx) => Err(QueryError::ExpectedList(*idx)),
            }
        } else if child.kind_id() == self.block_sequence_id
            || child.kind_id() == self.flow_sequence_id
        {
            match component {
                Component::Index(idx) => self.descend_sequence(&child, *idx),
                Component::Key(key) => Err(QueryError::ExpectedMapping(key.to_string())),
            }
        } else {
            Err(QueryError::UnexpectedNode(child.kind().into()))
        }
    }

    fn descend_mapping<'b>(&self, node: &Node<'b>, expected: &str) -> Result<Node<'b>, QueryError> {
        let mut cur = node.walk();
        for child in node.named_children(&mut cur) {
            let key = match child.kind_id() {
                // If we're on a `flow_pair` or `block_mapping_pair`, we
                // need to get the `key` child.
                id if id == self.flow_pair_id || id == self.block_mapping_pair_id => child
                    .child_by_field_name("key")
                    .ok_or_else(|| QueryError::MissingChildField(child.kind().into(), "key"))?,
                // NOTE: Annoying edge case: if we have a flow mapping
                // like `{ foo }`, then `foo` is a `flow_node` instead
                // of a `flow_pair`.
                id if id == self.flow_node_id => child,
                _ => continue,
            };

            // NOTE: To get the key's actual value, we need to get down to its
            // inner scalar. This is slightly annoying, since keys can be
            // quoted strings with no interior unquoted child. In those cases,
            // we need to manually unquote them.
            //
            // NOTE: text unwraps are infallible, since our document is UTF-8.
            // NOTE: The key might have an anchor prefix (e.g. `{ &v foo: bar }`),
            // so we need to skip any anchor nodes to find the actual scalar.
            let key_value = {
                let mut cursor = key.walk();
                let scalar = key
                    .named_children(&mut cursor)
                    .find(|n| n.kind_id() != self.anchor_id);

                match scalar {
                    Some(scalar) => {
                        let key_value = scalar
                            .utf8_text(self.source().as_bytes())
                            .expect("impossible: value for key should be UTF-8 by construction");

                        match scalar.kind() {
                            "single_quote_scalar" | "double_quote_scalar" => {
                                let mut chars = key_value.chars();
                                chars.next();
                                chars.next_back();
                                chars.as_str()
                            }
                            _ => key_value,
                        }
                    }
                    None => key
                        .utf8_text(self.source().as_bytes())
                        .expect("impossible: key should be UTF-8 by construction"),
                }
            };

            if key_value == expected {
                // HACK: a mapping key might not have a corresponding value,
                // in which case we fall back and return the `block_mapping_pair`
                // itself here. This technically breaks our contract of returning
                // only block_node/flow_node nodes during descent, but not
                // in a way that matters (since an empty value is terminal anyways).
                return Ok(child.child_by_field_name("value").unwrap_or(child));
            }
        }

        // None of the keys in the mapping matched.
        Err(QueryError::ExhaustedMapping(expected.into()))
    }

    /// Given a `block_sequence` or `flow_sequence` node, return
    /// a full list of child nodes after expanding any aliases present.
    ///
    /// The returned child nodes are the inner
    /// `block_node`/`flow_node`/`flow_pair` nodes for each sequence item.
    fn flatten_sequence<'b>(&'b self, node: &Node<'b>) -> Result<Vec<Node<'b>>, QueryError> {
        let mut children = vec![];

        let mut cur = node.walk();
        for child in node.named_children(&mut cur).filter(|child| {
            child.kind_id() == self.block_sequence_item_id
                || child.kind_id() == self.flow_node_id
                || child.kind_id() == self.flow_pair_id
        }) {
            let mut child = child;

            // If we have a `block_sequence_item`, we need to get its
            // inner `block_node`/`flow_node`, which might be interceded
            // by comments.
            if child.kind_id() == self.block_sequence_item_id {
                let mut cur = child.walk();
                child = child
                    .named_children(&mut cur)
                    .find(|c| c.kind_id() == self.block_node_id || c.kind_id() == self.flow_node_id)
                    .ok_or_else(|| {
                        QueryError::MissingChild(child.kind().into(), "block_sequence_item".into())
                    })?;
            }

            // `child` is now a `block_node`, a `flow_node`, or `flow_pair`:
            //
            // `block_node` looks like `- a: b`
            // `flow_node` looks like `- a`
            // `flow_pair` looks like `[a: b]`
            //
            // Aliases are drop-in replacements for their anchored values,
            // so we just keep the child as-is. The alias will be resolved
            // during descent when we navigate into it.
            children.push(child);
        }

        Ok(children)
    }

    fn descend_sequence<'b>(&'b self, node: &Node<'b>, idx: usize) -> Result<Node<'b>, QueryError> {
        let children = self.flatten_sequence(node)?;
        let Some(child) = children.get(idx) else {
            return Err(QueryError::ExhaustedList(idx, children.len()));
        };

        if child.kind_id() == self.flow_pair_id {
            // Similarly, if our index happens to be a `flow_pair`, we need to
            // get the `value` child to get the next `flow_node`.
            // The `value` might not be present (e.g. `{foo: }`), in which case
            // we treat the `flow_pair` itself as terminal like with the mapping hack.
            return Ok(child.child_by_field_name("value").unwrap_or(*child));
        }

        Ok(*child)
    }
}

#[cfg(test)]
mod tests {
    use std::vec;

    use crate::{Component, Document, FeatureKind, QueryError, Route};

    #[test]
    fn test_query_parent() {
        let route = route!("foo", "bar", "baz");
        assert_eq!(
            route.parent().unwrap().route,
            [Component::Key("foo".into()), Component::Key("bar".into())]
        );

        let route = route!("foo");
        assert!(route.parent().is_some());

        let route = Route::from(vec![]);
        assert!(route.parent().is_none());
    }

    #[test]
    fn test_location_spanned_by_comment() {
        let doc = Document::new(
            r#"
foo: bar
# comment
baz: quux
        "#,
        )
        .unwrap();

        // Before the comment.
        assert!(!doc.range_spanned_by_comment(1, 4));
        // Single point within the comment's span.
        assert!(doc.range_spanned_by_comment(13, 13));
        // Within the comment's span.
        assert!(doc.range_spanned_by_comment(13, 15));
        // Starts inside the comment, ends outside.
        assert!(!doc.range_spanned_by_comment(13, 21));
    }

    #[test]
    fn test_offset_inside_comment() {
        let doc = Document::new("foo: bar # abc def").unwrap();

        let comment = doc.source().find('#').unwrap();
        for idx in 0..doc.source().len() {
            if idx < comment {
                assert!(!doc.offset_inside_comment(idx));
            } else {
                assert!(doc.offset_inside_comment(idx));
            }
        }
    }

    #[test]
    fn test_query_builder() {
        let route = route!("foo", "bar", 1, 123, "lol");

        assert_eq!(
            route.route,
            [
                Component::Key("foo".into()),
                Component::Key("bar".into()),
                Component::Index(1),
                Component::Index(123),
                Component::Key("lol".into()),
            ]
        )
    }

    #[test]
    fn test_basic() {
        let doc = r#"
foo: bar
baz:
  sub:
    keys:
      abc:
        - 123
        - 456
        - [a, b, c, {d: e}]
        "#;

        let doc = Document::new(doc).unwrap();
        let route = Route {
            route: vec![
                Component::Key("baz".into()),
                Component::Key("sub".into()),
                Component::Key("keys".into()),
                Component::Key("abc".into()),
                Component::Index(2),
                Component::Index(3),
            ],
        };

        assert_eq!(
            doc.extract_with_leading_whitespace(&doc.query_pretty(&route).unwrap()),
            "{d: e}"
        );
    }

    #[test]
    fn test_top_feature() {
        let doc = r#"
foo: bar
baz:
  abc: def
"#;

        let doc = Document::new(doc).unwrap();
        let feature = doc.top_feature().unwrap();

        assert_eq!(doc.extract(&feature).trim(), doc.source().trim());
        assert_eq!(feature.kind(), FeatureKind::Mapping);
    }

    #[test]
    fn test_feature_comments() {
        let doc = r#"
root: # rootlevel
  a: 1 # foo
  b: 2 # bar
  c: 3
  d: 4 # baz
  e: [1, 2, {nested: key}] # quux

bar: # outside
# outside too
        "#;

        let doc = Document::new(doc).unwrap();

        // Querying the root gives us all comments underneath it.
        let route = Route {
            route: vec![Component::Key("root".into())],
        };
        let feature = doc.query_pretty(&route).unwrap();
        assert_eq!(
            doc.feature_comments(&feature)
                .iter()
                .map(|f| doc.extract(f))
                .collect::<Vec<_>>(),
            &["# rootlevel", "# foo", "# bar", "# baz", "# quux"]
        );

        // Querying a nested key gives us its adjacent comment,
        // even though it's above it on the AST.
        let route = Route {
            route: vec![
                Component::Key("root".into()),
                Component::Key("e".into()),
                Component::Index(1),
            ],
        };
        let feature = doc.query_pretty(&route).unwrap();
        assert_eq!(
            doc.feature_comments(&feature)
                .iter()
                .map(|f| doc.extract(f))
                .collect::<Vec<_>>(),
            &["# quux"]
        );
    }

    #[test]
    fn test_feature_kind() {
        let doc = r#"
block-mapping:
  foo: bar

"block-mapping-quoted":
  foo: bar

block-sequence:
  - foo
  - bar

"block-sequence-quoted":
  - foo
  - bar

flow-mapping: {foo: bar}

flow-sequence: [foo, bar]

scalars:
  - abc
  - 'abc'
  - "abc"
  - 123
  - -123
  - 123.456
  - true
  - false
  - null
  - |
    multiline
    text
  - >
    folded
    text

nested:
  foo:
    - bar
    - baz
    - { a: b }
    - { c: }
"#;
        let doc = Document::new(doc).unwrap();

        for (route, expected_kind) in &[
            (
                vec![Component::Key("block-mapping".into())],
                FeatureKind::Mapping,
            ),
            (
                vec![Component::Key("block-mapping-quoted".into())],
                FeatureKind::Mapping,
            ),
            (
                vec![Component::Key("block-sequence".into())],
                FeatureKind::Sequence,
            ),
            (
                vec![Component::Key("block-sequence-quoted".into())],
                FeatureKind::Sequence,
            ),
            (
                vec![Component::Key("flow-mapping".into())],
                FeatureKind::Mapping,
            ),
            (
                vec![Component::Key("flow-sequence".into())],
                FeatureKind::Sequence,
            ),
            (
                vec![Component::Key("scalars".into()), Component::Index(0)],
                FeatureKind::Scalar,
            ),
            (
                vec![Component::Key("scalars".into()), Component::Index(1)],
                FeatureKind::Scalar,
            ),
            (
                vec![Component::Key("scalars".into()), Component::Index(2)],
                FeatureKind::Scalar,
            ),
            (
                vec![Component::Key("scalars".into()), Component::Index(3)],
                FeatureKind::Scalar,
            ),
            (
                vec![Component::Key("scalars".into()), Component::Index(4)],
                FeatureKind::Scalar,
            ),
            (
                vec![Component::Key("scalars".into()), Component::Index(5)],
                FeatureKind::Scalar,
            ),
            (
                vec![Component::Key("scalars".into()), Component::Index(6)],
                FeatureKind::Scalar,
            ),
            (
                vec![Component::Key("scalars".into()), Component::Index(7)],
                FeatureKind::Scalar,
            ),
            (
                vec![Component::Key("scalars".into()), Component::Index(8)],
                FeatureKind::Scalar,
            ),
            (
                vec![Component::Key("scalars".into()), Component::Index(9)],
                FeatureKind::Scalar,
            ),
            (
                vec![Component::Key("scalars".into()), Component::Index(10)],
                FeatureKind::Scalar,
            ),
            (
                vec![
                    Component::Key("nested".into()),
                    Component::Key("foo".into()),
                    Component::Index(2),
                ],
                FeatureKind::Mapping,
            ),
            (
                vec![
                    Component::Key("nested".into()),
                    Component::Key("foo".into()),
                    Component::Index(3),
                ],
                FeatureKind::Mapping,
            ),
        ] {
            let route = Route::from(route.clone());
            let feature = doc.query_exact(&route).unwrap().unwrap();
            assert_eq!(feature.kind(), *expected_kind);
        }
    }

    #[test]
    fn test_duplicate_anchors() {
        let test_cases: Vec<(&str, Vec<(Route, &str)>)> = vec![
            // Same anchor name defined twice, alias resolves based on document position
            (
                "first: &x value1\nsecond: &x value2\nref: *x",
                vec![(route!("ref"), "value2")],
            ),
            // Alias before redefinition sees old value, alias after sees new value
            (
                "a1: &x old_x\nref_x: *x\na2: &x new_x\nref_x2: *x",
                vec![(route!("ref_x"), "old_x"), (route!("ref_x2"), "new_x")],
            ),
            // Inline flow sequence with duplicate anchor
            (
                "foo: [&x x, *x, &x y, *x]",
                vec![
                    (route!("foo", 0), "x"),
                    (route!("foo", 1), "x"),
                    (route!("foo", 2), "y"),
                    (route!("foo", 3), "y"),
                ],
            ),
        ];

        for (yaml, queries) in test_cases {
            let doc = Document::new(yaml).unwrap();
            for (route, expected) in queries {
                let feature = doc.query_exact(&route).unwrap().unwrap();
                assert_eq!(doc.extract(&feature), expected, "YAML: {}", yaml);
            }
        }
    }

    #[test]
    fn test_anchor_map() {
        let anchors = r#"
foo: &foo-anchor
  bar: &bar-anchor
    baz: quux
        "#;

        let doc = Document::new(anchors).unwrap();
        let anchor_map = doc.tree.borrow_dependent();

        assert_eq!(anchor_map.len(), 2);
        // Each anchor name maps to a BTreeMap of positions -> nodes
        assert_eq!(anchor_map["foo-anchor"].len(), 1);
        assert_eq!(anchor_map["bar-anchor"].len(), 1);
        assert_eq!(
            anchor_map["foo-anchor"].values().next().unwrap().kind(),
            "block_mapping"
        );
        assert_eq!(
            anchor_map["bar-anchor"].values().next().unwrap().kind(),
            "block_mapping"
        );
    }

    #[test]
    fn test_sequence_alias_not_flattened() {
        // Backstop test for #1551
        let doc = r#"
defaults: &defaults
  - a
  - b
  - c
list:
  - *defaults
  - d
  - e
        "#;

        let doc = Document::new(doc).unwrap();

        for (route, expected_kind, expected_value) in [
            (
                route!("list", 0),
                FeatureKind::Sequence,
                "- a\n  - b\n  - c",
            ),
            (route!("list", 1), FeatureKind::Scalar, "d"),
            (route!("list", 2), FeatureKind::Scalar, "e"),
        ] {
            let feature = doc.query_exact(&route).unwrap().unwrap();
            assert_eq!(feature.kind(), expected_kind);
            assert_eq!(doc.extract(&feature).trim(), expected_value);
        }

        assert!(matches!(
            doc.query_exact(&route!("list", 3)),
            Err(QueryError::ExhaustedList(3, 3))
        ));
    }

    #[test]
    fn test_inline_anchor_alias_patterns() {
        let test_cases: Vec<(&str, Vec<(Route, &str)>)> = vec![
            // Basic flow sequence cases
            (
                "foo: [&x v, *x]",
                vec![(route!("foo", 0), "v"), (route!("foo", 1), "v")],
            ),
            (
                "foo: [a, &x v, *x]",
                vec![
                    (route!("foo", 0), "a"),
                    (route!("foo", 1), "v"),
                    (route!("foo", 2), "v"),
                ],
            ),
            (
                "foo: [&a 1, &b 2, *a, *b]",
                vec![
                    (route!("foo", 0), "1"),
                    (route!("foo", 1), "2"),
                    (route!("foo", 2), "1"),
                    (route!("foo", 3), "2"),
                ],
            ),
            // Flow mapping cases
            (
                "top: { &a foo: &b bar, nested: *a, other: *b }",
                vec![
                    (route!("top", "foo"), "bar"),
                    (route!("top", "nested"), "foo"),
                    (route!("top", "other"), "bar"),
                ],
            ),
            (
                "top: { &a k1: v1, &b k2: v2, ref1: *a, ref2: *b }",
                vec![
                    (route!("top", "k1"), "v1"),
                    (route!("top", "k2"), "v2"),
                    (route!("top", "ref1"), "k1"),
                    (route!("top", "ref2"), "k2"),
                ],
            ),
            // Anchor on complex values
            (
                "top: { seq: &x [a, b], ref: *x }",
                vec![
                    (route!("top", "seq", 0), "a"),
                    (route!("top", "ref", 1), "b"),
                ],
            ),
            (
                "top: { map: &x {a: 1}, ref: *x }",
                vec![
                    (route!("top", "map", "a"), "1"),
                    (route!("top", "ref", "a"), "1"),
                ],
            ),
            // Quoted keys with anchors (alias returns the quoted form)
            (
                r#"top: { &x "foo": bar, nested: *x }"#,
                vec![
                    (route!("top", "foo"), "bar"),
                    (route!("top", "nested"), "\"foo\""),
                ],
            ),
            (
                "top: { &x 'foo': bar, nested: *x }",
                vec![
                    (route!("top", "foo"), "bar"),
                    (route!("top", "nested"), "'foo'"),
                ],
            ),
        ];

        for (yaml, queries) in test_cases {
            let doc = Document::new(yaml).unwrap();
            for (route, expected) in queries {
                let feature = doc.query_exact(&route).unwrap().unwrap();
                assert_eq!(doc.extract(&feature), expected, "YAML: {}", yaml);
            }
        }
    }
}
