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

impl Location {
    /// Create a Location from a SpanInfo and LineIndex.
    fn from_span(span: &SpanInfo, _line_index: &LineIndex) -> Self {
        Self {
            byte_span: (span.byte_range.start, span.byte_range.end),
            point_span: (
                (span.point_range.start.row, span.point_range.start.column),
                (span.point_range.end.row, span.point_range.end.column),
            ),
        }
    }
}

/// Unique identifier for a CST node. Provides O(1) parent access without lifetime issues.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct NodeId(usize);

/// Storage for all CST nodes. Single source of truth for the entire tree.
#[derive(Clone, Debug)]
struct CstStore {
    /// All nodes stored in a contiguous Vec for cache-friendly access.
    nodes: Vec<YamlNodeData>,
    /// The root node of the document.
    root_id: NodeId,
}

/// A node in the CST with parent tracking via NodeId.
#[derive(Clone, Debug)]
struct YamlNodeData {
    /// Parent node ID (None only for root).
    parent_id: Option<NodeId>,
    /// The node's type and data.
    kind: YamlNodeKind,
}

#[derive(Clone, Debug)]
enum YamlNodeKind {
    Mapping(MappingData),
    Sequence(SequenceData),
    Scalar(ScalarData),
    /// An alias node that references an anchor's content.
    Alias(AliasData),
}

/// A mapping node (object/dict) in the reduced CST.
#[derive(Clone, Debug)]
struct MappingData {
    /// Map from key string to entry (key location + optional value node).
    entries: HashMap<String, MappingEntry>,

    /// The exact span of this mapping in the source.
    exact_span: SpanInfo,
}

#[derive(Clone, Debug)]
struct MappingEntry {
    /// The key node ID.
    key_id: NodeId,

    /// The value node ID (None for absent values like `foo:` or `{ foo }`).
    value_id: Option<NodeId>,

    /// Span of the entire mapping pair (key: value).
    pair_span: SpanInfo,
}

/// A sequence node (array/list) in the reduced CST.
#[derive(Clone, Debug)]
struct SequenceData {
    /// The item node IDs in this sequence.
    item_ids: Vec<NodeId>,

    /// The exact span of this sequence in the source.
    exact_span: SpanInfo,
}

/// A scalar node (leaf value) in the reduced CST.
#[derive(Clone, Debug)]
struct ScalarData {
    /// The exact span of this scalar in the source.
    exact_span: SpanInfo,

    /// The span including any block scalar indicator (|, >).
    pretty_span: Option<SpanInfo>,
}

/// An alias node that references an anchor's content.
#[derive(Clone, Debug)]
struct AliasData {
    /// The span of the alias itself (e.g., `*foo`).
    span: SpanInfo,

    /// The NodeId of the anchored content this alias references.
    target_id: NodeId,
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
pub struct Feature<'doc> {
    /// Reference to the CST node by ID (None for key-only features).
    node_id: Option<NodeId>,

    /// Borrow of the document for node lookups.
    doc: &'doc Document,

    /// The exact location of the route result.
    pub location: Location,

    /// The "context" location for the route result.
    /// This is typically the surrounding mapping or list structure.
    pub context: Option<Location>,
}

impl<'doc> Feature<'doc> {
    /// Return this feature's parent feature, if it has one.
    pub fn parent(&self) -> Option<Feature<'doc>> {
        let node_id = self.node_id?;
        let node = self.doc.cst.nodes.get(node_id.0)?;
        let parent_id = node.parent_id?;

        Some(Feature {
            node_id: Some(parent_id),
            doc: self.doc,
            location: self
                .doc
                .compute_location_for_node(parent_id, QueryMode::Exact),
            context: None,
        })
    }

    /// Return this feature's [`FeatureKind`].
    pub fn kind(&self) -> FeatureKind {
        let Some(id) = self.node_id else {
            return FeatureKind::Scalar; // Key-only features
        };

        let Some(node) = self.doc.cst.nodes.get(id.0) else {
            // Invalid NodeId - should never happen, but return safe default
            return FeatureKind::Scalar;
        };

        match &node.kind {
            YamlNodeKind::Mapping(_) => FeatureKind::Mapping,
            YamlNodeKind::Sequence(_) => FeatureKind::Sequence,
            YamlNodeKind::Scalar(_) => FeatureKind::Scalar,
            YamlNodeKind::Alias(alias) => {
                // Follow the alias to get the target's kind
                match &self.doc.cst.nodes[alias.target_id.0].kind {
                    YamlNodeKind::Mapping(_) => FeatureKind::Mapping,
                    YamlNodeKind::Sequence(_) => FeatureKind::Sequence,
                    YamlNodeKind::Scalar(_) => FeatureKind::Scalar,
                    YamlNodeKind::Alias(_) => panic!("nested alias in CST"),
                }
            }
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

// Note: The From<Node> impl has been removed in favor of CST-based Feature construction

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
    /// Make extracted features as "exact" as possible, e.g. by
    /// including only the exact span of the route result.
    Exact,
}

/// Result of a CST query, preserving context about how the route terminated.
enum QueryResult<'a> {
    /// Route ended at a node (sequence index or non-key traversal).
    Node(NodeId),
    /// Route ended with a key lookup in a mapping.
    MappingKey {
        parent_id: NodeId,
        entry: &'a MappingEntry,
    },
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
    /// Map from tree-sitter node byte positions to CST NodeIds.
    /// Used to look up the CST NodeId for anchor parents when resolving aliases.
    ts_node_to_id: HashMap<usize, NodeId>,
}

impl CstBuilder {
    fn new() -> Self {
        Self {
            nodes: Vec::new(),
            next_id: 0,
            ts_node_to_id: HashMap::new(),
        }
    }

    fn alloc_id(&mut self) -> NodeId {
        let id = NodeId(self.next_id);
        self.next_id += 1;
        // Reserve a slot in the Vec to ensure NodeId matches the index
        self.nodes.push(YamlNodeData {
            parent_id: None, // Placeholder, will be set by set_node
            kind: YamlNodeKind::Scalar(ScalarData {
                exact_span: SpanInfo {
                    byte_range: 0..0,
                    point_range: tree_sitter::Point { row: 0, column: 0 }..tree_sitter::Point {
                        row: 0,
                        column: 0,
                    },
                },
                pretty_span: None,
            }), // Placeholder
        });
        id
    }

    fn set_node(&mut self, id: NodeId, parent_id: Option<NodeId>, kind: YamlNodeKind) {
        // Update the node at the correct index
        self.nodes[id.0] = YamlNodeData { parent_id, kind };
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
    // Skip wrapper nodes (stream, document, block_node, flow_node) and non-content nodes (comments, directives)
    let mut current = node;
    while matches!(
        current.kind(),
        "stream" | "document" | "block_node" | "flow_node"
    ) {
        // Find first content child (skip comments, directives, document markers)
        let mut child_cursor = current.walk();
        if let Some(child) = current.named_children(&mut child_cursor).find(|n| {
            !matches!(
                n.kind(),
                "comment" | "yaml_directive" | "tag_directive" | "reserved_directive"
            )
        }) {
            current = child;
        } else {
            break;
        }
    }

    // Handle anchors: skip to the anchored content (also skipping comments)
    if current.kind() == "anchor" {
        if let Some(parent) = current.parent() {
            let mut cursor = parent.walk();
            if let Some(content) = parent
                .named_children(&mut cursor)
                .find(|child| child.kind() != "anchor" && child.kind() != "comment")
            {
                current = content;
            }
        }
    }

    // Handle aliases: create an Alias node that references the anchor's content
    if current.kind() == "alias" {
        let alias_span = span_from_node(&current);

        // Extract alias name by slicing off the leading '*'
        let alias_text = current
            .utf8_text(source.as_bytes())
            .map_err(|_| QueryError::Other("invalid UTF-8 in alias".into()))?;
        let alias_name = &alias_text[1..]; // Skip the '*' character

        // Find the most recent anchor with this name
        let target_ts_node = anchor_map
            .get(alias_name)
            .and_then(|positions| positions.range(..current.start_byte()).next_back())
            .map(|(_, node)| *node)
            .ok_or_else(|| QueryError::Other(format!("undefined alias: {}", alias_name)))?;

        // Look up the CST NodeId for the anchor's content (already built)
        let target_id = *builder
            .ts_node_to_id
            .get(&target_ts_node.start_byte())
            .ok_or_else(|| {
                QueryError::Other(format!("anchor content not yet built for alias: {}", alias_name))
            })?;

        // Create an Alias node with its own span but referencing the target
        let node_id = builder.alloc_id();
        builder.set_node(
            node_id,
            parent_id,
            YamlNodeKind::Alias(AliasData {
                span: alias_span,
                target_id,
            }),
        );

        return Ok(node_id);
    }

    // Allocate ID for this node
    let node_id = builder.alloc_id();

    // Build node kind based on type
    let kind = match current.kind() {
        "block_mapping" | "flow_mapping" | "flow_pair" | "block_mapping_pair" => {
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
            })
        }
    };

    // Store node with parent link
    builder.set_node(node_id, parent_id, kind);

    // Register this tree-sitter node's position in the mapping for alias resolution
    builder.ts_node_to_id.insert(current.start_byte(), node_id);

    Ok(node_id)
}

/// Build a mapping node from a tree-sitter mapping or pair node.
///
/// Handles both explicit mappings (block_mapping, flow_mapping) which contain
/// multiple pairs, and implicit single-entry mappings from standalone pairs
/// (e.g., `key: value` appearing directly in a sequence like `[1, 2, key: value]`).
fn build_mapping_node(
    builder: &mut CstBuilder,
    node: Node,
    parent_id: NodeId,
    source: &str,
    anchor_map: &AnchorMap,
) -> Result<YamlNodeKind, QueryError> {
    let mut entries = HashMap::new();

    // Determine if this is a standalone pair (implicit mapping) or a full mapping
    let pairs: Vec<(Node, SpanInfo)> = match node.kind() {
        "flow_pair" | "block_mapping_pair" => {
            // Standalone pair - treat as single-entry mapping
            vec![(node, span_from_node(&node))]
        }
        _ => {
            // Full mapping - collect all pair children
            let mut cursor = node.walk();
            node.named_children(&mut cursor)
                .filter_map(|child| match child.kind() {
                    "flow_pair" | "block_mapping_pair" => {
                        Some((child, span_from_node(&child)))
                    }
                    "flow_node" => {
                        // Bare key in flow mapping like { foo }
                        // Use parent's span for yamlpatch compatibility
                        Some((child, span_from_node(&node)))
                    }
                    _ => None,
                })
                .collect()
        }
    };

    for (pair_node, pair_span) in pairs {
        // Extract key node - for bare flow_node, it IS the key
        let (key_node, value_node_opt) = if pair_node.kind() == "flow_node" {
            (pair_node, None)
        } else {
            let key = pair_node
                .child_by_field_name("key")
                .ok_or_else(|| QueryError::MissingChildField(pair_node.kind().into(), "key"))?;
            let value = pair_node.child_by_field_name("value");
            (key, value)
        };

        let key_str = extract_key_string(&key_node, source)?;

        // Build key as a CST node
        let key_id = build_node(builder, key_node, Some(parent_id), source, anchor_map)?;

        let value_id = if let Some(val_node) = value_node_opt {
            Some(build_node(builder, val_node, Some(parent_id), source, anchor_map)?)
        } else {
            None
        };

        entries.insert(
            key_str,
            MappingEntry {
                key_id,
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
        // Skip comments - they are not sequence items
        if child.kind() == "comment" {
            continue;
        }

        // Handle block_sequence_item wrappers
        let content = if child.kind() == "block_sequence_item" {
            // Get the first non-comment named child
            let mut item_cursor = child.walk();
            child
                .named_children(&mut item_cursor)
                .find(|n| n.kind() != "comment")
                .unwrap_or(child)
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
fn extract_key_string(node: &Node, source: &str) -> Result<String, QueryError> {
    // Skip anchor nodes to get to the actual key
    let mut cursor = node.walk();
    let scalar = node
        .named_children(&mut cursor)
        .find(|n| n.kind() != "anchor")
        .unwrap_or(*node);

    let text = scalar
        .utf8_text(source.as_bytes())
        .map_err(|e| QueryError::Other(format!("invalid UTF-8 in key: {}", e)))?;

    // Unquote if necessary
    let unquoted = match scalar.kind() {
        "single_quote_scalar" | "double_quote_scalar" => {
            let mut chars = text.chars();
            chars.next(); // Skip opening quote
            chars.next_back(); // Skip closing quote
            chars.as_str().to_string()
        }
        _ => text.to_string(),
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

    // Field IDs for tree-sitter nodes, used for comment extraction
    comment_id: u16,
}

impl std::fmt::Debug for Document {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Document")
            .field("cst", &self.cst)
            .field("line_index", &self.line_index)
            .field("comment_id", &self.comment_id)
            .finish()
    }
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
            None, // No parent for root
            tree_struct.borrow_owner().source.as_str(),
            tree_struct.borrow_dependent(),
        )?;
        let cst = builder.finish(root_id);

        Ok(Self {
            tree: tree_struct,
            cst,
            line_index,
            comment_id: language.id_for_node_kind("comment", true),
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

    /// Compute a Location for a CST node by ID.
    fn compute_location_for_node(&self, node_id: NodeId, mode: QueryMode) -> Location {
        let node = &self.cst.nodes[node_id.0];
        let span = match (&node.kind, mode) {
            (YamlNodeKind::Scalar(scalar), QueryMode::Pretty) => {
                scalar.pretty_span.as_ref().unwrap_or(&scalar.exact_span)
            }
            (YamlNodeKind::Mapping(map), _) => &map.exact_span,
            (YamlNodeKind::Sequence(seq), _) => &seq.exact_span,
            (YamlNodeKind::Scalar(scalar), _) => &scalar.exact_span,
            // Alias nodes use their own span (the *alias location), not the target's
            (YamlNodeKind::Alias(alias), _) => &alias.span,
        };

        Location::from_span(span, &self.line_index)
    }

    /// Resolve an alias to its target, following chains if necessary.
    fn resolve_alias(&self, node_id: NodeId) -> NodeId {
        match &self.cst.nodes[node_id.0].kind {
            YamlNodeKind::Alias(alias) => self.resolve_alias(alias.target_id),
            _ => node_id,
        }
    }

    /// Query the CST following the route, returning context about how the route terminated.
    fn query_cst(&self, route: &Route) -> Result<QueryResult<'_>, QueryError> {
        let mut current_id = self.resolve_alias(self.cst.root_id);

        for (i, component) in route.route.iter().enumerate() {
            let current = &self.cst.nodes[current_id.0];
            let is_last = i == route.route.len() - 1;

            match (&current.kind, component) {
                (YamlNodeKind::Mapping(map), Component::Key(key)) => {
                    let entry = map
                        .entries
                        .get(key.as_ref())
                        .ok_or_else(|| QueryError::ExhaustedMapping(key.to_string()))?;

                    if is_last {
                        return Ok(QueryResult::MappingKey {
                            parent_id: current_id,
                            entry,
                        });
                    }

                    // Resolve aliases when moving to next node
                    current_id = self.resolve_alias(entry.value_id.ok_or_else(|| {
                        QueryError::Other(format!("absent value for key: {}", key))
                    })?);
                }
                (YamlNodeKind::Sequence(seq), Component::Index(idx)) => {
                    // Resolve aliases when moving to next node
                    current_id = self.resolve_alias(*seq
                        .item_ids
                        .get(*idx)
                        .ok_or(QueryError::ExhaustedList(*idx, seq.item_ids.len()))?);
                }
                (YamlNodeKind::Mapping(_), Component::Index(idx)) => {
                    return Err(QueryError::ExpectedList(*idx));
                }
                (YamlNodeKind::Sequence(_), Component::Key(key)) => {
                    return Err(QueryError::ExpectedMapping(key.to_string()));
                }
                (YamlNodeKind::Scalar(_), Component::Key(key)) => {
                    return Err(QueryError::ExpectedMapping(key.to_string()));
                }
                (YamlNodeKind::Scalar(_), Component::Index(idx)) => {
                    return Err(QueryError::ExpectedList(*idx));
                }
                (YamlNodeKind::Alias(_), _) => {
                    panic!("alias should have been resolved")
                }
            }
        }

        Ok(QueryResult::Node(current_id))
    }

    /// Create a Feature from a CST node.
    fn create_feature(&self, node_id: NodeId, mode: QueryMode) -> Feature<'_> {
        let location = self.compute_location_for_node(node_id, mode);

        // Compute context from parent if it exists
        let node = &self.cst.nodes[node_id.0];
        let context = node
            .parent_id
            .map(|parent_id| self.compute_location_for_node(parent_id, QueryMode::Exact));

        Feature {
            node_id: Some(node_id),
            doc: self,
            location,
            context,
        }
    }

    /// Returns a [`Feature`] for the topmost semantic object in this document.
    ///
    /// This is typically useful as a "fallback" feature, e.g. for positioning
    /// relative to the "top" of the document.
    pub fn top_feature(&self) -> Result<Feature<'_>, QueryError> {
        Ok(self.create_feature(self.cst.root_id, QueryMode::Exact))
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
        self.query_exact(route).is_ok()
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
        match self.query_cst(route)? {
            QueryResult::MappingKey { parent_id, entry } => Ok(Feature {
                node_id: entry.value_id,
                doc: self,
                location: Location::from_span(&entry.pair_span, &self.line_index),
                context: Some(self.compute_location_for_node(parent_id, QueryMode::Exact)),
            }),
            QueryResult::Node(node_id) => Ok(self.create_feature(node_id, QueryMode::Pretty)),
        }
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
        match self.query_cst(route)? {
            QueryResult::MappingKey { parent_id, entry } => {
                let Some(value_id) = entry.value_id else {
                    // Absent value like `foo:` or `{ foo }`
                    return Ok(None);
                };
                // Resolve aliases to get the semantic value's location
                let resolved_id = self.resolve_alias(value_id);
                Ok(Some(Feature {
                    node_id: Some(resolved_id),
                    doc: self,
                    location: self.compute_location_for_node(resolved_id, QueryMode::Exact),
                    context: Some(self.compute_location_for_node(parent_id, QueryMode::Exact)),
                }))
            }
            QueryResult::Node(node_id) => Ok(Some(self.create_feature(node_id, QueryMode::Exact))),
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
        match self.query_cst(route)? {
            QueryResult::MappingKey { parent_id, entry } => Ok(Feature {
                node_id: Some(entry.key_id),
                doc: self,
                location: self.compute_location_for_node(entry.key_id, QueryMode::Exact),
                context: Some(self.compute_location_for_node(parent_id, QueryMode::Exact)),
            }),
            QueryResult::Node(_) => Err(QueryError::Other(
                "key_only requires route to end with key".into(),
            )),
        }
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
            doc: &'tree Document,
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
                    .map(|c| Feature {
                        node_id: None, // Comments aren't in CST
                        doc,
                        location: Location::from(c),
                        context: None,
                    }),
            );

            for child in node.children(&mut cur) {
                comments.extend(trawl(doc, &child, comment_id, start_line, end_line));
            }

            comments
        }

        trawl(
            self,
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
