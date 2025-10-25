use std::borrow::Cow;
use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Arc, RwLock};

use eframe::egui::text::{LayoutJob, LayoutSection};
use eframe::egui::{self, Color32, TextFormat};
use egui_node_graph2::*;
use rsprocess::translator::Formatter;

// The folder where `eframe` will store its state is
// * Linux:   `/home/UserName/.local/share/APP_ID`
// * macOS:   `/Users/UserName/Library/Application Support/APP_ID`
// * Windows: `C:\Users\UserName\AppData\Roaming\APP_ID\data`

// ========= First, define your user data types =============

/// The NodeData holds the data available in each node.
#[cfg_attr(
    feature = "persistence",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct NodeData {
    pub(crate) template: NodeInstruction,
}

/// `BasicDataType`'s are what defines the possible range of connections when
/// attaching two ports together.
#[derive(PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "persistence",
    derive(serde::Serialize, serde::Deserialize)
)]
#[allow(dead_code)]
pub enum BasicDataType {
    Error,

    String,
    Path,
    System,
    PositiveInt,
    Symbol,
    Experiment,
    Graph,
    GroupingFunction,
    DisplayNode,
    DisplayEdge,
    ColorNode,
    ColorEdge,
    Environment,
    Set,
    Context,
    Reactions,
    PositiveSystem,
    Trace,
    PositiveTrace,
    PositiveSet,
    PositiveEnvironment,
    PositiveContext,
    PositiveReactions,
}

/// Should reflect `BasicDataType`'s values, holding the data that will be
/// assigned to each arch between nodes. The library makes no attempt to check
/// this consistency.
#[derive(Clone, Debug)]
#[cfg_attr(
    feature = "persistence",
    derive(serde::Serialize, serde::Deserialize)
)]
pub enum BasicValue {
    SaveString {
        path:  String,
        value: String,
    },
    Error {
        value: LayoutJob,
    },

    String {
        value: String,
    },
    Path {
        value: String,
    },
    System {
        value: rsprocess::system::System,
    },
    PositiveInt {
        value: usize,
    },
    Symbol {
        value: String,
    },
    Experiment {
        value: (Vec<u32>, Vec<rsprocess::set::Set>),
    },
    Graph {
        value:
            petgraph::Graph<rsprocess::system::System, rsprocess::label::Label>,
    },
    GroupingFunction {
        value: assert::relabel::Assert,
    },
    DisplayNode {
        value: rsprocess::graph::NodeDisplay,
    },
    DisplayEdge {
        value: rsprocess::graph::EdgeDisplay,
    },
    ColorNode {
        value: rsprocess::graph::NodeColor,
    },
    ColorEdge {
        value: rsprocess::graph::EdgeColor,
    },
    Environment {
        value: rsprocess::environment::Environment,
    },
    Set {
        value: rsprocess::set::Set,
    },
    Context {
        value: rsprocess::process::Process,
    },
    Reactions {
        value: Vec<rsprocess::reaction::Reaction>,
    },
    PositiveSystem {
        value: rsprocess::system::PositiveSystem,
    },
    Trace {
        value: rsprocess::trace::SlicingTrace<
            rsprocess::set::Set,
            rsprocess::reaction::Reaction,
            rsprocess::system::System,
        >,
    },
    PositiveTrace {
        value: rsprocess::trace::SlicingTrace<
            rsprocess::set::PositiveSet,
            rsprocess::reaction::PositiveReaction,
            rsprocess::system::PositiveSystem,
        >,
    },
    PositiveSet {
        value: rsprocess::set::PositiveSet,
    },
    PositiveEnvironment {
        value: rsprocess::environment::PositiveEnvironment,
    },
    PositiveContext {
        value: rsprocess::process::PositiveProcess,
    },
    PositiveReactions {
        value: Vec<rsprocess::reaction::PositiveReaction>,
    },
}

impl Hash for BasicValue {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        macro_rules! default_hash {
            ($($i:ident),*) => (
                match self {
                    $(Self::$i { value } => { value.hash(state) })*
                    _ => {}
                }
            );
        }
        // ---------------------------------------------------------------------
        // -------------------- Add here additional types ----------------------
        // ---------------------------------------------------------------------
        default_hash!(
            String,
            Path,
            System,
            PositiveInt,
            Symbol,
            Experiment,
            GroupingFunction,
            DisplayNode,
            DisplayEdge,
            ColorNode,
            ColorEdge,
            Error,
            Environment,
            Set,
            Context,
            Reactions,
            PositiveSystem,
            Trace,
            PositiveTrace,
            PositiveSet,
            PositiveEnvironment,
            PositiveContext,
            PositiveReactions
        );

        match self {
            | Self::SaveString { path, value } => {
                path.hash(state);
                value.hash(state);
            },
            | Self::Graph { value } => {
                value.node_weights().for_each(|e| e.hash(state));
                value.edge_weights().for_each(|e| e.hash(state));
            },
            | _ => {},
        }
    }
}

impl Default for BasicValue {
    fn default() -> Self {
        // NOTE: dummy `Default` implementation, to circumvent some internal
        // borrow checker issues.
        Self::String {
            value: String::default(),
        }
    }
}

/// What will be displayed in the "new node" popup. The user code needs to tell
/// the library how to convert a NodeTemplate into a Node. Also used how the
/// information should be processed.
#[derive(Clone, Copy)]
#[cfg_attr(
    feature = "persistence",
    derive(serde::Serialize, serde::Deserialize)
)]
pub enum NodeInstruction {
    String,
    Path,
    ReadPath,
    Symbol,
    Experiment,
    GroupFunction,
    SaveString,
    DisplayNode,
    DisplayEdge,
    ColorNode,
    ColorEdge,
    Environment,
    Set,
    Context,
    Reactions,
    PositiveSet,
    ToPositiveSet,

    // system instructions
    ComposeSystem,
    DecomposeSystem,
    System,
    Statistics,
    Target,
    Run,
    Loop,
    Frequency,
    LimitFrequency,
    FastFrequency,
    SystemGraph,

    // graph instructions
    BisimilarityKanellakisSmolka,
    BisimilarityPaigeTarjanNoLabels,
    BisimilarityPaigeTarjan,
    Dot,
    GraphML,

    // positive system instructions
    PositiveSystem,
    PositiveTarget,
    PositiveRun,
    PositiveLoop,
    PositiveFrequency,
    PositiveLimitFrequency,
    PositiveFastFrequency,
    ComposePositiveSystem,
    DecomposePositiveSystem,
    // PositiveGraph,

    // trace instructions
    Trace,
    PositiveTrace,
    SliceTrace,
    PositiveSliceTrace,
}

impl NodeInstruction {
    pub(crate) fn inputs(&self) -> Vec<(String, BasicDataType)> {
        use BasicDataType::*;
        match self {
            | Self::String => vec![("value", String)],
            | Self::Path => vec![("value", String)],
            | Self::ReadPath => vec![("path", Path)],
            | Self::System => vec![("string", String)],
            | Self::Statistics => vec![("sys", System)],
            | Self::Target => vec![("sys", System), ("limit", PositiveInt)],
            | Self::Run => vec![("sys", System), ("limit", PositiveInt)],
            | Self::Loop => vec![("sys", System), ("symbol", Symbol)],
            | Self::Symbol => vec![("string", String)],
            | Self::Frequency => vec![("sys", System)],
            | Self::LimitFrequency =>
                vec![("sys", System), ("experiment", Experiment)],
            | Self::Experiment => vec![("string", String)],
            | Self::FastFrequency =>
                vec![("sys", System), ("experiment", Experiment)],
            | Self::BisimilarityKanellakisSmolka => vec![
                ("first graph", Graph),
                ("second graph", Graph),
                ("group", GroupingFunction),
            ],
            | Self::BisimilarityPaigeTarjanNoLabels => vec![
                ("first graph", Graph),
                ("second graph", Graph),
                ("group", GroupingFunction),
            ],
            | Self::BisimilarityPaigeTarjan => vec![
                ("first graph", Graph),
                ("second graph", Graph),
                ("group", GroupingFunction),
            ],
            | Self::GroupFunction => vec![("string", String)],
            | Self::SystemGraph => vec![("sys", System)],
            | Self::SaveString => vec![("path", Path), ("string", String)],
            | Self::Dot => vec![
                ("graph", Graph),
                ("display node", DisplayNode),
                ("display edge", DisplayEdge),
                ("color node", ColorNode),
                ("color edge", ColorEdge),
            ],
            | Self::DisplayNode => vec![("value", String)],
            | Self::DisplayEdge => vec![("value", String)],
            | Self::ColorNode => vec![("value", String)],
            | Self::ColorEdge => vec![("value", String)],
            | Self::GraphML => vec![
                ("graph", Graph),
                ("display node", DisplayNode),
                ("display edge", DisplayEdge),
            ],
            | Self::ComposeSystem => vec![
                ("environment", Environment),
                ("initial entities", Set),
                ("context", Context),
                ("reactions", Reactions),
            ],
            | Self::Environment => vec![("string", String)],
            | Self::Set => vec![("string", String)],
            | Self::Context => vec![("string", String)],
            | Self::Reactions => vec![("string", String)],
            | Self::PositiveSystem => vec![("system", System)],
            | Self::PositiveTarget =>
                vec![("sys", PositiveSystem), ("limit", PositiveInt)],
            | Self::PositiveRun =>
                vec![("sys", PositiveSystem), ("limit", PositiveInt)],
            | Self::PositiveLoop =>
                vec![("sys", PositiveSystem), ("symbol", Symbol)],
            | Self::PositiveFrequency => vec![("sys", PositiveSystem)],
            | Self::PositiveLimitFrequency =>
                vec![("sys", PositiveSystem), ("experiment", Experiment)],
            | Self::PositiveFastFrequency =>
                vec![("sys", PositiveSystem), ("experiment", Experiment)],
            | Self::Trace => vec![("sys", System), ("limit", PositiveInt)],
            | Self::PositiveTrace =>
                vec![("sys", PositiveSystem), ("limit", PositiveInt)],
            | Self::SliceTrace => vec![("trace", Trace), ("marking", Set)],
            | Self::PositiveSliceTrace =>
                vec![("trace", PositiveTrace), ("marking", PositiveSet)],
            | Self::PositiveSet => vec![("string", String)],
            | Self::ToPositiveSet => vec![("value", Set)],
            | Self::DecomposeSystem => vec![("system", System)],
            | Self::ComposePositiveSystem => vec![
                ("environment", PositiveEnvironment),
                ("initial entities", PositiveSet),
                ("context", PositiveContext),
                ("reactions", PositiveReactions),
            ],
            | Self::DecomposePositiveSystem => vec![("system", PositiveSystem)],
        }
        .into_iter()
        .map(|e| (e.0.to_string(), e.1))
        .collect::<Vec<_>>()
    }

    pub(crate) fn output(&self) -> Vec<(String, BasicDataType)> {
        use BasicDataType::*;
        let res = match self {
            | Self::String => vec![("out", String)],
            | Self::Path => vec![("out", Path)],
            | Self::ReadPath => vec![("out", String)],
            | Self::System => vec![("system", System)],
            | Self::Statistics => vec![("out", String)],
            | Self::Target => vec![("out", String)],
            | Self::Run => vec![("out", String)],
            | Self::Loop => vec![("out", String)],
            | Self::Symbol => vec![("out", Symbol)],
            | Self::Frequency => vec![("out", String)],
            | Self::LimitFrequency => vec![("out", String)],
            | Self::Experiment => vec![("out", Experiment)],
            | Self::FastFrequency => vec![("out", String)],
            | Self::BisimilarityKanellakisSmolka => vec![("out", String)],
            | Self::BisimilarityPaigeTarjanNoLabels => vec![("out", String)],
            | Self::BisimilarityPaigeTarjan => vec![("out", String)],
            | Self::GroupFunction => vec![("out", GroupingFunction)],
            | Self::SystemGraph => vec![("out", Graph)],
            | Self::SaveString => vec![],
            | Self::Dot => vec![("out", String)],
            | Self::DisplayNode => vec![("out", DisplayNode)],
            | Self::DisplayEdge => vec![("out", DisplayEdge)],
            | Self::ColorNode => vec![("out", ColorNode)],
            | Self::ColorEdge => vec![("out", ColorEdge)],
            | Self::GraphML => vec![("out", String)],
            | Self::ComposeSystem => vec![("out", System)],
            | Self::Environment => vec![("out", Environment)],
            | Self::Set => vec![("out", Set)],
            | Self::Context => vec![("out", Context)],
            | Self::Reactions => vec![("out", Reactions)],
            | Self::PositiveSystem => vec![("out", PositiveSystem)],
            | Self::PositiveTarget => vec![("out", String)],
            | Self::PositiveRun => vec![("out", String)],
            | Self::PositiveLoop => vec![("out", String)],
            | Self::PositiveFrequency => vec![("out", String)],
            | Self::PositiveLimitFrequency => vec![("out", String)],
            | Self::PositiveFastFrequency => vec![("out", String)],
            | Self::Trace => vec![("out", Trace)],
            | Self::PositiveTrace => vec![("out", PositiveTrace)],
            | Self::SliceTrace => vec![("out", Trace)],
            | Self::PositiveSliceTrace => vec![("out", PositiveTrace)],
            | Self::PositiveSet => vec![("out", PositiveSet)],
            | Self::ToPositiveSet => vec![("out", PositiveSet)],
            | Self::ComposePositiveSystem => vec![("out", PositiveSystem)],
            | Self::DecomposeSystem => vec![
                ("environment", Environment),
                ("initial entities", Set),
                ("context", Context),
                ("reactions", Reactions),
            ],
            | Self::DecomposePositiveSystem => vec![
                ("environment", PositiveEnvironment),
                ("initial entities", PositiveSet),
                ("context", PositiveContext),
                ("reactions", PositiveReactions),
            ],
        };
        res.into_iter().map(|res| (res.0.to_string(), res.1)).collect::<_>()
    }

    #[allow(clippy::type_complexity)]
    pub(crate) fn create_input(
        ty: BasicDataType,
    ) -> Box<dyn Fn(NodeId, &mut NodeGraph, &str)> {
        macro_rules! helper {
            ($name: ident, $def: expr) => {
                Box::new(
                    |node_id: NodeId, graph: &mut NodeGraph, name: &str| {
                        graph.add_input_param(
                            node_id,
                            name.to_string(),
                            BasicDataType::$name,
                            BasicValue::$name { value: $def },
                            InputParamKind::ConnectionOrConstant,
                            true,
                        );
                    },
                )
            };
        }

        match ty {
            | BasicDataType::Error =>
                Box::new(|_: NodeId, _: &mut NodeGraph, _: &str| {}),

            | BasicDataType::Path => helper!(Path, String::new()),
            | BasicDataType::String => helper!(String, String::new()),
            | BasicDataType::System =>
                helper!(System, rsprocess::system::System::default()),
            | BasicDataType::PositiveInt => helper!(PositiveInt, 1),
            | BasicDataType::Symbol => helper!(Symbol, "*".into()),
            | BasicDataType::Experiment =>
                helper!(Experiment, (vec![], vec![])),
            | BasicDataType::Graph => helper!(Graph, petgraph::Graph::new()),
            | BasicDataType::GroupingFunction =>
                helper!(GroupingFunction, assert::relabel::Assert::default()),
            | BasicDataType::DisplayNode =>
                helper!(DisplayNode, rsprocess::graph::NodeDisplay {
                    base: vec![rsprocess::graph::NodeDisplayBase::Hide],
                }),
            | BasicDataType::DisplayEdge =>
                helper!(DisplayEdge, rsprocess::graph::EdgeDisplay {
                    base: vec![rsprocess::graph::EdgeDisplayBase::Hide],
                }),
            | BasicDataType::ColorNode =>
                helper!(ColorNode, rsprocess::graph::NodeColor {
                    conditionals: vec![],
                    base_color:   "white".into(),
                }),
            | BasicDataType::ColorEdge =>
                helper!(ColorEdge, rsprocess::graph::EdgeColor {
                    conditionals: vec![],
                    base_color:   "black".into(),
                }),
            | BasicDataType::Environment => helper!(
                Environment,
                rsprocess::environment::Environment::default()
            ),
            | BasicDataType::Set =>
                helper!(Set, rsprocess::set::Set::default()),
            | BasicDataType::Context =>
                helper!(Context, rsprocess::process::Process::default()),
            | BasicDataType::Reactions => helper!(Reactions, vec![]),
            | BasicDataType::PositiveSystem => helper!(
                PositiveSystem,
                rsprocess::system::PositiveSystem::default()
            ),
            | BasicDataType::Trace =>
                helper!(Trace, rsprocess::trace::SlicingTrace::default()),
            | BasicDataType::PositiveTrace => helper!(
                PositiveTrace,
                rsprocess::trace::SlicingTrace::default()
            ),
            | BasicDataType::PositiveSet =>
                helper!(PositiveSet, rsprocess::set::PositiveSet::default()),
            | BasicDataType::PositiveEnvironment =>
                helper!(PositiveEnvironment, rsprocess::environment::PositiveEnvironment::default()),
            | BasicDataType::PositiveContext =>
                helper!(PositiveContext, rsprocess::process::PositiveProcess::default()),
            | BasicDataType::PositiveReactions =>
                helper!(PositiveReactions, vec![]),
        }
    }

    #[allow(clippy::type_complexity)]
    pub(crate) fn create_output(
        ty: BasicDataType,
    ) -> Box<dyn Fn(NodeId, &mut NodeGraph, &str)> {
        macro_rules! helper {
            ($name: ident) => {
                Box::new(
                    |node_id: NodeId, graph: &mut NodeGraph, name: &str| {
                        graph.add_output_param(
                            node_id,
                            name.to_string(),
                            BasicDataType::$name,
                        );
                    },
                )
            };
        }

        match ty {
            | BasicDataType::Error =>
                Box::new(|_: NodeId, _: &mut NodeGraph, _: &str| {}),

            | BasicDataType::Path => helper!(Path),
            | BasicDataType::String => helper!(String),
            | BasicDataType::System => helper!(System),
            | BasicDataType::PositiveInt => helper!(PositiveInt),
            | BasicDataType::Symbol => helper!(Symbol),
            | BasicDataType::Experiment => helper!(Experiment),
            | BasicDataType::Graph => helper!(Graph),
            | BasicDataType::GroupingFunction => helper!(GroupingFunction),
            | BasicDataType::DisplayNode => helper!(DisplayNode),
            | BasicDataType::DisplayEdge => helper!(DisplayEdge),
            | BasicDataType::ColorNode => helper!(ColorNode),
            | BasicDataType::ColorEdge => helper!(ColorEdge),
            | BasicDataType::Environment => helper!(Environment),
            | BasicDataType::Set => helper!(Set),
            | BasicDataType::Context => helper!(Context),
            | BasicDataType::Reactions => helper!(Reactions),
            | BasicDataType::PositiveSystem => helper!(PositiveSystem),
            | BasicDataType::Trace => helper!(Trace),
            | BasicDataType::PositiveTrace => helper!(PositiveTrace),
            | BasicDataType::PositiveSet => helper!(PositiveSet),
            | BasicDataType::PositiveEnvironment => helper!(PositiveEnvironment),
            | BasicDataType::PositiveContext => helper!(PositiveContext),
            | BasicDataType::PositiveReactions => helper!(PositiveReactions),
        }
    }
}

/// Additional messages generated and passed by our code (not already present
/// in the graph library)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CustomResponse {
    SetActiveNode(NodeId),
    ClearActiveNode,
    SaveToFile(NodeId),
    FieldModified(NodeId),
}

#[derive(Default, Debug)]
#[cfg_attr(
    feature = "persistence",
    derive(serde::Serialize, serde::Deserialize)
)]
struct CacheInternals {
    values:      HashMap<OutputId, BasicValue>,
    hash_values: HashMap<OutputId, u64>,
    hash_inputs: HashMap<OutputId, (u64, Vec<u64>)>,
    last_output: Option<LayoutJob>,
}

/// Cache used to save intermediate values between executions.
/// holds the output value of the outgoing edge, the hash of the value and the
/// hash of the inputs that generated the output.
#[derive(Default, Debug)]
#[cfg_attr(
    feature = "persistence",
    derive(serde::Serialize, serde::Deserialize)
)]
pub(crate) struct OutputsCache {
    internals: Arc<RwLock<CacheInternals>>,
}

impl OutputsCache {
    pub(crate) fn calculate_hash<T: std::hash::Hash>(t: &T) -> u64 {
        use std::hash::Hasher;
        let mut s = std::hash::DefaultHasher::new();
        t.hash(&mut s);
        s.finish()
    }

    pub fn input_hashes(&self, key: &OutputId) -> Option<Vec<u64>> {
        let internals = self.internals.read().unwrap();
        internals.hash_inputs.get(key).map(|el| el.1.to_vec())
    }

    pub fn same_hash_inputs(&self, key: &OutputId, inputs: &[u64]) -> bool {
        let hash_inputs = inputs.iter().fold(0, |acc, x| acc ^ x);
        let internals = self.internals.read().unwrap();
        internals
            .hash_inputs
            .get(key)
            .map(|val| hash_inputs == val.0)
            .is_some_and(|x| x)
    }

    fn associate(&self, key: OutputId, value: BasicValue, inputs: Vec<u64>) {
        let mut internals = self.internals.write().unwrap();
        let hash = Self::calculate_hash(&value);
        internals.values.insert(key, value);
        internals.hash_values.insert(key, hash);
        let hash_inputs = inputs.iter().fold(0, |acc, x| acc ^ x);
        internals.hash_inputs.insert(key, (hash_inputs, inputs));
    }

    pub fn retrieve_cache_output(
        &self,
        graph: &NodeGraph,
        node_id: NodeId,
        param_name: &str,
    ) -> anyhow::Result<BasicValue> {
        let input_id = graph[node_id].get_input(param_name)?;
        if let Some(other_output_id) = graph.connection(input_id) {
            let internals = self.internals.read().unwrap();

            if let Some(other_value) = internals.values.get(&other_output_id) {
                Ok(other_value.clone())
            } else {
                anyhow::bail!("Value not in cache")
            }
        } else {
            Ok(graph[input_id].value.clone())
        }
    }

    pub fn retrieve_output(&self, key: OutputId) -> Option<BasicValue> {
        let internals = self.internals.read().unwrap();
        internals.values.get(&key).cloned()
    }

    pub fn populate_output(
        &self,
        graph: &NodeGraph,
        node_id: NodeId,
        param_name: &str,
        value: BasicValue,
        hash_inputs: Vec<u64>,
    ) -> anyhow::Result<()> {
        let output_id = graph[node_id].get_output(param_name)?;
        self.associate(output_id, value, hash_inputs);
        Ok(())
    }

    pub fn invalidate_cache(&self, key: &OutputId) {
        let mut internals = self.internals.write().unwrap();
        internals.hash_inputs.remove(key);
        internals.hash_values.remove(key);
        internals.values.remove(key);
    }

    pub fn invalidate_outputs(&self, graph: &NodeGraph, node_id: NodeId) {
        for output_id in graph[node_id].output_ids() {
            self.invalidate_cache(&output_id);
        }
    }

    #[allow(dead_code)]
    pub fn reset_cache(&mut self) {
        let mut internals = self.internals.write().unwrap();
        *internals = CacheInternals::default();
    }

    pub fn get_last_state(&self) -> Option<LayoutJob> {
        let internals = self.internals.read().unwrap();
        internals.last_output.clone()
    }

    pub fn invalidate_last_state(&mut self) {
        let mut internals = self.internals.write().unwrap();
        internals.last_output = None;
    }

    pub fn set_last_state(&mut self, val: LayoutJob) {
        let mut internals = self.internals.write().unwrap();
        internals.last_output = Some(val);
    }
}

/// The graph 'global' state.
#[derive(Default)]
#[cfg_attr(
    feature = "persistence",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct GlobalState {
    pub active_node:    Option<NodeId>,
    pub save_node:      Option<NodeId>,
    pub display_result: bool,
    pub translator:     rsprocess::translator::Translator,

    pub cache: OutputsCache,
}

// Display instructions for each of the data types
impl DataTypeTrait<GlobalState> for BasicDataType {
    fn data_type_color(&self, _user_state: &mut GlobalState) -> egui::Color32 {
        match self {
            | Self::Error => egui::Color32::RED,

            | Self::String => egui::Color32::from_rgb(38, 109, 211),
            | Self::Path => egui::Color32::from_rgb(109, 211, 38),
            | Self::System => egui::Color32::from_rgb(238, 207, 109),
            | Self::PositiveInt => egui::Color32::BLUE,
            | Self::Symbol => egui::Color32::YELLOW,
            | Self::Experiment => egui::Color32::GRAY,
            | Self::Graph => egui::Color32::DARK_GREEN,
            | Self::GroupingFunction => egui::Color32::DARK_GRAY,
            | Self::DisplayNode => egui::Color32::from_rgb(46, 139, 87),
            | Self::DisplayEdge => egui::Color32::from_rgb(67, 205, 128),
            | Self::ColorNode => egui::Color32::from_rgb(78, 238, 148),
            | Self::ColorEdge => egui::Color32::from_rgb(84, 255, 159),
            | Self::Environment => egui::Color32::from_rgb(221, 160, 221),
            | Self::Set => egui::Color32::from_rgb(255, 0, 255),
            | Self::Context => egui::Color32::from_rgb(238, 130, 238),
            | Self::Reactions => egui::Color32::from_rgb(218, 112, 214),
            | Self::PositiveSystem => egui::Color32::from_rgb(238, 109, 153),
            | Self::Trace => egui::Color32::from_rgb(178, 34, 34),
            | Self::PositiveTrace => egui::Color32::from_rgb(178, 54, 54),
            | Self::PositiveSet => egui::Color32::from_rgb(255, 30, 255),
            | Self::PositiveEnvironment => egui::Color32::from_rgb(10, 20, 50),
            | Self::PositiveContext => egui::Color32::from_rgb(20, 10, 50),
            | Self::PositiveReactions => egui::Color32::from_rgb(50, 10, 20),
        }
    }

    fn name(&self) -> Cow<'_, str> {
        match self {
            | Self::Error => Cow::Borrowed("error"),

            | Self::String => Cow::Borrowed("string"),
            | Self::Path => Cow::Borrowed("path"),
            | Self::System => Cow::Borrowed("system"),
            | Self::PositiveInt => Cow::Borrowed("integer"),
            | Self::Symbol => Cow::Borrowed("symbol"),
            | Self::Experiment => Cow::Borrowed("experiment"),
            | Self::Graph => Cow::Borrowed("graph"),
            | Self::GroupingFunction => Cow::Borrowed("grouping function"),
            | Self::DisplayNode => Cow::Borrowed("display node"),
            | Self::DisplayEdge => Cow::Borrowed("display edge"),
            | Self::ColorNode => Cow::Borrowed("color node"),
            | Self::ColorEdge => Cow::Borrowed("color edge"),
            | Self::Environment => Cow::Borrowed("environment"),
            | Self::Set => Cow::Borrowed("set"),
            | Self::Context => Cow::Borrowed("context"),
            | Self::Reactions => Cow::Borrowed("reactions"),
            | Self::PositiveSystem => Cow::Borrowed("positive system"),
            | Self::Trace => Cow::Borrowed("trace"),
            | Self::PositiveTrace => Cow::Borrowed("positive trace"),
            | Self::PositiveSet => Cow::Borrowed("positive set"),
            | Self::PositiveEnvironment => Cow::Borrowed("positive environment"),
            | Self::PositiveContext => Cow::Borrowed("positive context"),
            | Self::PositiveReactions => Cow::Borrowed("positive reactions"),
        }
    }
}

// New node display information
impl NodeTemplateTrait for NodeInstruction {
    type NodeData = NodeData;
    type DataType = BasicDataType;
    type ValueType = BasicValue;
    type UserState = GlobalState;
    type CategoryType = &'static str;

    fn node_finder_label(
        &self,
        _user_state: &mut Self::UserState,
    ) -> Cow<'_, str> {
        Cow::Borrowed(match self {
            | Self::String => "String",
            | Self::Path => "Path",
            | Self::ReadPath => "Read a file",
            | Self::System => "Create a System",
            | Self::Statistics => "Statistics",
            | Self::Target => "Target",
            | Self::Run => "Run",
            | Self::Loop => "Loop",
            | Self::Symbol => "Symbol",
            | Self::Frequency => "Frequency",
            | Self::LimitFrequency => "Limit Frequency",
            | Self::Experiment => "Experiment",
            | Self::FastFrequency => "Fast Frequency",
            | Self::BisimilarityKanellakisSmolka =>
                "Bisimilarity Kanellakis & Smolka",
            | Self::BisimilarityPaigeTarjanNoLabels =>
                "Bisimilarity Paige & Tarjan (ignore labels)",
            | Self::BisimilarityPaigeTarjan => "Bisimilarity Paige & Tarjan",
            | Self::GroupFunction => "Group Function for Graphs",
            | Self::SystemGraph => "Graph of a System",
            | Self::SaveString => "Save string to file",
            | Self::Dot => "Create Dot file",
            | Self::DisplayNode => "Display node function",
            | Self::DisplayEdge => "Display edge function",
            | Self::ColorNode => "Color node function",
            | Self::ColorEdge => "Color edge function",
            | Self::GraphML => "Create GraphML file",
            | Self::ComposeSystem => "Compose a System",
            | Self::Environment => "Environment",
            | Self::Set => "Set",
            | Self::Context => "Context",
            | Self::Reactions => "Reactions",
            | Self::PositiveSystem => "Positive System",
            | Self::PositiveTarget => "Target",
            | Self::PositiveRun => "Run",
            | Self::PositiveLoop => "Loop",
            | Self::PositiveFrequency => "Frequency",
            | Self::PositiveLimitFrequency => "Limit Frequency",
            | Self::PositiveFastFrequency => "Fast Frequency",
            | Self::Trace => "Trace",
            | Self::PositiveTrace => "Positive Trace",
            | Self::SliceTrace => "Slice Trace",
            | Self::PositiveSliceTrace => "Positive Slice Trace",
            | Self::PositiveSet => "Positive Set",
            | Self::ToPositiveSet => "Convert to Positive Set",
            | Self::ComposePositiveSystem => "Compose a positive system",
            | Self::DecomposeSystem => "Decompose a system",
            | Self::DecomposePositiveSystem  => "Decompose a positive system",
        })
    }

    // Groups in the new node dropdown
    fn node_finder_categories(
        &self,
        _user_state: &mut Self::UserState,
    ) -> Vec<&'static str> {
        match self {
            | Self::String
            | Self::Path
            | Self::ReadPath
            | Self::Symbol
            | Self::SaveString => vec!["String"],
            | Self::System
            | Self::Statistics
            | Self::Target
            | Self::Run
            | Self::Loop
            | Self::ComposeSystem
            | Self::Environment
            | Self::Set
            | Self::Context
            | Self::Reactions
            | Self::DecomposeSystem => vec!["System"],
            | Self::Frequency
            | Self::LimitFrequency
            | Self::Experiment
            | Self::FastFrequency => vec!["System", "Frequency"],
            | Self::BisimilarityKanellakisSmolka
            | Self::BisimilarityPaigeTarjanNoLabels
            | Self::BisimilarityPaigeTarjan
            | Self::GroupFunction => vec!["System", "Bisimilarity"],
            | Self::SystemGraph => vec!["System", "Graph"],
            | Self::Dot
            | Self::DisplayNode
            | Self::DisplayEdge
            | Self::ColorNode
            | Self::ColorEdge
            | Self::GraphML => vec!["Graph"],
            | Self::PositiveSystem
            | Self::PositiveTarget
            | Self::PositiveRun
            | Self::PositiveLoop
            | Self::PositiveFrequency
            | Self::PositiveLimitFrequency
            | Self::PositiveFastFrequency
            | Self::PositiveSet
            | Self::ToPositiveSet
            | Self::ComposePositiveSystem
            | Self::DecomposePositiveSystem => vec!["Positive System"],
            | Self::Trace => vec!["Trace", "System"],
            | Self::PositiveTrace => vec!["Trace", "Positive System"],
            | Self::SliceTrace | Self::PositiveSliceTrace => vec!["Trace"],
        }
    }

    fn node_graph_label(&self, user_state: &mut Self::UserState) -> String {
        self.node_finder_label(user_state).into()
    }

    fn user_data(&self, _user_state: &mut Self::UserState) -> Self::NodeData {
        NodeData { template: *self }
    }

    fn build_node(
        &self,
        graph: &mut Graph<Self::NodeData, Self::DataType, Self::ValueType>,
        _user_state: &mut Self::UserState,
        node_id: NodeId,
    ) {
        for (i, data) in self.inputs() {
            Self::create_input(data)(node_id, graph, &i);
        }
        for (o, data) in self.output() {
            Self::create_output(data)(node_id, graph, &o);
        }
    }
}

pub struct AllInstructions;
impl NodeTemplateIter for AllInstructions {
    type Item = NodeInstruction;

    fn all_kinds(&self) -> Vec<Self::Item> {
        // ---------------------------------------------------------------------
        // -------------------- Add here additional types ----------------------
        // ---------------------------------------------------------------------
        vec![
            NodeInstruction::String,
            NodeInstruction::Path,
            NodeInstruction::ReadPath,
            NodeInstruction::System,
            NodeInstruction::Statistics,
            NodeInstruction::Target,
            NodeInstruction::Run,
            NodeInstruction::Loop,
            NodeInstruction::Symbol,
            NodeInstruction::Frequency,
            NodeInstruction::LimitFrequency,
            NodeInstruction::Experiment,
            NodeInstruction::FastFrequency,
            NodeInstruction::BisimilarityKanellakisSmolka,
            NodeInstruction::BisimilarityPaigeTarjanNoLabels,
            NodeInstruction::BisimilarityPaigeTarjan,
            NodeInstruction::GroupFunction,
            NodeInstruction::SystemGraph,
            NodeInstruction::SaveString,
            NodeInstruction::Dot,
            NodeInstruction::DisplayNode,
            NodeInstruction::DisplayEdge,
            NodeInstruction::ColorNode,
            NodeInstruction::ColorEdge,
            NodeInstruction::GraphML,
            NodeInstruction::ComposeSystem,
            NodeInstruction::Environment,
            NodeInstruction::Set,
            NodeInstruction::Context,
            NodeInstruction::Reactions,
            NodeInstruction::PositiveSystem,
            NodeInstruction::PositiveTarget,
            NodeInstruction::PositiveRun,
            NodeInstruction::PositiveLoop,
            NodeInstruction::PositiveFrequency,
            NodeInstruction::PositiveLimitFrequency,
            NodeInstruction::PositiveFastFrequency,
            NodeInstruction::Trace,
            NodeInstruction::PositiveTrace,
            NodeInstruction::SliceTrace,
            NodeInstruction::PositiveSliceTrace,
            NodeInstruction::PositiveSet,
            NodeInstruction::ToPositiveSet,
            NodeInstruction::ComposePositiveSystem,
            NodeInstruction::DecomposeSystem,
            NodeInstruction::DecomposePositiveSystem,
        ]
    }
}

/// Describes what ui to diplay for each input types.
impl WidgetValueTrait for BasicValue {
    type Response = CustomResponse;
    type UserState = GlobalState;
    type NodeData = NodeData;

    fn value_widget(
        &mut self,
        param_name: &str,
        node_id: NodeId,
        ui: &mut egui::Ui,
        _user_state: &mut GlobalState,
        _node_data: &NodeData,
    ) -> Vec<CustomResponse> {
        let mut responses = vec![];

        match self {
            // Dummy values used to save files, no ui since not needed
            | BasicValue::SaveString { path: _, value: _ } => {},
            | BasicValue::Error { value: _ } => {},

            | BasicValue::String { value } => {
                ui.label(param_name);
                ui.horizontal(|ui| {
                    let field = ui.add(
                        egui::TextEdit::multiline(value)
                            .hint_text("String here")
                            .clip_text(false),
                    );
                    if field.changed() {
                        responses.push(CustomResponse::FieldModified(node_id));
                    }
                });
            },
            | BasicValue::Path { value } => {
                ui.label(param_name);
                ui.horizontal(|ui| {
                    let field = ui.add(
                        egui::TextEdit::multiline(value)
                            .hint_text("Path here")
                            .clip_text(false),
                    );
                    if field.changed() {
                        responses.push(CustomResponse::FieldModified(node_id));
                    }
                });
            },
            | BasicValue::System { value: _ } => {
                ui.label(param_name);
            },
            | BasicValue::PositiveInt { value } => {
                ui.add(egui::DragValue::new(value));
            },
            | BasicValue::Symbol { value } => {
                ui.label(param_name);
                ui.horizontal(|ui| {
                    let field = ui.add(
                        egui::TextEdit::singleline(value)
                            .hint_text("Symbol here")
                            .clip_text(false),
                    );
                    if field.changed() {
                        responses.push(CustomResponse::FieldModified(node_id));
                    }
                });
            },
            | BasicValue::Experiment { value: _ } => {
                ui.label(param_name);
            },
            | BasicValue::Graph { value: _ } => {
                ui.label(param_name);
            },
            | BasicValue::GroupingFunction { value: _ } => {
                ui.label(param_name);
            },
            | BasicValue::DisplayNode { value: _ } => {
                ui.label(param_name);
            },
            | BasicValue::DisplayEdge { value: _ } => {
                ui.label(param_name);
            },
            | BasicValue::ColorNode { value: _ } => {
                ui.label(param_name);
            },
            | BasicValue::ColorEdge { value: _ } => {
                ui.label(param_name);
            },
            | BasicValue::Environment { value: _ } => {
                ui.label(param_name);
            },
            | BasicValue::Set { value: _ } => {
                ui.label(param_name);
            },
            | BasicValue::Context { value: _ } => {
                ui.label(param_name);
            },
            | BasicValue::Reactions { value: _ } => {
                ui.label(param_name);
            },
            | BasicValue::PositiveSystem { value: _ } => {
                ui.label(param_name);
            },
            | BasicValue::Trace { value: _ } => {
                ui.label(param_name);
            },
            | BasicValue::PositiveTrace { value: _ } => {
                ui.label(param_name);
            },
            | BasicValue::PositiveSet { value: _ } => {
                ui.label(param_name);
            },
            | BasicValue::PositiveEnvironment { value: _ } => {
                ui.label(param_name);
            },
            | BasicValue::PositiveContext { value: _ } => {
                ui.label(param_name);
            },
            | BasicValue::PositiveReactions { value: _ } => {
                ui.label(param_name);
            },
        }

        responses
    }
}

impl UserResponseTrait for CustomResponse {}
impl NodeDataTrait for NodeData {
    type Response = CustomResponse;
    type UserState = GlobalState;
    type DataType = BasicDataType;
    type ValueType = BasicValue;

    // Bottom part of each node. Used here for Active nodes and for Write in
    // WriteString nodes.
    fn bottom_ui(
        &self,
        ui: &mut egui::Ui,
        node_id: NodeId,
        graph: &Graph<NodeData, BasicDataType, BasicValue>,
        user_state: &mut Self::UserState,
    ) -> Vec<NodeResponse<CustomResponse, NodeData>>
    where
        CustomResponse: UserResponseTrait,
    {
        let mut responses = vec![];
        let is_active = user_state
            .active_node
            .map(|id| id == node_id)
            .unwrap_or(false);

        match (is_active, graph[node_id].user_data.template) {
            | (_, NodeInstruction::SaveString) => {
                if ui.button("Write").clicked() {
                    responses.push(NodeResponse::User(
                        CustomResponse::SaveToFile(node_id),
                    ));
                }
            },
            | (_, ni) if ni.output().len() > 1 => {
                // no button for nodes with more than one output
            }
            | (true, _) => {
                let button = egui::Button::new(
                    egui::RichText::new("👁 Active").color(egui::Color32::BLACK),
                )
                .fill(egui::Color32::GOLD);
                if ui.add(button).clicked() {
                    responses.push(NodeResponse::User(
                        CustomResponse::ClearActiveNode,
                    ));
                }
            },
            | (false, _) =>
                if ui.button("👁 Set active").clicked() {
                    responses.push(NodeResponse::User(
                        CustomResponse::SetActiveNode(node_id),
                    ));
                },
        }

        responses
    }
}

type NodeGraph = Graph<NodeData, BasicDataType, BasicValue>;

type EditorState = GraphEditorState<
    NodeData,
    BasicDataType,
    BasicValue,
    NodeInstruction,
    GlobalState,
>;

#[derive(Default)]
pub struct AppHandle {
    // The `GraphEditorState` is the top-level object. You "register" all your
    // custom types by specifying it as its generic parameters.
    state: EditorState,

    user_state: GlobalState,
}

#[cfg(feature = "persistence")]
const PERSISTENCE_KEY: &str = "egui_node_graph";

#[cfg(feature = "persistence")]
impl AppHandle {
    /// If the persistence feature is enabled, Called once before the first
    /// frame. Load previous app state (if any).
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let state = cc
            .storage
            .and_then(|storage| eframe::get_value(storage, PERSISTENCE_KEY))
            .unwrap_or_default();
        Self {
            state,
            user_state: GlobalState::default(),
        }
    }
}

/// Main endpoint to be executed
impl eframe::App for AppHandle {
    #[cfg(feature = "persistence")]
    /// If the persistence function is enabled,
    /// Called by the frame work to save state before shutdown.
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, PERSISTENCE_KEY, &self.state);
    }

    /// Called each time the UI needs repainting, which may be many times per
    /// second. Put your widgets into a `SidePanel`, `TopPanel`,
    /// `CentralPanel`, `Window` or `Area`.
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            egui::MenuBar::new().ui(ui, |ui| {
                egui::widgets::global_theme_preference_switch(ui);

                // can only show file selection on native and with persistence
                // enabled
                #[cfg(feature = "persistence")]
                #[cfg(not(target_arch = "wasm32"))]
                {
                    use eframe::egui::{PopupCloseBehavior, RectAlign};

                    let response = egui::Frame::group(ui.style())
                        .show(ui, |ui| {
                            ui.set_max_height(20.);
                            ui.set_max_width(30.);
                            ui.vertical_centered(|ui| ui.button("File")).inner
                        })
                        .inner;
                    egui::Popup::menu(&response)
                        .align(RectAlign::BOTTOM_END)
                        .gap(4.)
                        .close_behavior(PopupCloseBehavior::CloseOnClickOutside)
                        .id(egui::Id::new("file"))
                        .show(|ui| {
                            if ui.button("New File").clicked() {
                                let new_state = EditorState::default();
                                eframe::set_value(
                                    _frame
                                        .storage_mut()
                                        .expect("no storage found"),
                                    PERSISTENCE_KEY,
                                    &new_state,
                                );
                                self.state = new_state;
                                ui.close();
                            }
                            if ui.button("Open File…").clicked()
                                && let Some(path) = rfd::FileDialog::new()
                                    .add_filter("ron", &["ron"])
                                    .pick_file()
                            {
                                if let Ok(s) = std::fs::read_to_string(path) {
                                    match ron::from_str::<EditorState>(&s) {
                                        | Ok(state) => {
                                            eframe::set_value(
                                                _frame
                                                    .storage_mut()
                                                    .expect("no storage found"),
                                                PERSISTENCE_KEY,
                                                &state,
                                            );
                                            self.state = state;
                                        },
                                        | Err(err) => {
                                            println!("error {err:?}");
                                        },
                                    }
                                }
                                ui.close();
                            }
                            if ui.button("Save as…").clicked()
                                && let Some(path) = rfd::FileDialog::new()
                                    .add_filter("ron", &["ron"])
                                    .save_file()
                            {
                                let value =
                                    match ron::ser::to_string(&self.state) {
                                        | Ok(value) => value,
                                        | Err(e) => {
                                            println!("error {e}");
                                            panic!()
                                        },
                                    };
                                match std::fs::write(path, value) {
                                    | Ok(_) => {},
                                    | Err(e) => {
                                        println!("error saving {e:?}")
                                    },
                                }

                                ui.close();
                            }
                        });
                }

                #[cfg(debug_assertions)]
                {
                    use eframe::egui::{PopupCloseBehavior, RectAlign};

                    let response = egui::Frame::group(ui.style())
                        .show(ui, |ui| {
                            ui.set_max_height(20.);
                            ui.set_max_width(40.);
                            ui.vertical_centered(|ui| ui.button("Cache")).inner
                        })
                        .inner;
                    egui::Popup::menu(&response)
                        .align(RectAlign::BOTTOM_END)
                        .gap(4.)
                        .close_behavior(PopupCloseBehavior::CloseOnClickOutside)
                        .id(egui::Id::new("cache"))
                        .show(|ui| {
                            if ui.button("Clear").clicked() {
                                self.user_state.cache.reset_cache();
                                ui.close();
                            }
                        });
                }
            });
        });

        let graph_response = egui::CentralPanel::default()
            .show(ctx, |ui| {
                self.state.draw_graph_editor(
                    ui,
                    AllInstructions,
                    &mut self.user_state,
                    Vec::default(),
                )
            })
            .inner;

        for node_response in graph_response.node_responses.iter() {
            // graph events
            match node_response {
                | NodeResponse::User(CustomResponse::SetActiveNode(node)) => {
                    self.user_state.active_node = Some(*node);
                    self.user_state.display_result = true;
                    self.user_state.cache.invalidate_last_state();
                },
                | NodeResponse::User(CustomResponse::ClearActiveNode) => {
                    self.user_state.active_node = None;
                    self.user_state.display_result = false;
                    self.user_state.cache.invalidate_last_state();
                },
                | NodeResponse::User(CustomResponse::SaveToFile(node)) => {
                    self.user_state.save_node = Some(*node);
                    self.user_state.display_result = true;
                    self.user_state.cache.invalidate_last_state();
                },
                | NodeResponse::User(CustomResponse::FieldModified(node)) => {
                    self.user_state.cache.invalidate_last_state();
                    self.user_state.cache.invalidate_outputs(&self.state.graph, *node);
                }
                | NodeResponse::DisconnectEvent { output, input: _ } => {
                    self.user_state.cache.invalidate_cache(output);
                },
                | NodeResponse::ConnectEventEnded {
                    output,
                    input: _,
                    input_hook: _,
                } => {
                    self.user_state.cache.invalidate_cache(output);
                },
                | _ => {},
            }
        }

        if self.user_state.display_result {
            let text = {
                if !graph_response.node_responses.is_empty() {
                    let computed_output = create_output(self, ctx);
                    self.user_state
                        .cache
                        .set_last_state(computed_output.clone());
                    computed_output
                } else if let Some(pre_computed) =
                    self.user_state.cache.get_last_state()
                {
                    pre_computed
                } else {
                    let computed_output = create_output(self, ctx);
                    self.user_state
                        .cache
                        .set_last_state(computed_output.clone());
                    computed_output
                }
            };

            let window = egui::SidePanel::right("Results").resizable(true);

            window.show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.vertical_centered(|ui| {
                        ui.heading("Results");
                    });
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        ui.label(text);
                    });
                });
            });
        }
    }
}

fn create_output(ng: &mut AppHandle, ctx: &egui::Context) -> LayoutJob {
    let mut text = LayoutJob::default();

    match (ng.user_state.save_node, ng.user_state.active_node) {
        | (Some(node), _) if ng.state.graph.nodes.contains_key(node) => {
            let value = crate::app_logic::evaluate_node(
                &ng.state.graph,
                node,
                &ng.user_state.cache,
                &mut ng.user_state.translator,
                ctx,
            );
            ng.user_state.save_node = None;
            match value {
                | Ok(BasicValue::SaveString { path, value }) => {
                    match std::fs::write(&path, value) {
                        | Ok(_) => {
                            // TODO: this only appears for one frame
                            text.append(
                                &format!("Wrote file {}.", path),
                                0.,
                                TextFormat {
                                    ..Default::default()
                                },
                            );
                        },
                        | Err(e) => {
                            // TODO: this only appears for one frame
                            text.append(&format!("{e}"), 0., TextFormat {
                                ..Default::default()
                            });
                        },
                    }
                },
                | Err(_) => {
                    text = get_layout(value, &ng.user_state.translator, ctx);
                },
                | Ok(_) => {
                    text = get_layout(value, &ng.user_state.translator, ctx);
                    {
                        // prepend doesnt exist for layoutjob
                        let new_text = "Could not save invalid value:";
                        let start = 0;
                        text.text.insert_str(0, new_text);
                        let byte_range = start..new_text.len();
                        text.sections.insert(0, LayoutSection {
                            leading_space: 0.,
                            byte_range,
                            format: TextFormat {
                                color: Color32::RED,
                                ..Default::default()
                            },
                        });
                    }
                },
            }
        },
        | (None, Some(node)) if ng.state.graph.nodes.contains_key(node) => {
            text = get_layout(
                crate::app_logic::evaluate_node(
                    &ng.state.graph,
                    node,
                    &ng.user_state.cache,
                    &mut ng.user_state.translator,
                    ctx,
                ),
                &ng.user_state.translator,
                ctx,
            );
        },
        | (None, None) => {
            ng.user_state.display_result = false;
        },
        | (_, _) => {
            ng.user_state.active_node = None;
            ng.user_state.save_node = None;
            ng.user_state.display_result = false;
        },
    }

    text
}

fn get_layout(
    value: anyhow::Result<BasicValue>,
    translator: &rsprocess::translator::Translator,
    ctx: &egui::Context,
) -> LayoutJob {
    let mut text = LayoutJob::default();

    match value {
        | Ok(value) => match value {
            | BasicValue::SaveString { path, value: _ } => text.append(
                &format!("Saving to file \"{}\"", path),
                0.,
                Default::default(),
            ),
            | BasicValue::Error { value } => {
                text = value;
            },

            | BasicValue::Path { value } =>
                text.append(&value, 0., Default::default()),
            | BasicValue::String { value } =>
                text.append(&value, 0., Default::default()),
            | BasicValue::System { value } => text.append(
                &format!("{}", Formatter::from(translator, &value)),
                0.,
                Default::default(),
            ),
            | BasicValue::PositiveInt { value } =>
                text.append(&format!("{value}"), 0., Default::default()),
            | BasicValue::Symbol { value } =>
                text.append(&value, 0., Default::default()),
            | BasicValue::Experiment { value } => {
                for (weight, set) in value.0.iter().zip(value.1.iter()) {
                    text.append(
                        &format!(
                            "weight {} for set {}\n",
                            weight,
                            Formatter::from(translator, set)
                        ),
                        0.,
                        Default::default(),
                    )
                }
            },
            | BasicValue::Graph { value } => text.append(
                &format!(
                    "A graph with {} nodes and {} edges.",
                    value.node_count(),
                    value.edge_count()
                ),
                0.,
                Default::default(),
            ),
            | BasicValue::GroupingFunction { value } => text.append(
                &format!("{}", Formatter::from(translator, &value)),
                0.,
                Default::default(),
            ),
            | BasicValue::DisplayNode { value } =>
                text.append(&format!("{value:?}"), 0., TextFormat {
                    ..Default::default()
                }),
            | BasicValue::DisplayEdge { value } =>
                text.append(&format!("{value:?}"), 0., Default::default()),
            | BasicValue::ColorNode { value } =>
                text.append(&format!("{value:?}"), 0., Default::default()),
            | BasicValue::ColorEdge { value } =>
                text.append(&format!("{value:?}"), 0., Default::default()),
            | BasicValue::Environment { value } => text.append(
                &format!("{}", Formatter::from(translator, &value)),
                0.,
                Default::default(),
            ),
            | BasicValue::Set { value } => text.append(
                &format!("{}", Formatter::from(translator, &value)),
                0.,
                Default::default(),
            ),
            | BasicValue::Context { value } => text.append(
                &format!("{}", Formatter::from(translator, &value)),
                0.,
                Default::default(),
            ),
            | BasicValue::Reactions { value } => {
                text.append("(", 0., TextFormat {
                    ..Default::default()
                });
                let mut i = value.iter().peekable();
                while let Some(r) = i.next() {
                    if i.peek().is_some() {
                        text.append(
                            &format!("{}, ", Formatter::from(translator, r)),
                            0.,
                            Default::default(),
                        );
                    } else {
                        text.append(
                            &format!("{}", Formatter::from(translator, r)),
                            0.,
                            Default::default(),
                        );
                    }
                }
                text.append(")", 0., Default::default());
            },
            | BasicValue::PositiveSystem { value } => text.append(
                &format!("{}", Formatter::from(translator, &value)),
                0.,
                Default::default(),
            ),
            | BasicValue::Trace { value } => text.append(
                &format!("{}", Formatter::from(translator, &value)),
                0.,
                TextFormat {
                    font_id: eframe::egui::TextStyle::Monospace
                        .resolve(&ctx.style()),
                    ..Default::default()
                },
            ),
            | BasicValue::PositiveTrace { value } => text.append(
                &format!("{}", Formatter::from(translator, &value)),
                0.,
                TextFormat {
                    font_id: eframe::egui::TextStyle::Monospace
                        .resolve(&ctx.style()),
                    ..Default::default()
                },
            ),
            | BasicValue::PositiveSet { value } => text.append(
                &format!("{}", Formatter::from(translator, &value)),
                0., Default::default(),
            ),
            | BasicValue::PositiveEnvironment { value } => text.append(
                &format!("{}", Formatter::from(translator, &value)),
                0., Default::default(),
            ),
            | BasicValue::PositiveContext { value } => text.append(
                &format!("{}", Formatter::from(translator, &value)),
                0., Default::default(),
            ),
            | BasicValue::PositiveReactions { value } => {
                text.append("(", 0., TextFormat {
                    ..Default::default()
                });
                let mut i = value.iter().peekable();
                while let Some(r) = i.next() {
                    if i.peek().is_some() {
                        text.append(
                            &format!("{}, ", Formatter::from(translator, r)),
                            0.,
                            Default::default(),
                        );
                    } else {
                        text.append(
                            &format!("{}", Formatter::from(translator, r)),
                            0.,
                            Default::default(),
                        );
                    }
                }
                text.append(")", 0., Default::default());
            }
        },
        | Err(err) => {
            text.append(&format!("{err:?}"), 0., TextFormat {
                color: Color32::RED,
                ..Default::default()
            });
        },
    }
    text
}
