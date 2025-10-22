use std::collections::{HashSet, VecDeque};
use std::rc::Rc;

use egui_node_graph2::*;
use rsprocess::frequency::BasicFrequency;
use rsprocess::system::{ExtensionsSystem, LoopSystem};
use rsprocess::translator::Formatter;

use crate::app::{
    BasicDataType, BasicValue, NodeData, NodeInstruction, OutputsCache,
};
use crate::helper;

type MyGraph = Graph<NodeData, BasicDataType, BasicValue>;

/// Recursively evaluates all dependencies of this node, then evaluates the node
/// itself. Except we use a queue so we dont pollute the stack.
pub fn evaluate_node(
    graph: &MyGraph,
    node_id: NodeId,
    outputs_cache: &OutputsCache,
    translator: &mut rsprocess::translator::Translator,
    ctx: &eframe::egui::Context,
) -> anyhow::Result<BasicValue> {
    // generates list of nodes to evaluate and invalidates cache of those nodes
    let to_evaluate = generate_to_evaluate(graph, outputs_cache, node_id)?;

    let mut to_ret = None;

    #[cfg(debug_assertions)]
    {
        if !to_evaluate.is_empty() {
            println!("evaluating nodes {:?}", to_evaluate);
        }
    }

    // for each node to evaluate (in the correct order) finds the output and
    // populates the cache
    for node_id in to_evaluate {
        let node = &graph[node_id];
        let output_name = graph[node_id]
            .user_data
            .template
            .output()
            .unwrap_or(("".into(), BasicDataType::Error))
            .0;

        match process_template(
            graph,
            node_id,
            outputs_cache,
            &node.user_data.template,
            &output_name,
            translator,
            &mut to_ret,
            ctx,
        )? {
            | None => {},
            | Some(val) => return Ok(val),
        }
    }

    if let Some(res) = to_ret.take() {
        Ok(res)
    } else {
        let output_field = graph[node_id]
            .user_data
            .template
            .output()
            .map(|el| el.0)
            .unwrap_or("".into());
        let output_id = graph[node_id].get_output(&output_field)?;

        Ok(outputs_cache.retrieve_output(output_id).unwrap())
    }
}

fn generate_to_evaluate(
    graph: &MyGraph,
    outputs_cache: &OutputsCache,
    node_id: NodeId,
) -> anyhow::Result<Vec<NodeId>> {
    let mut dependencies = vec![];
    let mut queue = VecDeque::new();
    queue.push_back(node_id);

    // first find all possible dependencies
    while let Some(n_id) = queue.pop_front() {
        dependencies.push(n_id);
        for id in graph[n_id].input_ids() {
            if let Some(output_id) = graph.connection(id) {
                let node = graph.get_output(output_id).node;
                queue.push_back(node);
            }
        }
    }
    dependencies.reverse();

    // then keep only the ones that have an input that is different or not
    // cached
    let mut res = vec![];
    let mut invalid_ids = HashSet::new();

    for n_id in dependencies {
        let mut input_hashes = vec![];

        if let NodeInstruction::SaveString = graph[n_id].user_data.template {
            res.push(n_id);
            invalid_ids.insert(n_id);
            outputs_cache.invalidate_outputs(graph, n_id);
            continue;
        }

        let first_output = if let Some(o) = graph[n_id].output_ids().next() {
            o
        } else {
            continue;
        };
        let hashes =
            if let Some(hashes) = outputs_cache.input_hashes(&first_output) {
                hashes
            } else {
                res.push(n_id);
                invalid_ids.insert(n_id);
                outputs_cache.invalidate_outputs(graph, n_id);
                continue;
            };

        for (input_id, input_hash) in graph[n_id].input_ids().zip(hashes.iter())
        {
            if let Some(output_id) = graph.connection(input_id) {
                let node = graph.get_output(output_id).node;
                if invalid_ids.contains(&node) {
                    res.push(n_id);
                    invalid_ids.insert(n_id);
                    outputs_cache.invalidate_outputs(graph, n_id);
                    continue;
                }
                // if we have a connection we assume that the input hasnt
                // changed so we add the last known value
                input_hashes.push(*input_hash);
            } else {
                input_hashes
                    .push(OutputsCache::calculate_hash(&graph[input_id].value));
            }
        }

        for output_id in graph[n_id].output_ids() {
            if !outputs_cache.same_hash_inputs(&output_id, &input_hashes) {
                res.push(n_id);
                invalid_ids.insert(n_id);
                outputs_cache.invalidate_outputs(graph, n_id);
                continue;
            }
        }
    }

    // dedup while preserving order
    {
        let mut set = HashSet::new();
        res.retain(|x| set.insert(*x));
    }

    Ok(res)
}

// -----------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn process_template(
    graph: &MyGraph,
    node_id: NodeId,
    outputs_cache: &OutputsCache,
    template: &NodeInstruction,
    output_name: &str,
    translator: &mut rsprocess::translator::Translator,
    to_ret: &mut Option<BasicValue>,
    ctx: &eframe::egui::Context,
) -> anyhow::Result<Option<BasicValue>> {
    // macro that builds a tuple of retrieved values from cache
    // same order as in the definition of the inputs
    macro_rules! retrieve_from_cache {
        (@accum (0) -> ($($body:tt)*))
            => {retrieve_from_cache!(@as_expr ($($body)*))};
        (@accum (1) -> ($($body:tt)*))
            => {retrieve_from_cache!(
                @accum (0) -> ($($body)*
                              outputs_cache.retrieve_cache_output(
                                  graph,
                                  node_id,
                                  &graph[node_id]
                                      .user_data
                                      .template
                                      .inputs()[0].0.clone())?,))};
        (@accum (2) -> ($($body:tt)*))
            => {retrieve_from_cache!(
                @accum (1) -> ($($body)*
                              outputs_cache.retrieve_cache_output(
                                  graph,
                                  node_id,
                                  &graph[node_id]
                                      .user_data
                                      .template
                                      .inputs()[1].0.clone())?,))};
        (@accum (3) -> ($($body:tt)*))
            => {retrieve_from_cache!(
                @accum (2) -> ($($body)*
                              outputs_cache.retrieve_cache_output(
                                  graph,
                                  node_id,
                                  &graph[node_id]
                                      .user_data
                                      .template
                                      .inputs()[2].0.clone())?,))};
        (@accum (4) -> ($($body:tt)*))
            => {retrieve_from_cache!(
                @accum (3) -> ($($body)*
                              outputs_cache.retrieve_cache_output(
                                  graph,
                                  node_id,
                                  &graph[node_id]
                                      .user_data
                                      .template
                                      .inputs()[3].0.clone())?,))};
        (@accum (5) -> ($($body:tt)*))
            => {retrieve_from_cache!(
                @accum (4) -> ($($body)*
                              outputs_cache.retrieve_cache_output(
                                  graph,
                                  node_id,
                                  &graph[node_id]
                                      .user_data
                                      .template
                                      .inputs()[4].0.clone())?,))};
        (@accum (6) -> ($($body:tt)*))
            => {retrieve_from_cache!(
                @accum (5) -> ($($body)*
                              outputs_cache.retrieve_cache_output(
                                  graph,
                                  node_id,
                                  &graph[node_id]
                                      .user_data
                                      .template
                                      .inputs()[5].0.clone())?,))};
        (@accum (7) -> ($($body:tt)*))
            => {retrieve_from_cache!(
                @accum (6) -> ($($body)*
                              outputs_cache.retrieve_cache_output(
                                  graph,
                                  node_id,
                                  &graph[node_id]
                                      .user_data
                                      .template
                                      .inputs()[6].0.clone())?,))};
        (@accum (8) -> ($($body:tt)*))
            => {retrieve_from_cache!(
                @accum (7) -> ($($body)*
                              outputs_cache.retrieve_cache_output(
                                  graph,
                                  node_id,
                                  &graph[node_id]
                                      .user_data
                                      .template
                                      .inputs()[7].0.clone())?,))};
        (@accum (9) -> ($($body:tt)*))
            => {retrieve_from_cache!(
                @accum (8) -> ($($body)*
                              outputs_cache.retrieve_cache_output(
                                  graph,
                                  node_id,
                                  &graph[node_id]
                                      .user_data
                                      .template
                                      .inputs()[8].0.clone())?,))};
        (@accum (10) -> ($($body:tt)*))
            => {retrieve_from_cache!(
                @accum (9) -> ($($body)*
                              outputs_cache.retrieve_cache_output(
                                  graph,
                                  node_id,
                                  &graph[node_id]
                                      .user_data
                                      .template
                                      .inputs()[9].0.clone())?,))};

        (@as_expr $e:expr) => {$e};
        [0] => {
            compile_error!("Macro returns a value or a tuple, supply an \
                            integer greater than 0")
        };
        [1] => {
            outputs_cache.retrieve_cache_output(
                graph,
                node_id,
                &graph[node_id]
                    .user_data
                    .template
                    .inputs().first().unwrap().0.clone())?
        };
        [$n:tt] => {
            { retrieve_from_cache!(@accum ($n) -> ()) }
        };
    }

    // creates a vector of the hash of the inputs
    macro_rules! hash_inputs {
        ($($i:ident),*) => (vec![$(OutputsCache::calculate_hash(&$i)),*]);
    }

    match template {
        | NodeInstruction::String => {
            let s = retrieve_from_cache![1];
            let hash_inputs = hash_inputs!(s);

            if let BasicValue::String { value: _ } = s {
                let res = s;
                outputs_cache.populate_output(
                    graph,
                    node_id,
                    output_name,
                    res,
                    hash_inputs,
                )?;
            } else {
                anyhow::bail!("Not a string");
            }
        },
        | NodeInstruction::Path => {
            let s = retrieve_from_cache![1];
            let hash_inputs = hash_inputs!(s);

            if let BasicValue::String { value } = s {
                let res = BasicValue::Path { value };
                outputs_cache.populate_output(
                    graph,
                    node_id,
                    output_name,
                    res,
                    hash_inputs,
                )?;
            } else {
                anyhow::bail!("Not a string");
            }
        },
        | NodeInstruction::ReadPath => {
            let s = retrieve_from_cache![1];
            let hash_inputs = hash_inputs!(s);

            if let BasicValue::Path { value } = s {
                let file = match std::fs::read_to_string(value) {
                    | Ok(f) => f,
                    | Err(e) =>
                        return Ok(Some(BasicValue::Error {
                            value: helper::reformat_generic_error(
                                e.to_string(),
                                ctx,
                            ),
                        })),
                };
                let res = BasicValue::String { value: file };
                outputs_cache.populate_output(
                    graph,
                    node_id,
                    output_name,
                    res,
                    hash_inputs,
                )?;
            } else {
                anyhow::bail!("Not a path");
            }
        },
        | NodeInstruction::System => {
            let s = retrieve_from_cache![1];
            let hash_inputs = hash_inputs!(s);

            if let BasicValue::String { value } = s {
                let res = grammar_separated::grammar::SystemParser::new()
                    .parse(&mut *translator, &value);
                let sys = match res {
                    | Ok(s) => s,
                    | Err(parse_error) => {
                        return Ok(Some(BasicValue::Error {
                            value: helper::reformat_error(
                                parse_error,
                                &value,
                                ctx,
                            ),
                        }));
                    },
                };
                let res = BasicValue::System { value: sys };
                outputs_cache.populate_output(
                    graph,
                    node_id,
                    output_name,
                    res,
                    hash_inputs,
                )?;
            } else {
                anyhow::bail!("Not a string");
            }
        },
        | NodeInstruction::Statistics => {
            let s = retrieve_from_cache![1];
            let hash_inputs = hash_inputs!(s);

            if let BasicValue::System { value } = s {
                let res = BasicValue::String {
                    value: value.statistics(translator),
                };
                outputs_cache.populate_output(
                    graph,
                    node_id,
                    output_name,
                    res,
                    hash_inputs,
                )?;
            } else {
                anyhow::bail!("Not a system");
            }
        },
        | NodeInstruction::Target => {
            let (s, i) = retrieve_from_cache![2];
            let hash_inputs = hash_inputs!(s, i);

            match (s, i) {
                | (
                    BasicValue::System { value: s },
                    BasicValue::PositiveInt { value: i },
                ) => {
                    let limit = if i > 0 {
                        match s.target_limit(i) {
                            | Ok(l) => l,
                            | Err(e) => anyhow::bail!(e),
                        }
                    } else {
                        match s.target() {
                            | Ok(l) => l,
                            | Err(e) => anyhow::bail!(e),
                        }
                    };
                    let res = BasicValue::String {
                        value: format!(
                            "After {} steps arrived at state {}",
                            limit.0,
                            Formatter::from(translator, &limit.1)
                        ),
                    };
                    outputs_cache.populate_output(
                        graph,
                        node_id,
                        output_name,
                        res,
                        hash_inputs,
                    )?;
                },
                | (BasicValue::System { value: _ }, _) =>
                    anyhow::bail!("Not an integer"),
                | (_, BasicValue::PositiveInt { value: _ }) =>
                    anyhow::bail!("Not a system"),
                | (_, _) => anyhow::bail!("Inputs all wrong"),
            }
        },
        | NodeInstruction::Run => {
            let (s, i) = retrieve_from_cache![2];
            let hash_inputs = hash_inputs!(s, i);

            match (s, i) {
                | (
                    BasicValue::System { value: s },
                    BasicValue::PositiveInt { value: i },
                ) => {
                    let limit = if i > 0 {
                        match s.run_separated_limit(i) {
                            | Ok(l) => l,
                            | Err(e) => anyhow::bail!(e),
                        }
                    } else {
                        match s.run_separated() {
                            | Ok(l) => l,
                            | Err(e) => anyhow::bail!(e),
                        }
                    };
                    let mut output = String::new();
                    output.push_str(
                        "The trace is composed by the set of entities: ",
                    );
                    for (e, _c, _t) in limit {
                        output.push_str(&format!(
                            "{}",
                            Formatter::from(translator, &e)
                        ));
                    }
                    let res = BasicValue::String { value: output };
                    outputs_cache.populate_output(
                        graph,
                        node_id,
                        output_name,
                        res,
                        hash_inputs,
                    )?;
                },
                | (BasicValue::System { value: _ }, _) =>
                    anyhow::bail!("Not an integer"),
                | (_, BasicValue::PositiveInt { value: _ }) =>
                    anyhow::bail!("Not a system"),
                | (_, _) => anyhow::bail!("Inputs all wrong"),
            }
        },
        | NodeInstruction::Loop => {
            let (s, i) = retrieve_from_cache![2];
            let hash_inputs = hash_inputs!(s, i);

            match (s, i) {
                | (
                    BasicValue::System { value: sys },
                    BasicValue::Symbol { value: i },
                ) => {
                    let s = match translator.encode_not_mut(i) {
                        | Some(s) => s,
                        | None => anyhow::bail!("Symbol not found"),
                    };
                    let l = match sys.lollipops_only_loop_named(s) {
                        | Some(l) => l,
                        | None => anyhow::bail!("No loop found"),
                    };
                    let mut output = String::new();
                    output.push_str("The loop is composed by the sets: ");
                    for e in l {
                        output.push_str(&format!(
                            "{}",
                            Formatter::from(translator, &e)
                        ));
                    }
                    let res = BasicValue::String { value: output };
                    outputs_cache.populate_output(
                        graph,
                        node_id,
                        output_name,
                        res,
                        hash_inputs,
                    )?;
                },
                | (BasicValue::System { value: _ }, _) =>
                    anyhow::bail!("Not an integer"),
                | (_, BasicValue::PositiveInt { value: _ }) =>
                    anyhow::bail!("Not a system"),
                | (_, _) => anyhow::bail!("Inputs all wrong"),
            }
        },
        | NodeInstruction::Symbol => {
            let s = retrieve_from_cache![1];
            let hash_inputs = hash_inputs!(s);

            if let BasicValue::String { value } = s {
                let res = BasicValue::Symbol { value };
                outputs_cache.populate_output(
                    graph,
                    node_id,
                    output_name,
                    res,
                    hash_inputs,
                )?;
            } else {
                anyhow::bail!("Not a string");
            }
        },
        | NodeInstruction::Frequency => {
            let s = retrieve_from_cache![1];
            let hash_inputs = hash_inputs!(s);

            if let BasicValue::System { value } = s {
                let res = match rsprocess::frequency::Frequency::naive_frequency(
                    &value,
                ) {
                    | Ok(r) => r,
                    | Err(e) => anyhow::bail!(e),
                };
                let output = format!(
                    "Frequency of encountered symbols:\n{}",
                    Formatter::from(translator, &res)
                );
                let res = BasicValue::String { value: output };
                outputs_cache.populate_output(
                    graph,
                    node_id,
                    output_name,
                    res,
                    hash_inputs,
                )?;
            } else {
                anyhow::bail!("Not a system");
            }
        },
        | NodeInstruction::LimitFrequency => {
            let (sys, exp) = retrieve_from_cache![2];
            let hash_inputs = hash_inputs!(sys, exp);

            match (sys, exp) {
                | (
                    BasicValue::System { value: sys },
                    BasicValue::Experiment { value: exp },
                ) => {
                    let (_, sets) = exp;
                    let l =
                        match rsprocess::frequency::Frequency::limit_frequency(
                            &sets,
                            &sys.reaction_rules,
                            &sys.available_entities,
                        ) {
                            | Some(l) => l,
                            | None => anyhow::bail!("No loop found"),
                        };
                    let res = BasicValue::String {
                        value: format!(
                            "Frequency of encountered symbols:\n{}",
                            Formatter::from(translator, &l)
                        ),
                    };
                    outputs_cache.populate_output(
                        graph,
                        node_id,
                        output_name,
                        res,
                        hash_inputs,
                    )?;
                },
                | (BasicValue::System { value: _ }, _) =>
                    anyhow::bail!("Not an experiment"),
                | (_, BasicValue::Experiment { value: _ }) =>
                    anyhow::bail!("Not a system"),
                | (_, _) => anyhow::bail!("Inputs all wrong"),
            }
        },
        | NodeInstruction::Experiment => {
            let s = retrieve_from_cache![1];
            let hash_inputs = hash_inputs!(s);

            if let BasicValue::String { value } = s {
                let value =
                    match grammar_separated::grammar::ExperimentParser::new()
                        .parse(translator, &value)
                    {
                        | Ok(v) => v,
                        | Err(e) =>
                            return Ok(Some(BasicValue::Error {
                                value: helper::reformat_error(e, &value, ctx),
                            })),
                    };
                let res = BasicValue::Experiment { value };
                outputs_cache.populate_output(
                    graph,
                    node_id,
                    output_name,
                    res,
                    hash_inputs,
                )?;
            } else {
                anyhow::bail!("Not a string");
            }
        },
        | NodeInstruction::FastFrequency => {
            let (sys, exp) = retrieve_from_cache![2];
            let hash_inputs = hash_inputs!(sys, exp);

            match (sys, exp) {
                | (
                    BasicValue::System { value: sys },
                    BasicValue::Experiment { value: exp },
                ) => {
                    let (weights, sets) = exp;
                    let l =
                        match rsprocess::frequency::Frequency::fast_frequency(
                            &sets,
                            &sys.reaction_rules,
                            &sys.available_entities,
                            &weights,
                        ) {
                            | Some(l) => l,
                            | None => anyhow::bail!("No loop found"),
                        };
                    let res = BasicValue::String {
                        value: format!(
                            "Frequency of encountered symbols:\n{}",
                            Formatter::from(translator, &l)
                        ),
                    };
                    outputs_cache.populate_output(
                        graph,
                        node_id,
                        output_name,
                        res,
                        hash_inputs,
                    )?;
                },
                | (BasicValue::System { value: _ }, _) =>
                    anyhow::bail!("Not an experiment"),
                | (_, BasicValue::Experiment { value: _ }) =>
                    anyhow::bail!("Not a system"),
                | (_, _) => anyhow::bail!("Inputs all wrong"),
            }
        },
        | NodeInstruction::BisimilarityKanellakisSmolka => {
            let (graph_1, graph_2, grouping) = retrieve_from_cache![3];
            let hash_inputs = hash_inputs!(graph_1, graph_2, grouping);

            match (graph_1, graph_2, grouping) {
                | (
                    BasicValue::Graph { value: graph_1 },
                    BasicValue::Graph { value: graph_2 },
                    BasicValue::GroupingFunction { value: grouping },
                ) => {
                    use execution::data::MapEdges;
                    let graph_1 = match graph_1.map_edges(&grouping, translator)
                    {
                        | Ok(g) => g,
                        | Err(e) => anyhow::bail!(e),
                    };
                    let graph_2 = match graph_2.map_edges(&grouping, translator)
                    {
                        | Ok(g) => g,
                        | Err(e) => anyhow::bail!(e),
                    };

                    let l = bisimilarity::bisimilarity_kanellakis_smolka::bisimilarity(&&graph_1, &&graph_2);
                    let res = BasicValue::String {
                        value: format!("{l}"),
                    };
                    outputs_cache.populate_output(
                        graph,
                        node_id,
                        output_name,
                        res,
                        hash_inputs,
                    )?;
                },
                | (_, _, _) => anyhow::bail!("Invalid inputs to bisimilarity."),
            }
        },
        | NodeInstruction::GroupFunction => {
            let s = retrieve_from_cache![1];
            let hash_inputs = hash_inputs!(s);

            if let BasicValue::String { value } = s {
                let res = grammar_separated::assert::AssertParser::new()
                    .parse(&mut *translator, &value);
                let res = match res {
                    | Ok(s) => s,
                    | Err(parse_error) => {
                        return Ok(Some(BasicValue::Error {
                            value: helper::reformat_error(
                                parse_error,
                                &value,
                                ctx,
                            ),
                        }));
                    },
                };
                let res = BasicValue::GroupingFunction { value: *res };
                outputs_cache.populate_output(
                    graph,
                    node_id,
                    output_name,
                    res,
                    hash_inputs,
                )?;
            } else {
                anyhow::bail!("Not a string");
            }
        },
        | NodeInstruction::BisimilarityPaigeTarjanNoLabels => {
            let (graph_1, graph_2, grouping) = retrieve_from_cache![3];
            let hash_inputs = hash_inputs!(graph_1, graph_2, grouping);

            match (graph_1, graph_2, grouping) {
                | (
                    BasicValue::Graph { value: graph_1 },
                    BasicValue::Graph { value: graph_2 },
                    BasicValue::GroupingFunction { value: grouping },
                ) => {
                    use execution::data::MapEdges;
                    let graph_1 = match graph_1.map_edges(&grouping, translator)
                    {
                        | Ok(g) => g,
                        | Err(e) => anyhow::bail!(e),
                    };
                    let graph_2 = match graph_2.map_edges(&grouping, translator)
                    {
                        | Ok(g) => g,
                        | Err(e) => anyhow::bail!(e),
                    };

                    let l = bisimilarity::bisimilarity_paige_tarkan::bisimilarity_ignore_labels(&&graph_1, &&graph_2);
                    let res = BasicValue::String {
                        value: format!("{l}"),
                    };
                    outputs_cache.populate_output(
                        graph,
                        node_id,
                        output_name,
                        res,
                        hash_inputs,
                    )?;
                },
                | (_, _, _) => anyhow::bail!("Invalid inputs to bisimilarity."),
            }
        },
        | NodeInstruction::BisimilarityPaigeTarjan => {
            let (graph_1, graph_2, grouping) = retrieve_from_cache![3];
            let hash_inputs = hash_inputs!(graph_1, graph_2, grouping);

            match (graph_1, graph_2, grouping) {
                | (
                    BasicValue::Graph { value: graph_1 },
                    BasicValue::Graph { value: graph_2 },
                    BasicValue::GroupingFunction { value: grouping },
                ) => {
                    use execution::data::MapEdges;
                    let graph_1 = match graph_1.map_edges(&grouping, translator)
                    {
                        | Ok(g) => g,
                        | Err(e) => anyhow::bail!(e),
                    };
                    let graph_2 = match graph_2.map_edges(&grouping, translator)
                    {
                        | Ok(g) => g,
                        | Err(e) => anyhow::bail!(e),
                    };

                    let l =
                        bisimilarity::bisimilarity_paige_tarkan::bisimilarity(
                            &&graph_1, &&graph_2,
                        );
                    let res = BasicValue::String {
                        value: format!("{l}"),
                    };
                    outputs_cache.populate_output(
                        graph,
                        node_id,
                        output_name,
                        res,
                        hash_inputs,
                    )?;
                },
                | (_, _, _) => anyhow::bail!("Invalid inputs to bisimilarity."),
            }
        },
        | NodeInstruction::SystemGraph => {
            let s = retrieve_from_cache![1];
            let hash_inputs = hash_inputs!(s);

            if let BasicValue::System { value } = s {
                let value = match value.digraph() {
                    | Ok(g) => g,
                    | Err(e) => anyhow::bail!(e),
                };
                let res = BasicValue::Graph { value };
                outputs_cache.populate_output(
                    graph,
                    node_id,
                    output_name,
                    res,
                    hash_inputs,
                )?;
            } else {
                anyhow::bail!("Not a system");
            }
        },
        | NodeInstruction::SaveString => {
            let (path, string) = retrieve_from_cache![2];

            match (path, string) {
                | (
                    BasicValue::Path { value: path },
                    BasicValue::String { value },
                ) => {
                    *to_ret = Some(BasicValue::SaveString { path, value });
                },
                | (BasicValue::Path { .. }, _) => {
                    anyhow::bail!("Not a string");
                },
                | (_, BasicValue::String { .. }) => {
                    anyhow::bail!("Not a path");
                },
                | (_, _) => {
                    anyhow::bail!("Values of wrong type");
                },
            }
        },
        | NodeInstruction::Dot => {
            let (
                input_graph,
                display_node,
                display_edge,
                color_node,
                color_edge,
            ) = retrieve_from_cache![5];
            let hash_inputs = hash_inputs!(
                input_graph,
                display_node,
                display_edge,
                color_node,
                color_edge
            );

            match (
                input_graph,
                display_node,
                display_edge,
                color_node,
                color_edge,
            ) {
                | (
                    BasicValue::Graph { value: input_graph },
                    BasicValue::DisplayNode {
                        value: display_node,
                    },
                    BasicValue::DisplayEdge {
                        value: display_edge,
                    },
                    BasicValue::ColorNode { value: color_node },
                    BasicValue::ColorEdge { value: color_edge },
                ) => {
                    use std::rc::Rc;

                    let rc_translator = Rc::new(translator.clone());
                    let modified_graph = input_graph.map(
                        display_node
                            .generate(Rc::clone(&rc_translator), &input_graph),
                        display_edge
                            .generate(Rc::clone(&rc_translator), &input_graph),
                    );

                    let input_graph = Rc::new(input_graph.to_owned());

                    let node_formatter = color_node.generate(
                        Rc::clone(&input_graph),
                        translator.encode_not_mut("*"),
                    );
                    let edge_formatter =
                        color_edge.generate(Rc::clone(&input_graph));

                    let dot = rsprocess::dot::Dot::with_attr_getters(
                        &modified_graph,
                        &[],
                        &edge_formatter,
                        &node_formatter,
                    );
                    let res = BasicValue::String {
                        value: format!("{dot}"),
                    };
                    outputs_cache.populate_output(
                        graph,
                        node_id,
                        output_name,
                        res,
                        hash_inputs,
                    )?;
                },
                | _ => {
                    anyhow::bail!("Values of wrong type");
                },
            }
        },
        | NodeInstruction::DisplayNode => {
            let s = retrieve_from_cache![1];
            let hash_inputs = hash_inputs!(s);

            if let BasicValue::String { value } = s {
                let res =
                    grammar_separated::instructions::SeparatorNodeParser::new()
                        .parse(&mut *translator, &value);
                let res = match res {
                    | Ok(s) => s,
                    | Err(parse_error) => {
                        return Ok(Some(BasicValue::Error {
                            value: helper::reformat_error(
                                parse_error,
                                &value,
                                ctx,
                            ),
                        }));
                    },
                };
                let res = BasicValue::DisplayNode { value: res };
                outputs_cache.populate_output(
                    graph,
                    node_id,
                    output_name,
                    res,
                    hash_inputs,
                )?;
            } else {
                anyhow::bail!("Not a string");
            }
        },
        | NodeInstruction::DisplayEdge => {
            let s = retrieve_from_cache![1];
            let hash_inputs = hash_inputs!(s);

            if let BasicValue::String { value } = s {
                let res =
                    grammar_separated::instructions::SeparatorEdgeParser::new()
                        .parse(&mut *translator, &value);
                let res = match res {
                    | Ok(s) => s,
                    | Err(parse_error) => {
                        return Ok(Some(BasicValue::Error {
                            value: helper::reformat_error(
                                parse_error,
                                &value,
                                ctx,
                            ),
                        }));
                    },
                };
                let res = BasicValue::DisplayEdge { value: res };
                outputs_cache.populate_output(
                    graph,
                    node_id,
                    output_name,
                    res,
                    hash_inputs,
                )?;
            } else {
                anyhow::bail!("Not a string");
            }
        },
        | NodeInstruction::ColorNode => {
            let s = retrieve_from_cache![1];
            let hash_inputs = hash_inputs!(s);

            if let BasicValue::String { value } = s {
                let res =
                    grammar_separated::instructions::ColorNodeParser::new()
                        .parse(&mut *translator, &value);
                let res = match res {
                    | Ok(s) => s,
                    | Err(parse_error) => {
                        return Ok(Some(BasicValue::Error {
                            value: helper::reformat_error(
                                parse_error,
                                &value,
                                ctx,
                            ),
                        }));
                    },
                };
                let res = BasicValue::ColorNode { value: res };
                outputs_cache.populate_output(
                    graph,
                    node_id,
                    output_name,
                    res,
                    hash_inputs,
                )?;
            } else {
                anyhow::bail!("Not a string");
            }
        },
        | NodeInstruction::ColorEdge => {
            let s = retrieve_from_cache![1];
            let hash_inputs = hash_inputs!(s);

            if let BasicValue::String { value } = s {
                let res =
                    grammar_separated::instructions::ColorEdgeParser::new()
                        .parse(&mut *translator, &value);
                let res = match res {
                    | Ok(s) => s,
                    | Err(parse_error) => {
                        return Ok(Some(BasicValue::Error {
                            value: helper::reformat_error(
                                parse_error,
                                &value,
                                ctx,
                            ),
                        }));
                    },
                };
                let res = BasicValue::ColorEdge { value: res };
                outputs_cache.populate_output(
                    graph,
                    node_id,
                    output_name,
                    res,
                    hash_inputs,
                )?;
            } else {
                anyhow::bail!("Not a string");
            }
        },
        | NodeInstruction::GraphML => {
            let (input_graph, display_node, display_edge) =
                retrieve_from_cache![3];
            let hash_inputs =
                hash_inputs!(input_graph, display_node, display_edge);

            match (input_graph, display_node, display_edge) {
                | (
                    BasicValue::Graph { value: input_graph },
                    BasicValue::DisplayNode {
                        value: display_node,
                    },
                    BasicValue::DisplayEdge {
                        value: display_edge,
                    },
                ) => {
                    use std::rc::Rc;

                    let rc_translator = Rc::new(translator.clone());
                    let modified_graph = input_graph.map(
                        display_node
                            .generate(Rc::clone(&rc_translator), &input_graph),
                        display_edge.generate(rc_translator, &input_graph),
                    );

                    use petgraph_graphml::GraphMl;
                    let graphml = GraphMl::new(&modified_graph)
                        .pretty_print(true)
                        .export_node_weights_display()
                        .export_edge_weights_display();

                    let res = BasicValue::String {
                        value: format!("{graphml}"),
                    };
                    outputs_cache.populate_output(
                        graph,
                        node_id,
                        output_name,
                        res,
                        hash_inputs,
                    )?;
                },
                | _ => {
                    anyhow::bail!("Values of wrong type");
                },
            }
        },
        | NodeInstruction::ComposeSystem => {
            let (
                input_env,
                input_initial_etities,
                input_context,
                input_reactions,
            ) = retrieve_from_cache![4];
            let hash_inputs = hash_inputs!(
                input_env,
                input_initial_etities,
                input_context,
                input_reactions
            );

            match (
                input_env,
                input_initial_etities,
                input_context,
                input_reactions,
            ) {
                | (
                    BasicValue::Environment { value: env },
                    BasicValue::Set { value: set },
                    BasicValue::Context { value: context },
                    BasicValue::Reactions { value: reactions },
                ) => {
                    let res = BasicValue::System {
                        value: rsprocess::system::System::from(
                            Rc::new(env),
                            set,
                            context,
                            Rc::new(reactions),
                        ),
                    };
                    outputs_cache.populate_output(
                        graph,
                        node_id,
                        output_name,
                        res,
                        hash_inputs,
                    )?;
                },
                | _ => {
                    anyhow::bail!("Values of wrong type");
                },
            }
        },
        | NodeInstruction::Environment => {
            let s = retrieve_from_cache![1];
            let hash_inputs = hash_inputs!(s);

            if let BasicValue::String { value } = s {
                let res = grammar_separated::grammar::EnvironmentParser::new()
                    .parse(&mut *translator, &value);
                let env = match res {
                    | Ok(s) => s,
                    | Err(parse_error) => {
                        return Ok(Some(BasicValue::Error {
                            value: helper::reformat_error(
                                parse_error,
                                &value,
                                ctx,
                            ),
                        }));
                    },
                };
                let res = BasicValue::Environment { value: *env };
                outputs_cache.populate_output(
                    graph,
                    node_id,
                    output_name,
                    res,
                    hash_inputs,
                )?;
            } else {
                anyhow::bail!("Not a string");
            }
        },
        | NodeInstruction::Set => {
            let s = retrieve_from_cache![1];
            let hash_inputs = hash_inputs!(s);

            if let BasicValue::String { value } = s {
                let res = grammar_separated::grammar::SetParser::new()
                    .parse(&mut *translator, &value);
                let set = match res {
                    | Ok(s) => s,
                    | Err(parse_error) => {
                        return Ok(Some(BasicValue::Error {
                            value: helper::reformat_error(
                                parse_error,
                                &value,
                                ctx,
                            ),
                        }));
                    },
                };
                let res = BasicValue::Set { value: set };
                outputs_cache.populate_output(
                    graph,
                    node_id,
                    output_name,
                    res,
                    hash_inputs,
                )?;
            } else {
                anyhow::bail!("Not a string");
            }
        },
        | NodeInstruction::Context => {
            let s = retrieve_from_cache![1];
            let hash_inputs = hash_inputs!(s);

            if let BasicValue::String { value } = s {
                let res = grammar_separated::grammar::ContextParser::new()
                    .parse(&mut *translator, &value);
                let context = match res {
                    | Ok(s) => s,
                    | Err(parse_error) => {
                        return Ok(Some(BasicValue::Error {
                            value: helper::reformat_error(
                                parse_error,
                                &value,
                                ctx,
                            ),
                        }));
                    },
                };
                let res = BasicValue::Context { value: context };
                outputs_cache.populate_output(
                    graph,
                    node_id,
                    output_name,
                    res,
                    hash_inputs,
                )?;
            } else {
                anyhow::bail!("Not a string");
            }
        },
        | NodeInstruction::Reactions => {
            let s = retrieve_from_cache![1];
            let hash_inputs = hash_inputs!(s);

            if let BasicValue::String { value } = s {
                let res = grammar_separated::grammar::ReactionsParser::new()
                    .parse(&mut *translator, &value);
                let reactions = match res {
                    | Ok(s) => s,
                    | Err(parse_error) => {
                        return Ok(Some(BasicValue::Error {
                            value: helper::reformat_error(
                                parse_error,
                                &value,
                                ctx,
                            ),
                        }));
                    },
                };
                let res = BasicValue::Reactions { value: reactions };
                outputs_cache.populate_output(
                    graph,
                    node_id,
                    output_name,
                    res,
                    hash_inputs,
                )?;
            } else {
                anyhow::bail!("Not a string");
            }
        },
        | NodeInstruction::PositiveSystem => {
            let s = retrieve_from_cache![1];
            let hash_inputs = hash_inputs!(s);

            if let BasicValue::System { value } = s {
                let res = BasicValue::PositiveSystem {
                    value: value.into(),
                };
                outputs_cache.populate_output(
                    graph,
                    node_id,
                    output_name,
                    res,
                    hash_inputs,
                )?;
            } else {
                anyhow::bail!("Not a system");
            }
        },
        | NodeInstruction::PositiveTarget => {
            let (s, i) = retrieve_from_cache![2];
            let hash_inputs = hash_inputs!(s, i);

            match (s, i) {
                | (
                    BasicValue::PositiveSystem { value: s },
                    BasicValue::PositiveInt { value: i },
                ) => {
                    let limit = if i > 0 {
                        match s.target_limit(i) {
                            | Ok(l) => l,
                            | Err(e) => anyhow::bail!(e),
                        }
                    } else {
                        match s.target() {
                            | Ok(l) => l,
                            | Err(e) => anyhow::bail!(e),
                        }
                    };
                    let res = BasicValue::String {
                        value: format!(
                            "After {} steps arrived at state {}",
                            limit.0,
                            Formatter::from(translator, &limit.1)
                        ),
                    };
                    outputs_cache.populate_output(
                        graph,
                        node_id,
                        output_name,
                        res,
                        hash_inputs,
                    )?;
                },
                | (BasicValue::PositiveSystem { value: _ }, _) =>
                    anyhow::bail!("Not an integer"),
                | (_, BasicValue::PositiveInt { value: _ }) =>
                    anyhow::bail!("Not a positive system"),
                | (_, _) => anyhow::bail!("Inputs all wrong"),
            }
        },
        | NodeInstruction::PositiveRun => {
            let (s, i) = retrieve_from_cache![2];
            let hash_inputs = hash_inputs!(s, i);

            match (s, i) {
                | (
                    BasicValue::PositiveSystem { value: s },
                    BasicValue::PositiveInt { value: i },
                ) => {
                    let limit = if i > 0 {
                        match s.run_separated_limit(i) {
                            | Ok(l) => l,
                            | Err(e) => anyhow::bail!(e),
                        }
                    } else {
                        match s.run_separated() {
                            | Ok(l) => l,
                            | Err(e) => anyhow::bail!(e),
                        }
                    };
                    let mut output = String::new();
                    output.push_str(
                        "The trace is composed by the set of entities: ",
                    );
                    for (e, _c, _t) in limit {
                        output.push_str(&format!(
                            "{}",
                            Formatter::from(translator, &e)
                        ));
                    }
                    let res = BasicValue::String { value: output };
                    outputs_cache.populate_output(
                        graph,
                        node_id,
                        output_name,
                        res,
                        hash_inputs,
                    )?;
                },
                | (BasicValue::PositiveSystem { value: _ }, _) =>
                    anyhow::bail!("Not an integer"),
                | (_, BasicValue::PositiveInt { value: _ }) =>
                    anyhow::bail!("Not a positive system"),
                | (_, _) => anyhow::bail!("Inputs all wrong"),
            }
        },
        | NodeInstruction::PositiveLoop => {
            let (s, i) = retrieve_from_cache![2];
            let hash_inputs = hash_inputs!(s, i);

            match (s, i) {
                | (
                    BasicValue::PositiveSystem { value: sys },
                    BasicValue::Symbol { value: i },
                ) => {
                    let s = match translator.encode_not_mut(i) {
                        | Some(s) => s,
                        | None => anyhow::bail!("Symbol not found"),
                    };
                    let l = match sys.lollipops_only_loop_named(s) {
                        | Some(l) => l,
                        | None => anyhow::bail!("No loop found"),
                    };
                    let mut output = String::new();
                    output.push_str("The loop is composed by the sets: ");
                    for e in l {
                        output.push_str(&format!(
                            "{}",
                            Formatter::from(translator, &e)
                        ));
                    }
                    let res = BasicValue::String { value: output };
                    outputs_cache.populate_output(
                        graph,
                        node_id,
                        output_name,
                        res,
                        hash_inputs,
                    )?;
                },
                | (BasicValue::PositiveSystem { value: _ }, _) =>
                    anyhow::bail!("Not an integer"),
                | (_, BasicValue::PositiveInt { value: _ }) =>
                    anyhow::bail!("Not a positive system"),
                | (_, _) => anyhow::bail!("Inputs all wrong"),
            }
        },
        | NodeInstruction::PositiveFrequency => {
            let s = retrieve_from_cache![1];
            let hash_inputs = hash_inputs!(s);

            if let BasicValue::PositiveSystem { value } = s {
                let res = match rsprocess::frequency::PositiveFrequency::naive_frequency(&value) {
                    Ok(r) => r,
                    Err(e) => anyhow::bail!(e),
                };
                let output = format!(
                    "Frequency of encountered symbols:\n{}",
                    Formatter::from(translator, &res)
                );
                let res = BasicValue::String { value: output };
                outputs_cache.populate_output(
                    graph,
                    node_id,
                    output_name,
                    res,
                    hash_inputs,
                )?;
            } else {
                anyhow::bail!("Not a positive system");
            }
        },
        | NodeInstruction::PositiveLimitFrequency => {
            let (sys, exp) = retrieve_from_cache![2];
            let hash_inputs = hash_inputs!(sys, exp);

            match (sys, exp) {
                | (
                    BasicValue::PositiveSystem { value: sys },
                    BasicValue::Experiment { value: exp },
                ) => {
                    let (_, sets) = exp;
                    let l = match rsprocess::frequency::PositiveFrequency::limit_frequency(
                        &sets.into_iter().map(
                            |e| e.to_positive_set(rsprocess::element::IdState::Positive)
                        ).collect::<Vec<_>>(),
                        &sys.reaction_rules,
                        &sys.available_entities,
                    ) {
                        Some(l) => l,
                        None => anyhow::bail!("No loop found")
                    };
                    let res = BasicValue::String {
                        value: format!(
                            "Frequency of encountered symbols:\n{}",
                            Formatter::from(translator, &l)
                        ),
                    };
                    outputs_cache.populate_output(
                        graph,
                        node_id,
                        output_name,
                        res,
                        hash_inputs,
                    )?;
                },
                | (BasicValue::PositiveSystem { value: _ }, _) =>
                    anyhow::bail!("Not an experiment"),
                | (_, BasicValue::Experiment { value: _ }) =>
                    anyhow::bail!("Not a positive system"),
                | (_, _) => anyhow::bail!("Inputs all wrong"),
            }
        },
        | NodeInstruction::PositiveFastFrequency => {
            let (sys, exp) = retrieve_from_cache![2];
            let hash_inputs = hash_inputs!(sys, exp);

            match (sys, exp) {
                | (
                    BasicValue::PositiveSystem { value: sys },
                    BasicValue::Experiment { value: exp },
                ) => {
                    let (weights, sets) = exp;
                    let l = match rsprocess::frequency::PositiveFrequency::fast_frequency(
                        &sets.into_iter().map(
                            |e| e.to_positive_set(rsprocess::element::IdState::Positive)
                        ).collect::<Vec<_>>(),
                        &sys.reaction_rules,
                        &sys.available_entities,
                        &weights
                    ) {
                        Some(l) => l,
                        None => anyhow::bail!("No loop found")
                    };
                    let res = BasicValue::String {
                        value: format!(
                            "Frequency of encountered symbols:\n{}",
                            Formatter::from(translator, &l)
                        ),
                    };
                    outputs_cache.populate_output(
                        graph,
                        node_id,
                        output_name,
                        res,
                        hash_inputs,
                    )?;
                },
                | (BasicValue::PositiveSystem { value: _ }, _) =>
                    anyhow::bail!("Not an experiment"),
                | (_, BasicValue::Experiment { value: _ }) =>
                    anyhow::bail!("Not a system"),
                | (_, _) => anyhow::bail!("Inputs all wrong"),
            }
        },
        | NodeInstruction::Trace => {
            let (s, limit) = retrieve_from_cache![2];
            let hash_inputs = hash_inputs!(s, limit);

            match (s, limit) {
                | (
                    BasicValue::System { value: s },
                    BasicValue::PositiveInt { value: limit },
                ) => {
                    let trace = if limit == 0 {
                        match s.slice_trace() {
                            | Ok(t) => t,
                            | Err(e) => anyhow::bail!(e),
                        }
                    } else {
                        match s.slice_trace_limit(limit) {
                            | Ok(t) => t,
                            | Err(e) => anyhow::bail!(e),
                        }
                    };
                    let res = BasicValue::Trace { value: trace };
                    outputs_cache.populate_output(
                        graph,
                        node_id,
                        output_name,
                        res,
                        hash_inputs,
                    )?;
                },
                | (BasicValue::System { value: _ }, _) =>
                    anyhow::bail!("Not a positive integer"),
                | (_, BasicValue::PositiveInt { value: _ }) =>
                    anyhow::bail!("Not a system"),
                | (_, _) => anyhow::bail!("Inputs all wrong"),
            }
        },
        | NodeInstruction::PositiveTrace => {
            let (s, limit) = retrieve_from_cache![2];
            let hash_inputs = hash_inputs!(s, limit);

            match (s, limit) {
                | (
                    BasicValue::PositiveSystem { value: s },
                    BasicValue::PositiveInt { value: limit },
                ) => {
                    let trace = if limit == 0 {
                        match s.slice_trace() {
                            | Ok(t) => t,
                            | Err(e) => anyhow::bail!(e),
                        }
                    } else {
                        match s.slice_trace_limit(limit) {
                            | Ok(t) => t,
                            | Err(e) => anyhow::bail!(e),
                        }
                    };
                    let res = BasicValue::PositiveTrace { value: trace };
                    outputs_cache.populate_output(
                        graph,
                        node_id,
                        output_name,
                        res,
                        hash_inputs,
                    )?;
                },
                | (BasicValue::PositiveSystem { value: _ }, _) =>
                    anyhow::bail!("Not a positive integer"),
                | (_, BasicValue::PositiveInt { value: _ }) =>
                    anyhow::bail!("Not a positive system"),
                | (_, _) => anyhow::bail!("Inputs all wrong"),
            }
        },
        | NodeInstruction::SliceTrace => {
            let (trace, set) = retrieve_from_cache![2];
            let hash_inputs = hash_inputs!(trace, set);

            match (trace, set) {
                | (
                    BasicValue::Trace { value: trace },
                    BasicValue::Set { value: set },
                ) => {
                    let new_trace = match trace.slice(set) {
                        | Ok(t) => t,
                        | Err(e) => anyhow::bail!(e),
                    };

                    let res = BasicValue::Trace { value: new_trace };
                    outputs_cache.populate_output(
                        graph,
                        node_id,
                        output_name,
                        res,
                        hash_inputs,
                    )?;
                },
                | (BasicValue::Trace { value: _ }, _) =>
                    anyhow::bail!("Not a set"),
                | (_, BasicValue::Set { value: _ }) =>
                    anyhow::bail!("Not a trace"),
                | (_, _) => anyhow::bail!("Inputs all wrong"),
            }
        },
        | NodeInstruction::PositiveSliceTrace => {
            let (trace, set) = retrieve_from_cache![2];
            let hash_inputs = hash_inputs!(trace, set);

            match (trace, set) {
                | (
                    BasicValue::PositiveTrace { value: trace },
                    BasicValue::PositiveSet { value: set },
                ) => {
                    let new_trace = match trace.slice(set) {
                        | Ok(t) => t,
                        | Err(e) => anyhow::bail!(e),
                    };

                    let res = BasicValue::PositiveTrace { value: new_trace };
                    outputs_cache.populate_output(
                        graph,
                        node_id,
                        output_name,
                        res,
                        hash_inputs,
                    )?;
                },
                | (BasicValue::PositiveTrace { value: _ }, _) =>
                    anyhow::bail!("Not a set"),
                | (_, BasicValue::PositiveSet { value: _ }) =>
                    anyhow::bail!("Not a trace"),
                | (_, _) => anyhow::bail!("Inputs all wrong"),
            }
        },
        | NodeInstruction::PositiveSet => {
            let s = retrieve_from_cache![1];
            let hash_inputs = hash_inputs!(s);

            if let BasicValue::String { value } = s {
                let res = grammar_separated::grammar::PositiveSetParser::new()
                    .parse(&mut *translator, &value);
                let set = match res {
                    | Ok(s) => s,
                    | Err(parse_error) => {
                        return Ok(Some(BasicValue::Error {
                            value: helper::reformat_error(
                                parse_error,
                                &value,
                                ctx,
                            ),
                        }));
                    },
                };
                let res = BasicValue::PositiveSet { value: set };
                outputs_cache.populate_output(
                    graph,
                    node_id,
                    output_name,
                    res,
                    hash_inputs,
                )?;
            } else {
                anyhow::bail!("Not a string");
            }
        },
        | NodeInstruction::ToPositiveSet => {
            let s = retrieve_from_cache![1];
            let hash_inputs = hash_inputs!(s);

            if let BasicValue::Set { value } = s {
                let res = BasicValue::PositiveSet {
                    value: value
                        .to_positive_set(rsprocess::element::IdState::Positive),
                };
                outputs_cache.populate_output(
                    graph,
                    node_id,
                    output_name,
                    res,
                    hash_inputs,
                )?;
            } else {
                anyhow::bail!("Not a string");
            }
        },
    }
    Ok(None)
}
