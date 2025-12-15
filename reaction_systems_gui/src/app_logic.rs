use std::collections::{HashSet, VecDeque};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use egui_node_graph2::*;
use rsprocess::frequency::BasicFrequency;
use rsprocess::system::{BasicSystem, ExtensionsSystem, LoopSystem};
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
    translator: Arc<Mutex<rsprocess::translator::Translator>>,
    cancel_computation: Arc<AtomicBool>,
    ctx: &eframe::egui::Context,
) -> anyhow::Result<()> {
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
        // return early if someone asked
        if cancel_computation.load(std::sync::atomic::Ordering::Acquire) {
            anyhow::bail!("Computation Interrupted");
        }

        let node = &graph[node_id];
        let outputs = graph[node_id].user_data.template.output();
        let output_names =
            outputs.iter().map(|el| el.0.as_str()).collect::<Vec<_>>();

        match process_template(
            graph,
            node_id,
            outputs_cache,
            &node.user_data.template,
            output_names,
            Arc::clone(&translator),
            &mut to_ret,
            ctx,
        )? {
            | None => {},
            | Some(val) => {
                outputs_cache.set_last_state(val);
                return Ok(());
            },
        }
    }

    if let Some(res) = to_ret.take() {
        outputs_cache.set_last_state(res);
    } else {
        let output_field = graph[node_id]
            .user_data
            .template
            .output()
            .first()
            .map(|el| el.0.clone())
            .unwrap_or("".into());
        let output_id = graph[node_id].get_output(&output_field)?;

        let output = outputs_cache.retrieve_output(output_id).unwrap();
        outputs_cache.set_last_state(output);
    }
    ctx.request_repaint();
    Ok(())
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

        match graph[n_id].user_data.template {
            | NodeInstruction::SaveString |
            NodeInstruction::SaveSvg |
            NodeInstruction::SaveRasterization => {
                res.push(n_id);
                invalid_ids.insert(n_id);
                outputs_cache.invalidate_outputs(graph, n_id);
                continue;
            },
            | _ => {},
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
    output_names: Vec<&str>,
    translator: Arc<Mutex<rsprocess::translator::Translator>>,
    to_ret: &mut Option<BasicValue>,
    ctx: &eframe::egui::Context,
) -> anyhow::Result<Option<BasicValue>> {
    // macro that builds a tuple of retrieved values from cache
    // same order as in the definition of the inputs
    macro_rules! retrieve_from_cache {
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
        (@accum (0) -> ($($body:tt)*))
            => {retrieve_from_cache!(@as_expr ($($body)*))};
        (@accum (1) -> ($($body:tt)*))
            => {retrieve_from_cache!(
                @accum (0) -> (outputs_cache.retrieve_cache_output(
                                  graph,
                                  node_id,
                                  &graph[node_id]
                                      .user_data
                                      .template
                                      .inputs()[0].0.clone())?, $($body)*))};
        (@accum (2) -> ($($body:tt)*))
            => {retrieve_from_cache!(
                @accum (1) -> (outputs_cache.retrieve_cache_output(
                                  graph,
                                  node_id,
                                  &graph[node_id]
                                      .user_data
                                      .template
                                      .inputs()[1].0.clone())?, $($body)*))};
        (@accum (3) -> ($($body:tt)*))
            => {retrieve_from_cache!(
                @accum (2) -> (outputs_cache.retrieve_cache_output(
                                  graph,
                                  node_id,
                                  &graph[node_id]
                                      .user_data
                                      .template
                                      .inputs()[2].0.clone())?, $($body)*))};
        (@accum (4) -> ($($body:tt)*))
            => {retrieve_from_cache!(
                @accum (3) -> (outputs_cache.retrieve_cache_output(
                                  graph,
                                  node_id,
                                  &graph[node_id]
                                      .user_data
                                      .template
                                      .inputs()[3].0.clone())?, $($body)*))};
        (@accum (5) -> ($($body:tt)*))
            => {retrieve_from_cache!(
                @accum (4) -> (outputs_cache.retrieve_cache_output(
                                  graph,
                                  node_id,
                                  &graph[node_id]
                                      .user_data
                                      .template
                                      .inputs()[4].0.clone())?, $($body)*))};
        (@accum (6) -> ($($body:tt)*))
            => {retrieve_from_cache!(
                @accum (5) -> (outputs_cache.retrieve_cache_output(
                                  graph,
                                  node_id,
                                  &graph[node_id]
                                      .user_data
                                      .template
                                      .inputs()[5].0.clone())?, $($body)*))};
        (@accum (7) -> ($($body:tt)*))
            => {retrieve_from_cache!(
                @accum (6) -> (outputs_cache.retrieve_cache_output(
                                  graph,
                                  node_id,
                                  &graph[node_id]
                                      .user_data
                                      .template
                                      .inputs()[6].0.clone())?, $($body)*))};
        (@accum (8) -> ($($body:tt)*))
            => {retrieve_from_cache!(
                @accum (7) -> (outputs_cache.retrieve_cache_output(
                                  graph,
                                  node_id,
                                  &graph[node_id]
                                      .user_data
                                      .template
                                      .inputs()[7].0.clone())?, $($body)*))};
        (@accum (9) -> ($($body:tt)*))
            => {retrieve_from_cache!(
                @accum (8) -> (outputs_cache.retrieve_cache_output(
                                  graph,
                                  node_id,
                                  &graph[node_id]
                                      .user_data
                                      .template
                                      .inputs()[8].0.clone())?, $($body)*))};
        (@accum (10) -> ($($body:tt)*))
            => {retrieve_from_cache!(
                @accum (9) -> (outputs_cache.retrieve_cache_output(
                                  graph,
                                  node_id,
                                  &graph[node_id]
                                      .user_data
                                      .template
                                      .inputs()[9].0.clone())?, $($body)*))};
        (@as_expr $e:expr) => {$e};
    }

    // creates a vector of the hash of the inputs
    macro_rules! hash_inputs {
        ($($i:ident),*) => (vec![$(OutputsCache::calculate_hash(&$i)),*]);
    }

    macro_rules! set_cache_output {
        ($(($name:expr, $res:expr, $hash_inputs:expr)),*) => (
            $(outputs_cache.populate_output(
                graph,
                node_id,
                $name,
                $res,
                $hash_inputs,
            )?;)*
        );
    }

    match template {
        | NodeInstruction::String => {
            let s = retrieve_from_cache![1];
            let hash_inputs = hash_inputs!(s);

            if let BasicValue::String { value: _ } = s {
                let res = s;
                set_cache_output!((
                    output_names.first().unwrap(),
                    res,
                    hash_inputs
                ));
            } else {
                anyhow::bail!("Not a string");
            }
        },
        | NodeInstruction::Path => {
            let s = retrieve_from_cache![1];
            let hash_inputs = hash_inputs!(s);

            if let BasicValue::String { value } = s {
                let res = BasicValue::Path { value };
                set_cache_output!((
                    output_names.first().unwrap(),
                    res,
                    hash_inputs
                ));
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
                set_cache_output!((
                    output_names.first().unwrap(),
                    res,
                    hash_inputs
                ));
            } else {
                anyhow::bail!("Not a path");
            }
        },
        | NodeInstruction::System => {
            let s = retrieve_from_cache![1];
            let hash_inputs = hash_inputs!(s);

            if let BasicValue::String { value } = s {
                let res = grammar_separated::grammar::SystemParser::new()
                    .parse(&mut translator.lock().unwrap(), &value);
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
                set_cache_output!((
                    output_names.first().unwrap(),
                    res,
                    hash_inputs
                ));
            } else {
                anyhow::bail!("Not a string");
            }
        },
        | NodeInstruction::Statistics => {
            let s = retrieve_from_cache![1];
            let hash_inputs = hash_inputs!(s);

            if let BasicValue::System { value } = s {
                let res = BasicValue::String {
                    value: value.statistics(&translator.lock().unwrap()),
                };
                set_cache_output!((
                    output_names.first().unwrap(),
                    res,
                    hash_inputs
                ));
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
                            Formatter::from(
                                &translator.lock().unwrap(),
                                &limit.1
                            )
                        ),
                    };
                    set_cache_output!((
                        output_names.first().unwrap(),
                        res,
                        hash_inputs
                    ));
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
                            Formatter::from(&translator.lock().unwrap(), &e)
                        ));
                    }
                    let res = BasicValue::String { value: output };
                    set_cache_output!((
                        output_names.first().unwrap(),
                        res,
                        hash_inputs
                    ));
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
                    let s = match translator.lock().unwrap().encode_not_mut(i) {
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
                            Formatter::from(&translator.lock().unwrap(), &e)
                        ));
                    }
                    let res = BasicValue::String { value: output };
                    set_cache_output!((
                        output_names.first().unwrap(),
                        res,
                        hash_inputs
                    ));
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
                set_cache_output!((
                    output_names.first().unwrap(),
                    res,
                    hash_inputs
                ));
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
                    Formatter::from(&translator.lock().unwrap(), &res)
                );
                let res = BasicValue::String { value: output };
                set_cache_output!((
                    output_names.first().unwrap(),
                    res,
                    hash_inputs
                ));
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
                            Formatter::from(&translator.lock().unwrap(), &l)
                        ),
                    };
                    set_cache_output!((
                        output_names.first().unwrap(),
                        res,
                        hash_inputs
                    ));
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
                        .parse(&mut translator.lock().unwrap(), &value)
                    {
                        | Ok(v) => v,
                        | Err(e) =>
                            return Ok(Some(BasicValue::Error {
                                value: helper::reformat_error(e, &value, ctx),
                            })),
                    };
                let res = BasicValue::Experiment { value };
                set_cache_output!((
                    output_names.first().unwrap(),
                    res,
                    hash_inputs
                ));
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
                            Formatter::from(&translator.lock().unwrap(), &l)
                        ),
                    };
                    set_cache_output!((
                        output_names.first().unwrap(),
                        res,
                        hash_inputs
                    ));
                },
                | (BasicValue::System { value: _ }, _) =>
                    anyhow::bail!("Not an experiment"),
                | (_, BasicValue::Experiment { value: _ }) =>
                    anyhow::bail!("Not a system"),
                | (_, _) => anyhow::bail!("Inputs all wrong"),
            }
        },
        | NodeInstruction::BisimilarityKanellakisSmolka => {
            let (graph_1, graph_2, relabel) = retrieve_from_cache![3];
            let hash_inputs = hash_inputs!(graph_1, graph_2, relabel);

            match (graph_1, graph_2, relabel) {
                | (
                    BasicValue::Graph { value: graph_1 },
                    BasicValue::Graph { value: graph_2 },
                    BasicValue::AssertFunction { value: grouping },
                ) => {
                    use execution::data::MapEdges;
                    let graph_1 = match graph_1
                        .map_edges(&grouping, &mut translator.lock().unwrap())
                    {
                        | Ok(g) => g,
                        | Err(e) => anyhow::bail!(e),
                    };
                    let graph_2 = match graph_2
                        .map_edges(&grouping, &mut translator.lock().unwrap())
                    {
                        | Ok(g) => g,
                        | Err(e) => anyhow::bail!(e),
                    };

                    let l = bisimilarity::kanellakis_smolka::bisimilarity(&&graph_1, &&graph_2);
                    let res = BasicValue::String {
                        value: format!("{l}"),
                    };
                    set_cache_output!((
                        output_names.first().unwrap(),
                        res,
                        hash_inputs
                    ));
                },
                | (_, _, _) => anyhow::bail!("Invalid inputs to bisimilarity."),
            }
        },
        | NodeInstruction::AssertFunction => {
            let s = retrieve_from_cache![1];
            let hash_inputs = hash_inputs!(s);

            if let BasicValue::String { value } = s {
                let res = grammar_separated::assert::AssertParser::new()
                    .parse(&mut translator.lock().unwrap(), &value);
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
                match res.typecheck() {
                    | Ok(_) => {},
                    | Err(e) => anyhow::bail!(e),
                };
                let res = BasicValue::AssertFunction { value: *res };
                set_cache_output!((
                    output_names.first().unwrap(),
                    res,
                    hash_inputs
                ));
            } else {
                anyhow::bail!("Not a string");
            }
        },
        | NodeInstruction::BisimilarityPaigeTarjanNoLabels => {
            let (graph_1, graph_2, relabel) = retrieve_from_cache![3];
            let hash_inputs = hash_inputs!(graph_1, graph_2, relabel);

            match (graph_1, graph_2, relabel) {
                | (
                    BasicValue::Graph { value: graph_1 },
                    BasicValue::Graph { value: graph_2 },
                    BasicValue::AssertFunction { value: grouping },
                ) => {
                    use execution::data::MapEdges;
                    let graph_1 = match graph_1
                        .map_edges(&grouping, &mut translator.lock().unwrap())
                    {
                        | Ok(g) => g,
                        | Err(e) => anyhow::bail!(e),
                    };
                    let graph_2 = match graph_2
                        .map_edges(&grouping, &mut translator.lock().unwrap())
                    {
                        | Ok(g) => g,
                        | Err(e) => anyhow::bail!(e),
                    };

                    let l = bisimilarity::paige_tarjan::bisimilarity_ignore_labels(&&graph_1, &&graph_2);
                    let res = BasicValue::String {
                        value: format!("{l}"),
                    };
                    set_cache_output!((
                        output_names.first().unwrap(),
                        res,
                        hash_inputs
                    ));
                },
                | (_, _, _) => anyhow::bail!("Invalid inputs to bisimilarity."),
            }
        },
        | NodeInstruction::BisimilarityPaigeTarjan => {
            let (graph_1, graph_2, relabel) = retrieve_from_cache![3];
            let hash_inputs = hash_inputs!(graph_1, graph_2, relabel);

            match (graph_1, graph_2, relabel) {
                | (
                    BasicValue::Graph { value: graph_1 },
                    BasicValue::Graph { value: graph_2 },
                    BasicValue::AssertFunction { value: grouping },
                ) => {
                    use execution::data::MapEdges;
                    let graph_1 = match graph_1
                        .map_edges(&grouping, &mut translator.lock().unwrap())
                    {
                        | Ok(g) => g,
                        | Err(e) => anyhow::bail!(e),
                    };
                    let graph_2 = match graph_2
                        .map_edges(&grouping, &mut translator.lock().unwrap())
                    {
                        | Ok(g) => g,
                        | Err(e) => anyhow::bail!(e),
                    };

                    let l =
                        bisimilarity::paige_tarjan::bisimilarity(
                            &&graph_1, &&graph_2,
                        );
                    let res = BasicValue::String {
                        value: format!("{l}"),
                    };
                    set_cache_output!((
                        output_names.first().unwrap(),
                        res,
                        hash_inputs
                    ));
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
                set_cache_output!((
                    output_names.first().unwrap(),
                    res,
                    hash_inputs
                ));
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
                    *to_ret = Some(BasicValue::SaveBytes {
                        path,
                        value: value.into(),
                    });
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
                    let current_translator: rsprocess::translator::Translator =
                        translator.lock().unwrap().clone();
                    let arc_translator = Arc::new(current_translator);
                    let modified_graph = input_graph.map(
                        display_node.generate(
                            Arc::clone(&arc_translator),
                            &input_graph,
                        ),
                        display_edge.generate(
                            Arc::clone(&arc_translator),
                            &input_graph,
                        ),
                    );

                    let input_graph = Arc::new(input_graph.to_owned());

                    let node_formatter = color_node.generate(
                        Arc::clone(&input_graph),
                        arc_translator.encode_not_mut("*"),
                    );
                    let edge_formatter =
                        color_edge.generate(Arc::clone(&input_graph));

                    let dot = rsprocess::dot::Dot::with_attr_getters(
                        &modified_graph,
                        &[],
                        &edge_formatter,
                        &node_formatter,
                    );
                    let res = BasicValue::String {
                        value: format!("{dot}"),
                    };
                    set_cache_output!((
                        output_names.first().unwrap(),
                        res,
                        hash_inputs
                    ));
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
                        .parse(&mut translator.lock().unwrap(), &value);
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
                set_cache_output!((
                    output_names.first().unwrap(),
                    res,
                    hash_inputs
                ));
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
                        .parse(&mut translator.lock().unwrap(), &value);
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
                set_cache_output!((
                    output_names.first().unwrap(),
                    res,
                    hash_inputs
                ));
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
                        .parse(&mut translator.lock().unwrap(), &value);
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
                set_cache_output!((
                    output_names.first().unwrap(),
                    res,
                    hash_inputs
                ));
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
                        .parse(&mut translator.lock().unwrap(), &value);
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
                set_cache_output!((
                    output_names.first().unwrap(),
                    res,
                    hash_inputs
                ));
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
                    let current_translator = translator.lock().unwrap().clone();
                    let arc_translator = Arc::new(current_translator);
                    let modified_graph = input_graph.map(
                        display_node.generate(
                            Arc::clone(&arc_translator),
                            &input_graph,
                        ),
                        display_edge.generate(arc_translator, &input_graph),
                    );

                    use petgraph_graphml::GraphMl;
                    let graphml = GraphMl::new(&modified_graph)
                        .pretty_print(true)
                        .export_node_weights_display()
                        .export_edge_weights_display();

                    let res = BasicValue::String {
                        value: format!("{graphml}"),
                    };
                    set_cache_output!((
                        output_names.first().unwrap(),
                        res,
                        hash_inputs
                    ));
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
                            Arc::new(env),
                            set,
                            context,
                            Arc::new(reactions),
                        ),
                    };
                    set_cache_output!((
                        output_names.first().unwrap(),
                        res,
                        hash_inputs
                    ));
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
                    .parse(&mut translator.lock().unwrap(), &value);
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
                set_cache_output!((
                    output_names.first().unwrap(),
                    res,
                    hash_inputs
                ));
            } else {
                anyhow::bail!("Not a string");
            }
        },
        | NodeInstruction::Set => {
            let s = retrieve_from_cache![1];
            let hash_inputs = hash_inputs!(s);

            if let BasicValue::String { value } = s {
                let res = grammar_separated::grammar::SetParser::new()
                    .parse(&mut translator.lock().unwrap(), &value);
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
                set_cache_output!((
                    output_names.first().unwrap(),
                    res,
                    hash_inputs
                ));
            } else {
                anyhow::bail!("Not a string");
            }
        },
        | NodeInstruction::Context => {
            let s = retrieve_from_cache![1];
            let hash_inputs = hash_inputs!(s);

            if let BasicValue::String { value } = s {
                let res = grammar_separated::grammar::ContextParser::new()
                    .parse(&mut translator.lock().unwrap(), &value);
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
                set_cache_output!((
                    output_names.first().unwrap(),
                    res,
                    hash_inputs
                ));
            } else {
                anyhow::bail!("Not a string");
            }
        },
        | NodeInstruction::Reactions => {
            let s = retrieve_from_cache![1];
            let hash_inputs = hash_inputs!(s);

            if let BasicValue::String { value } = s {
                let res = grammar_separated::grammar::ReactionsParser::new()
                    .parse(&mut translator.lock().unwrap(), &value);
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
                set_cache_output!((
                    output_names.first().unwrap(),
                    res,
                    hash_inputs
                ));
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
                set_cache_output!((
                    output_names.first().unwrap(),
                    res,
                    hash_inputs
                ));
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
                            Formatter::from(
                                &translator.lock().unwrap(),
                                &limit.1
                            )
                        ),
                    };
                    set_cache_output!((
                        output_names.first().unwrap(),
                        res,
                        hash_inputs
                    ));
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
                            Formatter::from(&translator.lock().unwrap(), &e)
                        ));
                    }
                    let res = BasicValue::String { value: output };
                    set_cache_output!((
                        output_names.first().unwrap(),
                        res,
                        hash_inputs
                    ));
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
                    let s = match translator.lock().unwrap().encode_not_mut(i) {
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
                            Formatter::from(&translator.lock().unwrap(), &e)
                        ));
                    }
                    let res = BasicValue::String { value: output };
                    set_cache_output!((
                        output_names.first().unwrap(),
                        res,
                        hash_inputs
                    ));
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
                    Formatter::from(&translator.lock().unwrap(), &res)
                );
                let res = BasicValue::String { value: output };
                set_cache_output!((
                    output_names.first().unwrap(),
                    res,
                    hash_inputs
                ));
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
                            Formatter::from(&translator.lock().unwrap(), &l)
                        ),
                    };
                    set_cache_output!((
                        output_names.first().unwrap(),
                        res,
                        hash_inputs
                    ));
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
                            Formatter::from(&translator.lock().unwrap(), &l)
                        ),
                    };
                    set_cache_output!((
                        output_names.first().unwrap(),
                        res,
                        hash_inputs
                    ));
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
                    set_cache_output!((
                        output_names.first().unwrap(),
                        res,
                        hash_inputs
                    ));
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
                    set_cache_output!((
                        output_names.first().unwrap(),
                        res,
                        hash_inputs
                    ));
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
                    set_cache_output!((
                        output_names.first().unwrap(),
                        res,
                        hash_inputs
                    ));
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
                    set_cache_output!((
                        output_names.first().unwrap(),
                        res,
                        hash_inputs
                    ));
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
                    .parse(&mut translator.lock().unwrap(), &value);
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
                set_cache_output!((
                    output_names.first().unwrap(),
                    res,
                    hash_inputs
                ));
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
                set_cache_output!((
                    output_names.first().unwrap(),
                    res,
                    hash_inputs
                ));
            } else {
                anyhow::bail!("Not a string");
            }
        },
        | NodeInstruction::PositiveComposeSystem => {
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
                    BasicValue::PositiveEnvironment { value: env },
                    BasicValue::PositiveSet { value: set },
                    BasicValue::PositiveContext { value: context },
                    BasicValue::PositiveReactions { value: reactions },
                ) => {
                    let res = BasicValue::PositiveSystem {
                        value: rsprocess::system::PositiveSystem::from(
                            Arc::new(env),
                            set,
                            context,
                            Arc::new(reactions),
                        ),
                    };
                    set_cache_output!((
                        output_names.first().unwrap(),
                        res,
                        hash_inputs
                    ));
                },
                | _ => {
                    anyhow::bail!("Values of wrong type");
                },
            }
        },
        | NodeInstruction::DecomposeSystem => {
            let s = retrieve_from_cache![1];
            let hash_inputs = hash_inputs!(s);

            if let BasicValue::System { value } = s {
                let env = value.environment().clone();
                let initial = value.available_entities().clone();
                let context = value.context().clone();
                let reactions = value.reactions().clone();

                let env = BasicValue::Environment { value: env };
                let initial = BasicValue::Set { value: initial };
                let context = BasicValue::Context { value: context };
                let reactions = BasicValue::Reactions { value: reactions };

                set_cache_output!((output_names[0], env, hash_inputs.clone()));
                set_cache_output!((
                    output_names[1],
                    initial,
                    hash_inputs.clone()
                ));
                set_cache_output!((
                    output_names[2],
                    context,
                    hash_inputs.clone()
                ));
                set_cache_output!((output_names[3], reactions, hash_inputs));
            } else {
                anyhow::bail!("Not a system");
            }
        },
        | NodeInstruction::PositiveDecomposeSystem => {
            let s = retrieve_from_cache![1];
            let hash_inputs = hash_inputs!(s);

            if let BasicValue::PositiveSystem { value } = s {
                let env = value.environment().clone();
                let initial = value.available_entities().clone();
                let context = value.context().clone();
                let reactions = value.reactions().clone();

                let env = BasicValue::PositiveEnvironment { value: env };
                let initial = BasicValue::PositiveSet { value: initial };
                let context = BasicValue::PositiveContext { value: context };
                let reactions =
                    BasicValue::PositiveReactions { value: reactions };

                set_cache_output!((output_names[0], env, hash_inputs.clone()));
                set_cache_output!((
                    output_names[1],
                    initial,
                    hash_inputs.clone()
                ));
                set_cache_output!((
                    output_names[2],
                    context,
                    hash_inputs.clone()
                ));
                set_cache_output!((output_names[3], reactions, hash_inputs));
            } else {
                anyhow::bail!("Not a positive system");
            }
        },
        | NodeInstruction::TraceToString => {
            let trace = retrieve_from_cache![1];
            let hash_inputs = hash_inputs!(trace);

            if let BasicValue::Trace { value } = trace {
                let res = BasicValue::String {
                    value: format!(
                        "{}",
                        Formatter::from(&translator.lock().unwrap(), &value)
                    ),
                };
                set_cache_output!((
                    output_names.first().unwrap(),
                    res,
                    hash_inputs
                ));
            } else {
                anyhow::bail!("Not a trace");
            }
        },
        | NodeInstruction::PositiveTraceToString => {
            let trace = retrieve_from_cache![1];
            let hash_inputs = hash_inputs!(trace);

            if let BasicValue::PositiveTrace { value } = trace {
                let res = BasicValue::String {
                    value: format!(
                        "{}",
                        Formatter::from(&translator.lock().unwrap(), &value)
                    ),
                };
                set_cache_output!((
                    output_names.first().unwrap(),
                    res,
                    hash_inputs
                ));
            } else {
                anyhow::bail!("Not a positive trace");
            }
        },
        | NodeInstruction::PositiveEnvironment => {
            let s = retrieve_from_cache![1];
            let hash_inputs = hash_inputs!(s);

            if let BasicValue::String { value } = s {
                let res =
                    grammar_separated::grammar::PositiveEnvironmentParser::new(
                    )
                    .parse(&mut translator.lock().unwrap(), &value);
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
                let res = BasicValue::PositiveEnvironment { value: *env };
                set_cache_output!((
                    output_names.first().unwrap(),
                    res,
                    hash_inputs
                ));
            } else {
                anyhow::bail!("Not a string");
            }
        },
        | NodeInstruction::PositiveContext => {
            let s = retrieve_from_cache![1];
            let hash_inputs = hash_inputs!(s);

            if let BasicValue::String { value } = s {
                let res =
                    grammar_separated::grammar::PositiveContextParser::new()
                        .parse(&mut translator.lock().unwrap(), &value);
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
                let res = BasicValue::PositiveContext { value: context };
                set_cache_output!((
                    output_names.first().unwrap(),
                    res,
                    hash_inputs
                ));
            } else {
                anyhow::bail!("Not a string");
            }
        },
        | NodeInstruction::PositiveReactions => {
            let s = retrieve_from_cache![1];
            let hash_inputs = hash_inputs!(s);

            if let BasicValue::String { value } = s {
                let res =
                    grammar_separated::grammar::PositiveReactionsParser::new()
                        .parse(&mut translator.lock().unwrap(), &value);
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
                let res = BasicValue::PositiveReactions { value: reactions };
                set_cache_output!((
                    output_names.first().unwrap(),
                    res,
                    hash_inputs
                ));
            } else {
                anyhow::bail!("Not a string");
            }
        },
        | NodeInstruction::ToPositiveContext => {
            let s = retrieve_from_cache![1];
            let hash_inputs = hash_inputs!(s);

            if let BasicValue::Context { value } = s {
                let res = BasicValue::PositiveContext {
                    value: value.into(),
                };
                set_cache_output!((
                    output_names.first().unwrap(),
                    res,
                    hash_inputs
                ));
            } else {
                anyhow::bail!("Not a context");
            }
        },
        | NodeInstruction::ToPositiveEnvironment => {
            let s = retrieve_from_cache![1];
            let hash_inputs = hash_inputs!(s);

            if let BasicValue::Environment { value } = s {
                let res = BasicValue::PositiveEnvironment {
                    value: value.into(),
                };
                set_cache_output!((
                    output_names.first().unwrap(),
                    res,
                    hash_inputs
                ));
            } else {
                anyhow::bail!("Not an environment");
            }
        },
        | NodeInstruction::ToPositiveReactions => {
            let s = retrieve_from_cache![1];
            let hash_inputs = hash_inputs!(s);

            if let BasicValue::Reactions { value } = s {
                let res = BasicValue::PositiveReactions {
                    value:
                        rsprocess::reaction::PositiveReaction::from_reactions(
                            &value,
                        ),
                };
                set_cache_output!((
                    output_names.first().unwrap(),
                    res,
                    hash_inputs
                ));
            } else {
                anyhow::bail!("Not reactions");
            }
        },
        | NodeInstruction::OverwriteContextEntities => {
            let (sys, set) = retrieve_from_cache![2];
            let hash_inputs = hash_inputs!(sys, set);

            match (sys, set) {
                | (
                    BasicValue::System { value: sys },
                    BasicValue::Set { value: set },
                ) => {
                    let mut new_sys = sys.clone();
                    new_sys.overwrite_context_elements(set);
                    let res = BasicValue::System { value: new_sys };
                    set_cache_output!((
                        output_names.first().unwrap(),
                        res,
                        hash_inputs
                    ));
                },
                | (BasicValue::System { value: _ }, _) =>
                    anyhow::bail!("Not a set"),
                | (_, BasicValue::Set { value: _ }) =>
                    anyhow::bail!("Not a system"),
                | (_, _) => anyhow::bail!("Inputs all wrong"),
            }
        },
        | NodeInstruction::OverwriteReactionEntities => {
            let (sys, set) = retrieve_from_cache![2];
            let hash_inputs = hash_inputs!(sys, set);

            match (sys, set) {
                | (
                    BasicValue::System { value: sys },
                    BasicValue::Set { value: set },
                ) => {
                    let mut new_sys = sys.clone();
                    new_sys.overwrite_product_elements(set);
                    let res = BasicValue::System { value: new_sys };
                    set_cache_output!((
                        output_names.first().unwrap(),
                        res,
                        hash_inputs
                    ));
                },
                | (BasicValue::System { value: _ }, _) =>
                    anyhow::bail!("Not a set"),
                | (_, BasicValue::Set { value: _ }) =>
                    anyhow::bail!("Not a system"),
                | (_, _) => anyhow::bail!("Inputs all wrong"),
            }
        },
        | NodeInstruction::PositiveOverwriteContextEntities => {
            let (sys, set) = retrieve_from_cache![2];
            let hash_inputs = hash_inputs!(sys, set);

            match (sys, set) {
                | (
                    BasicValue::PositiveSystem { value: sys },
                    BasicValue::PositiveSet { value: set },
                ) => {
                    let mut new_sys = sys.clone();
                    new_sys.overwrite_context_elements(set);
                    let res = BasicValue::PositiveSystem { value: new_sys };
                    set_cache_output!((
                        output_names.first().unwrap(),
                        res,
                        hash_inputs
                    ));
                },
                | (BasicValue::PositiveSystem { value: _ }, _) =>
                    anyhow::bail!("Not a set"),
                | (_, BasicValue::PositiveSet { value: _ }) =>
                    anyhow::bail!("Not a system"),
                | (_, _) => anyhow::bail!("Inputs all wrong"),
            }
        },
        | NodeInstruction::PositiveOverwriteReactionEntities => {
            let (sys, set) = retrieve_from_cache![2];
            let hash_inputs = hash_inputs!(sys, set);

            match (sys, set) {
                | (
                    BasicValue::PositiveSystem { value: sys },
                    BasicValue::PositiveSet { value: set },
                ) => {
                    let mut new_sys = sys.clone();
                    new_sys.overwrite_product_elements(set);
                    let res = BasicValue::PositiveSystem { value: new_sys };
                    set_cache_output!((
                        output_names.first().unwrap(),
                        res,
                        hash_inputs
                    ));
                },
                | (BasicValue::PositiveSystem { value: _ }, _) =>
                    anyhow::bail!("Not a set"),
                | (_, BasicValue::PositiveSet { value: _ }) =>
                    anyhow::bail!("Not a system"),
                | (_, _) => anyhow::bail!("Inputs all wrong"),
            }
        },
        | NodeInstruction::PositiveGraph => {
            let s = retrieve_from_cache![1];
            let hash_inputs = hash_inputs!(s);

            if let BasicValue::PositiveSystem { value } = s {
                let value = match value.digraph() {
                    | Ok(g) => g,
                    | Err(e) => anyhow::bail!(e),
                };
                let res = BasicValue::PositiveGraph { value };
                set_cache_output!((
                    output_names.first().unwrap(),
                    res,
                    hash_inputs
                ));
            } else {
                anyhow::bail!("Not a system");
            }
        },
        | NodeInstruction::GroupFunction => {
            let s = retrieve_from_cache![1];
            let hash_inputs = hash_inputs!(s);

            if let BasicValue::String { value } = s {
                let res = grammar_separated::grouping::GroupParser::new()
                    .parse(&mut translator.lock().unwrap(), &value);
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
                match res.typecheck() {
                    | Ok(_) => {},
                    | Err(e) => anyhow::bail!(e),
                };
                let res = BasicValue::GroupFunction { value: *res };
                set_cache_output!((
                    output_names.first().unwrap(),
                    res,
                    hash_inputs
                ));
            } else {
                anyhow::bail!("Not a string");
            }
        },
        | NodeInstruction::GroupNodes => {
            let (g, grouping) = retrieve_from_cache![2];
            let hash_inputs = hash_inputs!(g, grouping);

            match (g, grouping) {
                | (
                    BasicValue::Graph { value: g },
                    BasicValue::GroupFunction { value: grouping },
                ) => {
                    use execution::data;
                    let mut graph = g.clone();
                    match data::grouping(
                        &mut graph,
                        &grouping,
                        &mut translator.lock().unwrap(),
                    ) {
                        | Ok(_) => {},
                        | Err(e) => anyhow::bail!(e),
                    };

                    let res = BasicValue::Graph { value: graph };
                    set_cache_output!((
                        output_names.first().unwrap(),
                        res,
                        hash_inputs
                    ));
                },
                | (BasicValue::Graph { value: _ }, _) =>
                    anyhow::bail!("Not a group function"),
                | (_, BasicValue::GroupFunction { value: _ }) =>
                    anyhow::bail!("Not a graph"),
                | _ => anyhow::bail!("Inputs all wrong"),
            }
        },
        | NodeInstruction::PositiveAssertFunction => {
            let s = retrieve_from_cache![1];
            let hash_inputs = hash_inputs!(s);

            if let BasicValue::String { value } = s {
                let res =
                    grammar_separated::positive_assert::AssertParser::new()
                        .parse(&mut translator.lock().unwrap(), &value);
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
                match res.typecheck() {
                    | Ok(_) => {},
                    | Err(e) => anyhow::bail!(e),
                };
                let res = BasicValue::PositiveAssertFunction { value: *res };
                set_cache_output!((
                    output_names.first().unwrap(),
                    res,
                    hash_inputs
                ));
            } else {
                anyhow::bail!("Not a string");
            }
        },
        | NodeInstruction::PositiveGroupFunction => {
            let s = retrieve_from_cache![1];
            let hash_inputs = hash_inputs!(s);

            if let BasicValue::String { value } = s {
                let res =
                    grammar_separated::positive_grouping::GroupParser::new()
                        .parse(&mut translator.lock().unwrap(), &value);
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
                match res.typecheck() {
                    | Ok(_) => {},
                    | Err(e) => anyhow::bail!(e),
                };
                let res = BasicValue::PositiveGroupFunction { value: *res };
                set_cache_output!((
                    output_names.first().unwrap(),
                    res,
                    hash_inputs
                ));
            } else {
                anyhow::bail!("Not a string");
            }
        },
        | NodeInstruction::PositiveGroupNodes => {
            let (g, grouping) = retrieve_from_cache![2];
            let hash_inputs = hash_inputs!(g, grouping);

            match (g, grouping) {
                | (
                    BasicValue::PositiveGraph { value: g },
                    BasicValue::PositiveGroupFunction { value: grouping },
                ) => {
                    use execution::data;
                    let mut graph = g.clone();
                    match data::positive_grouping(
                        &mut graph,
                        &grouping,
                        &mut translator.lock().unwrap(),
                    ) {
                        | Ok(_) => {},
                        | Err(e) => anyhow::bail!(e),
                    };

                    let res = BasicValue::PositiveGraph { value: graph };
                    set_cache_output!((
                        output_names.first().unwrap(),
                        res,
                        hash_inputs
                    ));
                },
                | (BasicValue::PositiveGraph { value: _ }, _) =>
                    anyhow::bail!("Not a positive group function"),
                | (_, BasicValue::PositiveGroupFunction { value: _ }) =>
                    anyhow::bail!("Not a positive graph"),
                | _ => anyhow::bail!("Inputs all wrong"),
            }
        },
        | NodeInstruction::PositiveBisimilarityKanellakisSmolka => {
            let (graph_1, graph_2, relabel) = retrieve_from_cache![3];
            let hash_inputs = hash_inputs!(graph_1, graph_2, relabel);

            match (graph_1, graph_2, relabel) {
                | (
                    BasicValue::PositiveGraph { value: graph_1 },
                    BasicValue::PositiveGraph { value: graph_2 },
                    BasicValue::PositiveAssertFunction { value: grouping },
                ) => {
                    use execution::data::PositiveMapEdges;
                    let graph_1 = match graph_1
                        .map_edges(&grouping, &mut translator.lock().unwrap())
                    {
                        | Ok(g) => g,
                        | Err(e) => anyhow::bail!(e),
                    };
                    let graph_2 = match graph_2
                        .map_edges(&grouping, &mut translator.lock().unwrap())
                    {
                        | Ok(g) => g,
                        | Err(e) => anyhow::bail!(e),
                    };

                    let l = bisimilarity::kanellakis_smolka::bisimilarity(&&graph_1, &&graph_2);
                    let res = BasicValue::String {
                        value: format!("{l}"),
                    };
                    set_cache_output!((
                        output_names.first().unwrap(),
                        res,
                        hash_inputs
                    ));
                },
                | (_, _, _) => anyhow::bail!("Invalid inputs to bisimilarity."),
            }
        },
        | NodeInstruction::PositiveBisimilarityPaigeTarjanNoLabels => {
            let (graph_1, graph_2, relabel) = retrieve_from_cache![3];
            let hash_inputs = hash_inputs!(graph_1, graph_2, relabel);

            match (graph_1, graph_2, relabel) {
                | (
                    BasicValue::PositiveGraph { value: graph_1 },
                    BasicValue::PositiveGraph { value: graph_2 },
                    BasicValue::PositiveAssertFunction { value: grouping },
                ) => {
                    use execution::data::PositiveMapEdges;
                    let graph_1 = match graph_1
                        .map_edges(&grouping, &mut translator.lock().unwrap())
                    {
                        | Ok(g) => g,
                        | Err(e) => anyhow::bail!(e),
                    };
                    let graph_2 = match graph_2
                        .map_edges(&grouping, &mut translator.lock().unwrap())
                    {
                        | Ok(g) => g,
                        | Err(e) => anyhow::bail!(e),
                    };

                    let l = bisimilarity::paige_tarjan::bisimilarity_ignore_labels(&&graph_1, &&graph_2);
                    let res = BasicValue::String {
                        value: format!("{l}"),
                    };
                    set_cache_output!((
                        output_names.first().unwrap(),
                        res,
                        hash_inputs
                    ));
                },
                | (_, _, _) => anyhow::bail!("Invalid inputs to bisimilarity."),
            }
        },
        | NodeInstruction::PositiveBisimilarityPaigeTarjan => {
            let (graph_1, graph_2, relabel) = retrieve_from_cache![3];
            let hash_inputs = hash_inputs!(graph_1, graph_2, relabel);

            match (graph_1, graph_2, relabel) {
                | (
                    BasicValue::PositiveGraph { value: graph_1 },
                    BasicValue::PositiveGraph { value: graph_2 },
                    BasicValue::PositiveAssertFunction { value: grouping },
                ) => {
                    use execution::data::PositiveMapEdges;
                    let graph_1 = match graph_1
                        .map_edges(&grouping, &mut translator.lock().unwrap())
                    {
                        | Ok(g) => g,
                        | Err(e) => anyhow::bail!(e),
                    };
                    let graph_2 = match graph_2
                        .map_edges(&grouping, &mut translator.lock().unwrap())
                    {
                        | Ok(g) => g,
                        | Err(e) => anyhow::bail!(e),
                    };

                    let l =
                        bisimilarity::paige_tarjan::bisimilarity(
                            &&graph_1, &&graph_2,
                        );
                    let res = BasicValue::String {
                        value: format!("{l}"),
                    };
                    set_cache_output!((
                        output_names.first().unwrap(),
                        res,
                        hash_inputs
                    ));
                },
                | (_, _, _) => anyhow::bail!("Invalid inputs to bisimilarity."),
            }
        },
        | NodeInstruction::PositiveDot => {
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
                    BasicValue::PositiveGraph { value: input_graph },
                    BasicValue::DisplayNode {
                        value: display_node,
                    },
                    BasicValue::DisplayEdge {
                        value: display_edge,
                    },
                    BasicValue::ColorNode { value: color_node },
                    BasicValue::ColorEdge { value: color_edge },
                ) => {
                    let current_translator = translator.lock().unwrap().clone();
                    let arc_translator = Arc::new(current_translator);
                    let modified_graph = input_graph.map(
                        display_node.generate_positive(
                            Arc::clone(&arc_translator),
                            &input_graph,
                        ),
                        display_edge.generate_positive(
                            Arc::clone(&arc_translator),
                            &input_graph,
                        ),
                    );

                    let input_graph = Arc::new(input_graph.to_owned());

                    let node_formatter = color_node.generate_positive(
                        Arc::clone(&input_graph),
                        arc_translator.encode_not_mut("*"),
                    );
                    let edge_formatter =
                        color_edge.generate_positive(Arc::clone(&input_graph));

                    let dot = rsprocess::dot::Dot::with_attr_getters(
                        &modified_graph,
                        &[],
                        &edge_formatter,
                        &node_formatter,
                    );
                    let res = BasicValue::String {
                        value: format!("{dot}"),
                    };
                    set_cache_output!((
                        output_names.first().unwrap(),
                        res,
                        hash_inputs
                    ));
                },
                | _ => {
                    anyhow::bail!("Values of wrong type");
                },
            }
        },
        | NodeInstruction::PositiveGraphML => {
            let (input_graph, display_node, display_edge) =
                retrieve_from_cache![3];
            let hash_inputs =
                hash_inputs!(input_graph, display_node, display_edge);

            match (input_graph, display_node, display_edge) {
                | (
                    BasicValue::PositiveGraph { value: input_graph },
                    BasicValue::DisplayNode {
                        value: display_node,
                    },
                    BasicValue::DisplayEdge {
                        value: display_edge,
                    },
                ) => {
                    let current_translator = translator.lock().unwrap().clone();
                    let arc_translator = Arc::new(current_translator);
                    let modified_graph = input_graph.map(
                        display_node.generate_positive(
                            Arc::clone(&arc_translator),
                            &input_graph,
                        ),
                        display_edge
                            .generate_positive(arc_translator, &input_graph),
                    );

                    use petgraph_graphml::GraphMl;
                    let graphml = GraphMl::new(&modified_graph)
                        .pretty_print(true)
                        .export_node_weights_display()
                        .export_edge_weights_display();

                    let res = BasicValue::String {
                        value: format!("{graphml}"),
                    };
                    set_cache_output!((
                        output_names.first().unwrap(),
                        res,
                        hash_inputs
                    ));
                },
                | _ => {
                    anyhow::bail!("Values of wrong type");
                },
            }
        },
        | NodeInstruction::Sleep => {
            #[cfg(not(target_arch = "wasm32"))]
            {
                let input_seconds = retrieve_from_cache![1];
                let hash_inputs = hash_inputs!(input_seconds);

                if let BasicValue::PositiveInt { value: _value } = input_seconds
                {
                    std::thread::sleep(std::time::Duration::from_secs(
                        _value as u64,
                    ));

                    set_cache_output!((
                        output_names.first().unwrap(),
                        input_seconds,
                        hash_inputs
                    ));
                } else {
                    anyhow::bail!("Not an integer");
                }
            }
            #[cfg(target_arch = "wasm32")]
            {
                anyhow::bail!("Cannot sleep on wams");
            }
        },
        | NodeInstruction::StringToSvg => {
            let s = retrieve_from_cache![1];
            let hash_inputs = hash_inputs!(s);

            if let BasicValue::String { value } = s {
                let res = match super::svg::Svg::parse_dot_string(&value) {
                    | Ok(svg) => svg,
                    | Err(e) => anyhow::bail!(e),
                };

                let res = BasicValue::Svg { value: res };
                set_cache_output!((
                    output_names.first().unwrap(),
                    res,
                    hash_inputs
                ));
            } else {
                anyhow::bail!("Not a string");
            }
        },
        | NodeInstruction::SaveSvg => {
            let (path, svg) = retrieve_from_cache![2];

            match (path, svg) {
                | (
                    BasicValue::Path { value: path },
                    BasicValue::Svg { value },
                ) => {
                    let mut path = path.to_string();
                    if !path.ends_with(".svg") {
                        path.push_str(".svg");
                    }
                    let svg = match value.svg() {
                        | Ok(svg) => svg,
                        | Err(e) => anyhow::bail!(e),
                    };
                    *to_ret = Some(BasicValue::SaveBytes { path, value: svg });
                },
                | (BasicValue::Path { .. }, _) => {
                    anyhow::bail!("Not an svg");
                },
                | (_, BasicValue::Svg { .. }) => {
                    anyhow::bail!("Not a path");
                },
                | (_, _) => {
                    anyhow::bail!("Values of wrong type");
                },
            }
        },
        | NodeInstruction::SaveRasterization => {
            let (path, svg) = retrieve_from_cache![2];

            match (path, svg) {
                | (
                    BasicValue::Path { value: path },
                    BasicValue::Svg { value },
                ) => {
                    let mut path = path.to_string();
                    if !path.ends_with(".png") {
                        path.push_str(".png");
                    }
                    let svg = match value.rasterize() {
                        | Ok(svg) => svg,
                        | Err(e) => anyhow::bail!(e),
                    };
                    *to_ret = Some(BasicValue::SaveBytes { path, value: svg });
                },
                | (BasicValue::Path { .. }, _) => {
                    anyhow::bail!("Not an svg");
                },
                | (_, BasicValue::Svg { .. }) => {
                    anyhow::bail!("Not a path");
                },
                | (_, _) => {
                    anyhow::bail!("Values of wrong type");
                },
            }
        },
        | NodeInstruction::ExecuteCommand => {
            let (command, value) = retrieve_from_cache![2];
            let hash_inputs = hash_inputs!(command, value);

            match (command, value) {
                | (
                    BasicValue::String { value: command },
                    BasicValue::String { value },
                ) => {
                    use std::process::{Command, Stdio};
                    use std::io::Write;

                    if command.is_empty() {
                        anyhow::bail!("Empty command");
                    }

                    // Parse the input into command and arguments
                    let mut parts = command.split_whitespace();
                    let command = parts.next().unwrap();
                    let args: Vec<&str> = parts.collect();

                    // build the command
                    let mut cmd = Command::new(command);
                    cmd.args(&args);
                    cmd.stdin(Stdio::piped());
                    cmd.stdout(Stdio::piped());
                    cmd.stderr(Stdio::piped());

                    // execute the command
                    let child = cmd.spawn();

                    let value = match child {
                        Ok(mut child) => {
                            // provide stdin
                            let mut stdin = match child.stdin.take() {
                                Some(o) => o,
                                None => anyhow::bail!("Failed to open stdin")
                            };
                            let value = value.clone();
                            std::thread::spawn(move || {
                                stdin.write_all(value.as_bytes())
                                    .expect("Failed to write to stdin");
                            });

                            match child.wait_with_output() {
                                Ok(output) => {
                                    if output.status.success() {
                                        String::from_utf8_lossy(&output.stdout).to_string()
                                    } else {
                                        String::from_utf8_lossy(&output.stderr).to_string()
                                    }
                                }
                                Err(e) => {
                                    anyhow::bail!("Failed to wait for command \
                                                   '{}': {}", command, e);
                                }
                            }
                        },
                        Err(e) => {
                            anyhow::bail!("Failed to execute command {}: {}",
                                          command, e);
                        }
                    };

                    let ret = BasicValue::String { value };

                    set_cache_output!((
                        output_names.first().unwrap(),
                        ret,
                        hash_inputs
                    ));
                },
                | (BasicValue::String { .. }, _) => {
                    anyhow::bail!("Not a string");
                },
                | (_, BasicValue::String { .. }) => {
                    anyhow::bail!("Not a string");
                },
                | (_, _) => {
                    anyhow::bail!("Values of wrong type");
                },
            }
        }
    }
    Ok(None)
}
