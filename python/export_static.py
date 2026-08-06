#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxsim import simplify

from common import (
    ROOT,
    STATIC_ROOT,
    elem_type_to_numpy,
    fixed_input_shapes,
    iter_model_entries,
    metrics_pass,
    model_key,
    openwakeword_mel_host_postprocess,
    output_metrics,
    sha256,
    write_json,
)


def make_feed(model: onnx.ModelProto, shapes: dict[str, list[int]], seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    feed: dict[str, np.ndarray] = {}
    for value_info in model.graph.input:
        name = value_info.name
        shape = tuple(shapes[name])
        dtype = elem_type_to_numpy(value_info.type.tensor_type.elem_type)
        if name == "sr":
            value = np.asarray(16000, dtype=dtype)
        elif name == "processed_lens":
            value = np.zeros(shape, dtype=dtype)
        elif name == "y":
            value = rng.integers(0, 263, size=shape, dtype=dtype)
        elif dtype == np.dtype(np.bool_):
            value = rng.integers(0, 2, size=shape).astype(dtype)
        elif np.issubdtype(dtype, np.integer):
            value = rng.integers(0, 4, size=shape, dtype=dtype)
        elif name == "input" and shape == (1, 1760):
            value = rng.integers(-4000, 4001, size=shape).astype(dtype)
        elif name in {"h", "c"} or name.startswith("cached_") or name == "embed_states":
            value = rng.normal(0.0, 0.05, size=shape).astype(dtype)
        elif name == "x" and len(shape) == 3 and shape[-1] == 80:
            value = rng.normal(-5.0, 3.0, size=shape).astype(dtype)
        else:
            value = rng.normal(0.0, 0.5, size=shape).astype(dtype)
        feed[name] = value
    return feed


def set_output_shapes(model: onnx.ModelProto, outputs: list[np.ndarray]) -> None:
    for value_info, value in zip(model.graph.output, outputs):
        tensor_shape = value_info.type.tensor_type.shape
        del tensor_shape.dim[:]
        for size in np.asarray(value).shape:
            tensor_shape.dim.add().dim_value = int(size)


def set_input_shapes(model: onnx.ModelProto, shapes: dict[str, list[int]]) -> None:
    for value_info in model.graph.input:
        tensor_shape = value_info.type.tensor_type.shape
        del tensor_shape.dim[:]
        for size in shapes[value_info.name]:
            tensor_shape.dim.add().dim_value = int(size)


def lower_jarvis_verifier_if(model: onnx.ModelProto) -> onnx.ModelProto:
    """Inline Jarvis' verifier branch and replace data-dependent If with Where."""
    graph = model.graph
    if_nodes = [(index, node) for index, node in enumerate(graph.node) if node.op_type == "If"]
    if len(if_nodes) != 1:
        raise ValueError(f"expected one Jarvis If node, found {len(if_nodes)}")
    index, if_node = if_nodes[0]
    branches = {
        attribute.name: attribute.g
        for attribute in if_node.attribute
        if attribute.type == onnx.AttributeProto.GRAPH
    }
    then_graph = branches["then_branch"]
    else_graph = branches["else_branch"]
    if len(else_graph.node) != 1 or else_graph.node[0].op_type != "Identity":
        raise ValueError("unexpected Jarvis else branch")

    existing_initializers = {value.name for value in graph.initializer}
    for initializer in then_graph.initializer:
        if initializer.name in existing_initializers:
            raise ValueError(f"duplicate initializer while lowering If: {initializer.name}")
        graph.initializer.append(copy.deepcopy(initializer))
        existing_initializers.add(initializer.name)

    existing_value_info = {
        value.name
        for value in [*graph.input, *graph.output, *graph.value_info]
    }
    for value_info in [*then_graph.value_info, *then_graph.output]:
        if value_info.name not in existing_value_info:
            graph.value_info.append(copy.deepcopy(value_info))
            existing_value_info.add(value_info.name)

    then_output = then_graph.output[0].name
    else_output = else_graph.node[0].input[0]
    where_node = onnx.helper.make_node(
        "Where",
        [if_node.input[0], then_output, else_output],
        list(if_node.output),
        name=f"{if_node.name}_Where",
    )
    replacement = [copy.deepcopy(node) for node in then_graph.node] + [where_node]
    nodes = list(graph.node)
    del graph.node[:]
    graph.node.extend(nodes[:index] + replacement + nodes[index + 1 :])
    onnx.checker.check_model(model)
    return model


def specialize_sherpa_decoder_valid_tokens(model: onnx.ModelProto) -> onnx.ModelProto:
    """Remove padding-token guards under Sherpa's valid token-id contract."""
    graph = model.graph
    clip = next((node for node in graph.node if node.name == "/decoder/Clip"), None)
    gather = next(
        (node for node in graph.node if node.name == "/decoder/embedding/Gather"),
        None,
    )
    mul = next((node for node in graph.node if node.name == "/decoder/Mul"), None)
    if clip is None or gather is None or mul is None:
        raise ValueError("unexpected Sherpa decoder padding-token subgraph")

    gather.input[1] = graph.input[0].name
    mul_output = mul.output[0]
    gather_output = gather.output[0]
    for node in graph.node:
        for index, name in enumerate(node.input):
            if name == mul_output:
                node.input[index] = gather_output

    remove_names = {
        "/decoder/Clip",
        "/decoder/GreaterOrEqual",
        "/decoder/Unsqueeze",
        "/decoder/Cast_1",
        "/decoder/Mul",
    }
    nodes = [node for node in graph.node if node.name not in remove_names]
    del graph.node[:]
    graph.node.extend(nodes)
    onnx.checker.check_model(model)
    return model


def lower_sherpa_reverse_mask_slice(model: onnx.ModelProto) -> onnx.ModelProto:
    """Replace the fixed 64-element reverse Slice that Pulsar2 mis-infers."""
    graph = model.graph
    reverse_slice = next((node for node in graph.node if node.name == "/Slice_2"), None)
    if reverse_slice is None or reverse_slice.op_type != "Slice":
        raise ValueError("missing Sherpa encoder reverse mask Slice")
    initializer_by_name = {value.name: value for value in graph.initializer}
    steps = onnx.numpy_helper.to_array(initializer_by_name[reverse_slice.input[4]]).reshape(-1)
    axes = onnx.numpy_helper.to_array(initializer_by_name[reverse_slice.input[3]]).reshape(-1)
    if steps.tolist() != [-1] or axes.tolist() != [1]:
        raise ValueError(f"unexpected reverse Slice axes/steps: {axes}, {steps}")

    indices_name = "/Slice_2_reverse_indices"
    graph.initializer.append(
        onnx.numpy_helper.from_array(
            np.arange(63, -1, -1, dtype=np.int64),
            name=indices_name,
        )
    )
    gather = onnx.helper.make_node(
        "Gather",
        [reverse_slice.input[0], indices_name],
        list(reverse_slice.output),
        name="/Slice_2_GatherReverse",
        axis=1,
    )
    nodes = [gather if node is reverse_slice else node for node in graph.node]
    del graph.node[:]
    graph.node.extend(nodes)
    onnx.checker.check_model(model)
    return model


def lower_sherpa_mask_broadcast(model: onnx.ModelProto) -> onnx.ModelProto:
    """Make LessOrEqual's (1, 1) -> (1, 64) broadcast explicit for Pulsar2."""
    graph = model.graph
    compare = next((node for node in graph.node if node.name == "/LessOrEqual"), None)
    if compare is None or compare.op_type != "LessOrEqual":
        raise ValueError("missing Sherpa encoder LessOrEqual mask comparison")
    shape_name = "/LessOrEqual_lhs_shape"
    expanded_name = "/LessOrEqual_lhs_expanded"
    graph.initializer.append(
        onnx.numpy_helper.from_array(
            np.asarray([1, 64], dtype=np.int64),
            name=shape_name,
        )
    )
    expand = onnx.helper.make_node(
        "Expand",
        [compare.input[0], shape_name],
        [expanded_name],
        name="/LessOrEqual_ExpandLhs",
    )
    compare.input[0] = expanded_name
    nodes = list(graph.node)
    index = nodes.index(compare)
    del graph.node[:]
    graph.node.extend(nodes[:index] + [expand] + nodes[index:])
    onnx.checker.check_model(model)
    return model


def lower_openwakeword_mel_log_to_host(model: onnx.ModelProto) -> onnx.ModelProto:
    """End the mel graph before Log; the runtime applies the exact dB tail."""
    graph = model.graph
    logs = [node for node in graph.node if node.op_type == "Log"]
    if len(logs) != 1 or len(graph.output) != 1:
        raise ValueError(
            f"unexpected openWakeWord mel graph: logs={len(logs)}, outputs={len(graph.output)}"
        )
    log = logs[0]
    downstream_tensors = set(log.output)
    remove_names = {log.name}
    changed = True
    while changed:
        changed = False
        for node in graph.node:
            if node.name in remove_names:
                continue
            if any(name in downstream_tensors for name in node.input):
                remove_names.add(node.name)
                downstream_tensors.update(node.output)
                changed = True

    output_name = graph.output[0].name
    identity = onnx.helper.make_node(
        "Identity",
        [log.input[0]],
        [output_name],
        name="HostLogPostprocessInput",
    )
    nodes = [node for node in graph.node if node.name not in remove_names]
    nodes.append(identity)
    del graph.node[:]
    graph.node.extend(nodes)
    used_inputs = {name for node in graph.node for name in node.input}
    initializers = [value for value in graph.initializer if value.name in used_inputs]
    del graph.initializer[:]
    graph.initializer.extend(initializers)
    onnx.checker.check_model(model)
    return model


def restore_sherpa_softplus(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
    """Collapse exported Softplus/Swoosh expansions so no explicit Log remains."""
    graph = model.graph
    nodes = list(graph.node)
    producer = {name: node for node in nodes for name in node.output}
    initializer_names = {value.name for value in graph.initializer}
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in nodes:
        for name in node.input:
            consumers.setdefault(name, []).append(node)

    remove_names: set[str] = set()
    replacements: dict[str, onnx.NodeProto] = {}
    log_nodes = [node for node in nodes if node.op_type == "Log"]
    for log in log_nodes:
        log_input_add = producer.get(log.input[0])
        if log_input_add is None or log_input_add.op_type != "Add":
            raise ValueError(f"unexpected Sherpa Log producer: {log.name}")
        direct_consumers = consumers.get(log.output[0], [])
        where = next((node for node in direct_consumers if node.op_type == "Where"), None)

        if where is not None:
            equal = next(
                (node for node in direct_consumers if node.op_type == "Equal"), None
            )
            if equal is None:
                raise ValueError(f"missing Softplus overflow guard for {log.name}")
            exp = next(
                (
                    producer[name]
                    for name in log_input_add.input
                    if name in producer and producer[name].op_type == "Exp"
                ),
                None,
            )
            if exp is None or exp.op_type != "Exp":
                raise ValueError(f"missing Softplus Exp for {log.name}")
            softplus_input = where.input[1]
            target = where
            remove_names.update(
                {log.name, log_input_add.name, exp.name, equal.name}
            )
        else:
            if len(direct_consumers) != 1 or direct_consumers[0].op_type != "Add":
                raise ValueError(f"unexpected stable Softplus tail for {log.name}")
            target = direct_consumers[0]
            other_input = next(name for name in target.input if name != log.output[0])
            maximum = producer.get(other_input)
            if maximum is None or maximum.op_type != "Max":
                raise ValueError(f"missing stable Softplus Max for {log.name}")
            input_candidates = [
                name
                for name in maximum.input
                if name not in initializer_names
                and (producer.get(name) is None or producer[name].op_type != "Constant")
            ]
            if len(input_candidates) != 1:
                raise ValueError(f"cannot find stable Softplus input for {log.name}")
            softplus_input = input_candidates[0]
            exp = next(
                (
                    producer[name]
                    for name in log_input_add.input
                    if name in producer and producer[name].op_type == "Exp"
                ),
                None,
            )
            if exp is None or exp.op_type != "Exp":
                raise ValueError(f"missing stable Softplus Exp for {log.name}")
            neg = producer.get(exp.input[0])
            absolute = producer.get(neg.input[0]) if neg is not None else None
            reflected = producer.get(absolute.input[0]) if absolute is not None else None
            if (
                neg is None
                or neg.op_type != "Neg"
                or absolute is None
                or absolute.op_type != "Abs"
                or reflected is None
                or reflected.op_type != "Sub"
            ):
                raise ValueError(f"unexpected stable Softplus expansion for {log.name}")
            remove_names.update(
                {
                    log.name,
                    log_input_add.name,
                    exp.name,
                    neg.name,
                    absolute.name,
                    reflected.name,
                    maximum.name,
                }
            )

        replacements[target.name] = onnx.helper.make_node(
            "Softplus",
            [softplus_input],
            list(target.output),
            name=f"{target.name}_NoLogSoftplus",
        )

    rewritten = []
    for node in nodes:
        if node.name in replacements:
            rewritten.append(replacements[node.name])
        elif node.name not in remove_names:
            rewritten.append(node)
    del graph.node[:]
    graph.node.extend(rewritten)
    onnx.checker.check_model(model)
    if any(node.op_type == "Log" for node in graph.node):
        raise RuntimeError("Sherpa Log replacement left explicit Log nodes")
    return model, len(log_nodes)


def describe_session(session: ort.InferenceSession) -> dict[str, list[dict[str, object]]]:
    def describe(values: list[ort.NodeArg]) -> list[dict[str, object]]:
        return [{"name": value.name, "shape": value.shape, "dtype": value.type} for value in values]

    return {"inputs": describe(session.get_inputs()), "outputs": describe(session.get_outputs())}


def make_session(path: Path, disable_optimization: bool) -> ort.InferenceSession:
    options = ort.SessionOptions()
    if disable_optimization:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def export_one(group: str, source: Path, output: Path, seed: int) -> dict[str, object]:
    model = onnx.load(source)
    onnx.checker.check_model(model)
    shapes = fixed_input_shapes(source, model)
    is_zipformer_encoder = group == "sherpa" and source.name.startswith("encoder-")
    check_n = 1 if is_zipformer_encoder else 3
    simplified, check_ok = simplify(
        model,
        overwrite_input_shapes=shapes,
        check_n=check_n,
        include_subgraph=True,
        skip_constant_folding=is_zipformer_encoder,
    )
    if not check_ok:
        raise RuntimeError(f"onnxsim validation failed for {source}")

    # onnxsim 0.4.36 returns a Silero control-flow ModelProto that passes its
    # in-memory check but fails after serialization in ORT 1.26 with
    # HasExternalDataInMemory. Keep the checked original graph and apply only
    # the fixed shape annotations for this optional VAD model.
    serialized_graph_fallback = source.name == "silero_vad.onnx"
    if serialized_graph_fallback:
        # Reload because onnxsim mutates tensor bookkeeping on the input proto
        # even though its public implementation starts from a deepcopy.
        simplified = onnx.load(source)
        set_input_shapes(simplified, shapes)

    lowered_control_flow = source.name == "hey_jarvis_v0.1.onnx"
    if lowered_control_flow:
        simplified = lower_jarvis_verifier_if(simplified)
        simplified, lowered_check_ok = simplify(
            simplified,
            check_n=3,
            include_subgraph=True,
        )
        if not lowered_check_ok:
            raise RuntimeError(f"onnxsim validation failed after If lowering for {source}")

    valid_token_lowering = group == "sherpa" and source.name.startswith("decoder-")
    if valid_token_lowering:
        simplified = specialize_sherpa_decoder_valid_tokens(simplified)
        simplified, decoder_check_ok = simplify(simplified, check_n=3)
        if not decoder_check_ok:
            raise RuntimeError(
                f"onnxsim validation failed after decoder token specialization for {source}"
            )

    reverse_slice_lowering = is_zipformer_encoder
    if reverse_slice_lowering:
        simplified = lower_sherpa_reverse_mask_slice(simplified)
        simplified = lower_sherpa_mask_broadcast(simplified)
        simplified, encoder_lowering_ok = simplify(
            simplified,
            check_n=1,
            skip_constant_folding=True,
        )
        if not encoder_lowering_ok:
            raise RuntimeError(
                f"onnxsim validation failed after reverse Slice lowering for {source}"
            )

    mel_host_postprocess = group == "openwakeword" and source.name == "melspectrogram.onnx"
    if mel_host_postprocess:
        simplified = lower_openwakeword_mel_log_to_host(simplified)

    softplus_replacement_count = 0
    if is_zipformer_encoder:
        simplified, softplus_replacement_count = restore_sherpa_softplus(simplified)
        simplified, softplus_check_ok = simplify(
            simplified,
            check_n=1,
            skip_constant_folding=True,
        )
        if not softplus_check_ok:
            raise RuntimeError(
                f"onnxsim validation failed after Softplus restoration for {source}"
            )
        if any(node.op_type == "Log" for node in simplified.graph.node):
            raise RuntimeError(f"explicit Log remains after Softplus restoration for {source}")

    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(simplified, output)
    feed = make_feed(model, shapes, seed)
    reference_session = make_session(source, serialized_graph_fallback)
    static_session = make_session(output, serialized_graph_fallback)
    reference_outputs = reference_session.run(None, feed)
    static_outputs = static_session.run(None, feed)
    if len(reference_outputs) != len(static_outputs):
        raise RuntimeError(f"output count changed for {source}")

    output_names = [value.name for value in reference_session.get_outputs()]
    comparable_static_outputs = list(static_outputs)
    if mel_host_postprocess:
        comparable_static_outputs[0] = openwakeword_mel_host_postprocess(
            comparable_static_outputs[0]
        )
    metrics = {
        name: output_metrics(reference, candidate)
        for name, reference, candidate in zip(
            output_names, reference_outputs, comparable_static_outputs
        )
    }
    failures = [name for name, value in metrics.items() if not metrics_pass(value)]
    if failures:
        raise RuntimeError(f"static equivalence failed for {source.name}: {failures}")

    static_model = onnx.load(output)
    set_output_shapes(static_model, static_outputs)
    onnx.checker.check_model(static_model)
    onnx.save(static_model, output)
    final_session = make_session(output, serialized_graph_fallback)
    final_outputs = final_session.run(None, feed)
    comparable_final_outputs = list(final_outputs)
    if mel_host_postprocess:
        comparable_final_outputs[0] = openwakeword_mel_host_postprocess(
            comparable_final_outputs[0]
        )
    final_metrics = {
        name: output_metrics(reference, candidate)
        for name, reference, candidate in zip(
            output_names, reference_outputs, comparable_final_outputs
        )
    }
    if any(not metrics_pass(value) for value in final_metrics.values()):
        raise RuntimeError(f"output shape annotation changed computation for {source}")

    float_cosines = [
        value["cosine"] for value in final_metrics.values() if "cosine" in value
    ]
    return {
        "key": model_key(group, output),
        "group": group,
        "source": str(source.relative_to(ROOT)),
        "static": str(output.relative_to(ROOT)),
        "source_sha256": sha256(source),
        "static_sha256": sha256(output),
        "fixed_input_shapes": shapes,
        "onnxsim": {
            "version": __import__("onnxsim").__version__,
            "check_n": check_n,
            "check_passed": bool(check_ok),
            "skip_constant_folding": is_zipformer_encoder,
            "serialized_graph_fallback": serialized_graph_fallback,
            "lowered_control_flow": lowered_control_flow,
            "valid_token_lowering": valid_token_lowering,
            "reverse_slice_lowering": reverse_slice_lowering,
            "mask_broadcast_lowering": reverse_slice_lowering,
            "mel_log_moved_to_host": mel_host_postprocess,
            "softplus_replacement_count": softplus_replacement_count,
            "nodes_before": len(model.graph.node),
            "nodes_after": len(static_model.graph.node),
        },
        "runtime": describe_session(final_session),
        "runtime_notes": {
            "ort_graph_optimization_disabled": serialized_graph_fallback,
            "host_postprocess": (
                "10*log10 plus 80 dB clipping"
                if mel_host_postprocess
                else None
            ),
        },
        "validation": {
            "seed": seed,
            "minimum_output_cosine": min(float_cosines) if float_cosines else None,
            "outputs": final_metrics,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260801)
    args = parser.parse_args()

    os.environ.setdefault("WORK_TMP", str(ROOT / ".work_tmp"))
    os.environ.setdefault("TMPDIR", str(ROOT / ".work_tmp/tmp"))
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".work_tmp/matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".work_tmp/xdg_cache"))
    for name in ("TMPDIR", "MPLCONFIGDIR", "XDG_CACHE_HOME"):
        Path(os.environ[name]).mkdir(parents=True, exist_ok=True)

    records = []
    for index, (group, source, output) in enumerate(iter_model_entries()):
        print(f"[{index + 1:02d}] {group}: {source.name}", flush=True)
        record = export_one(group, source, output, args.seed + index)
        records.append(record)
        print(
            f"     onnxsim ok, nodes {record['onnxsim']['nodes_before']} -> "
            f"{record['onnxsim']['nodes_after']}, min cosine "
            f"{record['validation']['minimum_output_cosine']}",
            flush=True,
        )

    manifest = {
        "environment": "kws-quant",
        "onnx_version": onnx.__version__,
        "onnxruntime_version": ort.__version__,
        "models": records,
    }
    write_json(STATIC_ROOT / "manifest.json", manifest)
    print(f"wrote {STATIC_ROOT / 'manifest.json'}")


if __name__ == "__main__":
    main()
