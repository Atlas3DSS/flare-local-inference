"""Encode bundle indexes and independently readable CUDA graph records."""

import struct
from . import runtime_package_pb2 as proto


def serving(index, server):
    """Encode complete recipe defaults and the structural native frontend contract."""
    from google.protobuf.json_format import ParseDict

    encoder = server["frontend"].get("prompt_encoder", "PROMPT_JINJA")
    if type(encoder) not in (str, int) or encoder not in ("PROMPT_JINJA", "DEEPSEEK_V4", "DEEPSEEK_V41", 0, 1, 2):
        raise ValueError("unsupported prompt encoder")
    expected_assets = {"tokenizer", "chat_template"} if encoder in ("PROMPT_JINJA", 0) else {"tokenizer"}
    if set(server["assets"]) != expected_assets:
        raise ValueError("runtime assets must match the prompt encoder: " + ", ".join(sorted(expected_assets)))
    if any(key in server for key in ("generation_defaults", "eos_token_ids")):
        raise ValueError("sampling and token defaults require typed package fields")
    sampling = server["sampling_defaults"]
    expected = {field.name for field in proto.DefaultSamplingParameters.DESCRIPTOR.fields} - {"ranges"}
    if set(sampling) - {"ranges"} != expected:
        raise ValueError("sampling defaults must explicitly declare every parameter")
    from ignition.recipes.declaration import SamplingDefaults, SamplingRange
    SamplingDefaults(**{name: sampling[name] for name in expected}, ranges={
        name: SamplingRange(**bounds) for name, bounds in sampling.get("ranges", {}).items()
    })
    ParseDict(sampling, index.sampling_defaults)
    ParseDict(server["frontend"], index.frontend)
    ParseDict({"speculative_proposal_source": "mtp", **{key: value for key, value in server.items()
               if key not in ("assets", "sampling_defaults", "frontend")}}, index.contract)


def _relocations(values):
    return [{**value, "kind": value["kind"].upper()} for value in values]


def _blob(value):
    return {"data": bytes(value["data"]), "relocations": _relocations(value["relocations"])}


def node(value):
    kind = value["kind"]
    result = {key: item for key, item in value.items() if key != "kind"}
    if kind == "kernel":
        result["arguments"] = [_blob(blob) for blob in value["arguments"]]
        result["attributes"] = [{"id": attribute["id"], "value": _blob(attribute["value"])}
                                 for attribute in value["attributes"]]
    return proto.Node(**{kind: result})


def write_graph(graph, path):
    with path.open("wb") as output:
        for value in graph["nodes"]:
            data = node(value).SerializeToString()
            output.write(struct.pack("<I", len(data)))
            output.write(data)
        for value in graph["edges"]:
            data = proto.Edge(**{**value, "data": bytes(value["data"])}).SerializeToString()
            output.write(struct.pack("<I", len(data)))
            output.write(data)
    return proto.GraphIndex(name=graph["name"], flags=graph["flags"], resource=path.name,
        node_count=len(graph["nodes"]), edge_count=len(graph["edges"]),
        functions=sorted({n["function"] for n in graph["nodes"] if n["kind"] == "kernel"}))


def bundle(document):
    allocations = []
    for value in document["allocations"]:
        allocation = {**value, "memory": value["memory"].upper(),
                      "relocations": _relocations(value["relocations"])}
        allocation["data_spans"] = [{"offset": offset, "nbytes": size} for offset, size in value["data_spans"]]
        allocations.append(allocation)
    modules = [{**value, "globals": [
        {**symbol, "relocations": _relocations(symbol["relocations"])}
        for symbol in value.get("globals", ())]} for value in document["modules"]]
    functions = [{**value, "attributes": [{"id": key, "value": setting}
        for key, setting in value["attributes"]]} for value in document["functions"]]
    context = document["context"]
    return proto.BundleIndex(architecture=document["architecture"], allocations=allocations,
        modules=modules, functions=functions, graphs=document["graphs"], reservations=document.get("reservations", ()),
        execution=document.get("execution"), context={**context, "limits": [
            {"id": key, "value": value} for key, value in context["limits"]]}, events=document.get("events", ()))
