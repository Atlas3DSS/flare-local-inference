"""Encode the native resource description using the shared Protobuf schema."""

from . import runtime_package_pb2 as proto


def _relocations(values):
    return [{**value, "kind": value["kind"].upper()} for value in values]


def _blob(value):
    return {"data": bytes(value["data"]), "relocations": _relocations(value["relocations"])}


def encode(document):
    if document["version"] != 3:
        raise ValueError("unsupported runtime package version")
    allocations = []
    for value in document["allocations"]:
        allocation = {**value, "memory": value["memory"].upper(),
                      "relocations": _relocations(value["relocations"])}
        spans = value["data_spans"]
        allocation["data_spans"] = [{"offset": offset, "nbytes": size} for offset, size in spans]
        allocations.append(allocation)
    modules = [{**value, "globals": [
        {**symbol, "relocations": _relocations(symbol["relocations"])}
        for symbol in value.get("globals", ())]} for value in document["modules"]]
    functions = [{**value, "attributes": [{"id": key, "value": setting}
        for key, setting in value["attributes"]]} for value in document["functions"]]
    graphs = []
    for graph in document["graphs"]:
        nodes = []
        for node in graph["nodes"]:
            kind = node["kind"]
            value = {key: value for key, value in node.items() if key != "kind"}
            if kind == "kernel":
                value["arguments"] = [_blob(blob) for blob in node["arguments"]]
                value["attributes"] = [{"id": attribute["id"], "value": _blob(attribute["value"])}
                                       for attribute in node["attributes"]]
            nodes.append({kind: value})
        graphs.append({**graph, "nodes": nodes, "edges": [
            {**edge, "data": bytes(edge["data"])} for edge in graph["edges"]]})
    context = document["context"]
    return proto.Package(version=document["version"], architecture=document["architecture"], allocations=allocations,
        modules=modules, functions=functions, graphs=graphs, reservations=document.get("reservations", ()),
        execution=document.get("execution"), context={**context, "limits": [
            {"id": key, "value": value} for key, value in context["limits"]]}).SerializeToString()
