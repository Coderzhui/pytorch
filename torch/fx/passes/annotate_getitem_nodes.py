import operator
from typing import cast

import torch
from torch.fx.node import Node


def annotate_getitem_nodes(graph: torch.fx.Graph) -> None:
    """
    Annotate the type of getitem nodes, inferred from the type of sequence node.
    If sequence node is not annotated with a type, do nothing.
    Currently support getitem nodes from tuple, list, and NamedTuple sequence node.

    This is helpful since annotations on local names within function are lost during FX transforms.
    Adding back known type annotation for getitem nodes to improve jit scriptability.

    Args:
        graph (Graph): The graph to be annotated
    """
    for node in graph.nodes:
        if node.target is operator.getitem:
            sequence_node = cast(Node, node.args[0])
            index_node = cast(int, node.args[1])
            if not sequence_node.type:
                continue
            # container types
            if hasattr(sequence_node.type, "_name"):
                parameterized_types = (
                    sequence_node.type.__args__
                )  # pyrefly: ignore[missing-attribute]
                if (
                    sequence_node.type._name == "Tuple"
                ):  # pyrefly: ignore[missing-attribute]
                    if len(parameterized_types) == 2 and isinstance(
                        parameterized_types[1], type(...)
                    ):
                        node.type = parameterized_types[0]
                    else:
                        if len(parameterized_types) <= index_node:
                            raise AssertionError(
                                f"Index {index_node} out of range for parameterized_types "
                                f"(len={len(parameterized_types)})"
                            )
                        node_type = parameterized_types[index_node]
                        node.type = node_type
                elif sequence_node.type._name == "List":
                    if len(parameterized_types) != 1:
                        raise AssertionError(
                            f"Expected 1 parameterized type, got {len(parameterized_types)}"
                        )
                    node.type = parameterized_types[0]
            # Generic Alias Type
            elif hasattr(sequence_node.type, "__origin__"):
                parameterized_types = (
                    sequence_node.type.__args__
                )  # pyrefly: ignore[missing-attribute]
                if sequence_node.type.__origin__ is tuple:
                    if len(parameterized_types) == 2 and isinstance(
                        parameterized_types[1], type(...)
                    ):
                        node.type = parameterized_types[0]
                    else:
                        if len(parameterized_types) <= index_node:
                            raise AssertionError(
                                f"Index {index_node} out of range for parameterized_types "
                                f"(len={len(parameterized_types)})"
                            )
                        node_type = parameterized_types[index_node]
                        node.type = node_type
                elif sequence_node.type.__origin__ is list:
                    if len(parameterized_types) != 1:
                        raise AssertionError(
                            f"Expected 1 parameterized type, got {len(parameterized_types)}"
                        )
                    node.type = parameterized_types[0]
            # NamedTuple type
            elif hasattr(sequence_node.type, "__annotations__"):
                if sequence_node.type == torch.Tensor:
                    continue
                sequence_node_field_types = sequence_node.type.__annotations__
                field_name = sequence_node.type._fields[
                    index_node
                ]  # pyrefly: ignore[missing-attribute]
                node.type = sequence_node_field_types[field_name]
