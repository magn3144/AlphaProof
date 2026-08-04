from typing import Any

from alphaproof.core.game import Node, action_to_tactic


def serialize_search_tree(root: Node) -> dict[str, Any]:
    """Serialize an AlphaProof root node as a flat JSON search tree."""
    nodes: list[dict[str, Any]] = []

    def serialize_node(node: Node) -> int:
        node_id = len(nodes)
        nodes.append({})
        children = [
            {
                'action': action_to_tactic(action),
                'node_id': serialize_node(child),
            }
            for action, child in node.children.items()
        ]
        nodes[node_id] = {
            'id': node_id,
            'state_id': node.state_id,
            'node_type': node.node_type.name,
            'observation': str(node.observation),
            'prior': node.prior,
            'reward': node.reward,
            'visit_count': node.visit_count,
            'evaluations': node.evaluations,
            'expansion': node.expansion,
            'simulation': node.simulation,
            'seconds': node.expansion_seconds,
            'num_tactics': node.num_tactics,
            'value_sum': node.value_sum,
            'value': node.value(),
            'terminal': node.is_terminal,
            'proven': node.is_optimal,
            'children': children,
        }
        return node_id

    root_id = serialize_node(root)
    return {'root_id': root_id, 'nodes': nodes}
