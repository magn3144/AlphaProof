from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Callable, cast
from urllib.parse import parse_qs, urlparse

import torch

if __package__ in (None, ''):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from alphaproof.core.actors import run_mcts
from alphaproof.core.config import DEFAULT_EXPERIMENT_PATH, Config, load_experiment_config
from alphaproof.core.environment import Action, Environment, NodeType, State
from alphaproof.core.game import (
    Game,
    Node,
    action_to_tactic,
    extract_proof_script,
    final_check,
)
from alphaproof.core.helper import replace_sorry_proof, theorem_for_game
from alphaproof.core.network import Network, NetworkSamplingOutput
from alphaproof.core.paths import LEAN_PROJECT_DIR
from alphaproof.inference.infer import load_network_checkpoint, make_config
from leantree import LeanProject


class SearchCancelled(Exception):
    """Stop an agent search at the next network evaluation."""


class ActionRecord:
    """A visible tactic proposal which may or may not enter the game tree."""

    def __init__(self, parent: Node, action: str = ''):
        self.id = uuid.uuid4().hex
        self.parent = parent
        self.action = action
        self.status = 'pending'
        self.error: str | None = None


class BaseSession:
    """Shared snapshot representation for manual and agent sessions."""

    def __init__(self, theorem: str, mode: str, num_simulations: int):
        self.theorem = theorem
        self.mode = mode
        self.status = 'ready'
        self.error: str | None = None
        self.game = Game(theorem, disprove=False, num_simulations=num_simulations)
        self.actions: dict[int, list[ActionRecord]] = {}
        self.node_ids: dict[int, str] = {}
        self.nodes: dict[str, Node] = {}
        self.lock = threading.RLock()

    def snapshot(self, session_id: str) -> dict[str, Any]:
        """Return the current tree and session status."""
        with self.lock:
            complete = self.status == 'solved'
            return {
                'sessionId': session_id,
                'mode': self.mode,
                'status': self.status,
                'error': self.error,
                'root': self._serialize_state(self.game.root),
                'complete': complete,
                'proofScript': self.proof_script() if complete else None,
            }

    def proof_script(self) -> str:
        """Build a complete Lean declaration from the proven subtree."""
        proof_lines = extract_proof_script(self.game.root)
        theorem = theorem_for_game(self.game.theorem, self.game.disprove)
        declaration = replace_sorry_proof(theorem, proof_lines)
        return f'import Mathlib\n\n{declaration}\n'

    def _serialize_state(
        self,
        node: Node,
        branch_label: str | None = None,
    ) -> dict[str, Any]:
        if node.node_type == NodeType.OR:
            children = [
                self._serialize_action(record)
                for record in self.actions.get(id(node), [])
            ]
        else:
            children = [
                self._serialize_state(child, f'Goal {index + 1}')
                for index, child in enumerate(list(node.children.values()))
            ]

        return {
            'kind': 'state',
            'id': self._node_id(node),
            'nodeType': node.node_type.name,
            'branchLabel': branch_label,
            'observation': 'Proof closed.' if node.is_terminal else str(node.observation),
            'goals': self._serialize_goals(node),
            'terminal': node.is_terminal,
            'proven': node.is_optimal,
            'children': children,
        }

    def _serialize_action(self, record: ActionRecord) -> dict[str, Any]:
        child = self._action_child(record)
        return {
            'kind': 'action',
            'id': record.id,
            'action': record.action,
            'status': record.status,
            'error': record.error,
            'proven': child.is_optimal if child is not None else False,
            'children': [self._serialize_state(child)] if child is not None else [],
        }

    def _action_child(self, record: ActionRecord) -> Node | None:
        for action, child in list(record.parent.children.items()):
            if action_to_tactic(action) == record.action:
                return child
        return None

    def _node_id(self, node: Node) -> str:
        key = id(node)
        if key not in self.node_ids:
            node_id = uuid.uuid4().hex
            self.node_ids[key] = node_id
            self.nodes[node_id] = node
        return self.node_ids[key]

    def _serialize_goals(self, node: Node) -> list[dict[str, Any]]:
        return [
            {
                'tag': goal.tag,
                'type': goal.type,
                'hypotheses': [
                    {
                        'name': hypothesis.user_name,
                        'type': hypothesis.type,
                        'value': hypothesis.value,
                    }
                    for hypothesis in goal.hypotheses
                ],
            }
            for goal in node.observation.goals
        ]

    def _find_node(self, state_id: int) -> Node:
        stack = [self.game.root]
        while stack:
            node = stack.pop()
            if node.state_id == state_id:
                return node
            stack.extend(list(node.children.values()))
        raise ValueError(f'Unknown state id: {state_id}')

    def _find_action(self, action_id: str) -> ActionRecord:
        for records in self.actions.values():
            for record in records:
                if record.id == action_id:
                    return record
        raise ValueError(f'Unknown action id: {action_id}')

    def _record_for_action(self, node: Node, action: str) -> ActionRecord:
        records = self.actions.setdefault(id(node), [])
        for record in records:
            if record.action == action:
                return record
        record = ActionRecord(node, action)
        records.append(record)
        return record


class ManualSession(BaseSession):
    """A user-driven proof game backed by the normal Lean environment."""

    def __init__(
        self,
        theorem: str,
        tactic_timeout: float,
        imports: tuple[str, ...] = ('Mathlib',),
        environment: Environment | None = None,
    ):
        super().__init__(theorem, mode='manual', num_simulations=0)
        self.tactic_timeout = tactic_timeout
        self.imports = imports
        self.environment = (
            environment
            if environment is not None
            else Environment(LeanProject(str(LEAN_PROJECT_DIR)), imports=imports)
        )
        try:
            state = self.environment.initial_state(theorem)
        except Exception:
            self.environment.close()
            raise
        self.game.root = self._new_node(None, state, NodeType.OR)

    def close(self) -> None:
        self.environment.close()

    def create_action(self, node_id: str) -> None:
        """Create an empty, saved tactic proposal below an OR state."""
        with self.lock:
            node = self._node_from_id(node_id)
            if node.node_type != NodeType.OR or node.is_terminal:
                raise ValueError('Actions can only be added to open proof goals.')
            self.actions.setdefault(id(node), []).append(ActionRecord(node))

    def update_action(self, action_id: str, action: str) -> None:
        """Update an unexecuted tactic proposal."""
        with self.lock:
            record = self._find_action(action_id)
            if record.status not in ('pending', 'invalid'):
                raise ValueError('Executed tactics cannot be edited.')
            action = action.strip()
            if action and any(
                other is not record and other.action == action
                for other in self.actions[id(record.parent)]
            ):
                raise ValueError('This tactic already exists for the goal.')
            record.action = action
            record.status = 'pending'
            record.error = None

    def execute_action(self, action_id: str) -> None:
        """Execute a saved tactic and attach its result to the game tree."""
        with self.lock:
            record = self._find_action(action_id)
            if record.status not in ('pending', 'invalid'):
                raise ValueError('This tactic has already been executed.')
            if not record.action:
                raise ValueError('Enter a tactic before running it.')
            record.status = 'running'
            record.error = None

        try:
            state = self.environment.step(
                record.parent.state_id,
                record.action,
                self.tactic_timeout,
            )
        except ValueError as exc:
            with self.lock:
                record.status = 'invalid'
                record.error = str(exc)
            return

        with self.lock:
            self._attach_state(record.parent, record.action, state)
            record.status = 'succeeded'
            self._refresh_optimal(self.game.root)
            self.status = 'solved' if self.game.root.is_optimal else 'ready'

    def _attach_state(self, node: Node, action: str, state: State) -> None:
        node_type = NodeType.AND if state.num_goals > 1 else NodeType.OR
        child = self._new_node(action, state, node_type)
        node.children[action] = child
        if node_type == NodeType.AND:
            for index in range(state.num_goals):
                focus_action = f'focus_goal {index}'
                focus_state = self.environment.step(
                    state.id,
                    focus_action,
                    self.tactic_timeout,
                )
                child.children[focus_action] = self._new_node(
                    focus_action,
                    focus_state,
                    NodeType.OR,
                )

    def _new_node(
        self,
        action: str | None,
        state: State,
        node_type: NodeType,
    ) -> Node:
        return Node(
            action=action,
            observation=state.observation,
            prior=1.0,
            state_id=state.id,
            node_type=node_type,
            reward=state.reward,
            is_optimal=state.terminal,
            is_terminal=state.terminal,
        )

    def _refresh_optimal(self, node: Node) -> bool:
        if node.is_terminal:
            node.is_optimal = True
        elif node.node_type == NodeType.OR:
            child_values = [
                self._refresh_optimal(child)
                for child in list(node.children.values())
            ]
            node.is_optimal = any(child_values)
        else:
            children = list(node.children.values())
            node.is_optimal = bool(children) and all(
                self._refresh_optimal(child) for child in children
            )
        return node.is_optimal

    def _node_from_id(self, node_id: str) -> Node:
        try:
            return self.nodes[node_id]
        except KeyError as exc:
            raise ValueError(f'Unknown node id: {node_id}') from exc

    def proof_script(self) -> str:
        proof_lines = extract_proof_script(self.game.root)
        theorem = theorem_for_game(self.game.theorem, self.game.disprove)
        declaration = replace_sorry_proof(theorem, proof_lines)
        imports = '\n'.join(f'import {module}' for module in self.imports)
        prefix = f'{imports}\n\n' if imports else ''
        return f'{prefix}{declaration}\n'


class ObservedEnvironment:
    """Record agent tactic attempts without changing environment behavior."""

    def __init__(self, session: AgentSession, environment: Environment):
        self.session = session
        self.environment = environment

    def step(
        self,
        state_id: int,
        action: Action,
        tactic_timeout: float,
    ) -> State:
        tactic = action_to_tactic(action)
        with self.session.lock:
            parent = self.session._find_node(state_id)
            if parent.node_type == NodeType.OR:
                record = self.session._record_for_action(parent, tactic)
                record.status = 'running'
                record.error = None
            else:
                record = None

        try:
            state = self.environment.step(state_id, action, tactic_timeout)
        except ValueError as exc:
            if record is not None:
                with self.session.lock:
                    record.status = 'invalid'
                    record.error = str(exc)
            raise

        if record is not None:
            with self.session.lock:
                record.status = 'succeeded'
        return state


class CancellableNetwork:
    """Stop MCTS between simulations when cancellation is requested."""

    def __init__(self, session: AgentSession, network: Network):
        self.session = session
        self.network = network

    def sample(self, observation: str) -> NetworkSamplingOutput:
        if self.session.cancelled.is_set():
            raise SearchCancelled
        return self.network.sample(observation)


class AgentSession(BaseSession):
    """A live view over an otherwise normal inference search."""

    def __init__(
        self,
        theorem: str,
        config: Config,
        network: Network,
        seed: int,
        imports: tuple[str, ...] = ('Mathlib',),
    ):
        super().__init__(theorem, mode='agent', num_simulations=config.num_simulations)
        self.config = config
        self.network = network
        self.seed = seed
        self.cancelled = threading.Event()
        self.imports = imports
        self.initialized = threading.Event()
        self.status = 'running'
        self.thread: threading.Thread | None = None

    def start(self, on_finish: Callable[[], None]) -> None:
        """Start MCTS after the session is visible in the store."""
        self.thread = threading.Thread(
            target=self._run,
            args=(on_finish,),
            name='AlphaProofAgentSearch',
            daemon=True,
        )
        self.thread.start()
        self.initialized.wait()

    def cancel(self) -> None:
        self.cancelled.set()

    def _run(self, on_finish: Callable[[], None]) -> None:
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(event_loop)
        environment: Environment | None = None
        try:
            environment = Environment(
                LeanProject(str(LEAN_PROJECT_DIR)), imports=self.imports
            )
            state = environment.initial_state(self.theorem)
            with self.lock:
                self.game.root = Node(
                    action=None,
                    observation=state.observation,
                    prior=1.0,
                    state_id=state.id,
                    node_type=NodeType.OR,
                    reward=state.reward,
                    is_optimal=state.terminal,
                    is_terminal=state.terminal,
                )
            self.initialized.set()
            run_mcts(
                self.config,
                self.game,
                cast(Network, CancellableNetwork(self, self.network)),
                cast(Environment, ObservedEnvironment(self, environment)),
            )
            if self.cancelled.is_set():
                self.status = 'cancelled'
            elif self.game.root.is_optimal:
                self.status = 'verifying'
                verified = final_check(
                        self.game,
                        environment,
                        self.config.final_check_timeout,
                )
                self.game.root.is_optimal = verified
                if self.cancelled.is_set():
                    self.status = 'cancelled'
                elif verified:
                    self.status = 'solved'
                else:
                    self.status = 'error'
                    self.error = self.game.error
            else:
                self.status = 'exhausted'
        except SearchCancelled:
            self.status = 'cancelled'
        except Exception as exc:
            self.status = 'error'
            self.error = str(exc)
        finally:
            self.initialized.set()
            try:
                if environment is not None:
                    environment.close()
            finally:
                asyncio.set_event_loop(None)
                event_loop.close()
                on_finish()


class AgentRuntime:
    """Network and search settings shared by sequential agent sessions."""

    def __init__(self, config: Config, network: Network, seed: int):
        self.config = config
        self.network = network
        self.seed = seed


class SessionStore:
    """Thread-safe storage for manual and agent proof sessions."""

    def __init__(
        self,
        tactic_timeout: float,
        imports: tuple[str, ...] = ('Mathlib',),
        agent_runtime: AgentRuntime | None = None,
        manual_environment_factory: Callable[[str], Environment] | None = None,
    ):
        self.tactic_timeout = tactic_timeout
        self.imports = imports
        self.agent_runtime = agent_runtime
        self.manual_environment_factory = manual_environment_factory
        self.sessions: dict[str, BaseSession] = {}
        self.active_agent_id: str | None = None
        self.lock = threading.Lock()

    def create_manual(self, theorem: str) -> dict[str, Any]:
        environment = (
            self.manual_environment_factory(theorem)
            if self.manual_environment_factory is not None
            else None
        )
        session = ManualSession(
            theorem,
            self.tactic_timeout,
            self.imports,
            environment,
        )
        return self._store(session)

    def create_agent(self, theorem: str) -> dict[str, Any]:
        if self.agent_runtime is None:
            raise ValueError('The backend was started without an agent model.')
        with self.lock:
            if self.active_agent_id is not None:
                raise ValueError('An agent search is already running.')

        session = AgentSession(
            theorem,
            self.agent_runtime.config,
            self.agent_runtime.network,
            self.agent_runtime.seed,
            self.imports,
        )
        session_id = uuid.uuid4().hex
        with self.lock:
            self.sessions[session_id] = session
            self.active_agent_id = session_id
        session.start(lambda: self._finish_agent(session_id))
        return session.snapshot(session_id)

    def create_action(self, session_id: str, node_id: str) -> dict[str, Any]:
        session = self._manual(session_id)
        session.create_action(node_id)
        return session.snapshot(session_id)

    def update_action(
        self,
        session_id: str,
        action_id: str,
        action: str,
    ) -> dict[str, Any]:
        session = self._manual(session_id)
        session.update_action(action_id, action)
        return session.snapshot(session_id)

    def run_action(self, session_id: str, action_id: str) -> dict[str, Any]:
        session = self._manual(session_id)
        session.execute_action(action_id)
        return session.snapshot(session_id)

    def snapshot(self, session_id: str) -> dict[str, Any]:
        return self.get(session_id).snapshot(session_id)

    def cancel(self, session_id: str) -> dict[str, Any]:
        session = self.get(session_id)
        if not isinstance(session, AgentSession):
            raise ValueError('Only agent searches can be cancelled.')
        session.cancel()
        return session.snapshot(session_id)

    def reset(self, session_id: str) -> dict[str, Any]:
        with self.lock:
            session = self.sessions.get(session_id)
            if isinstance(session, AgentSession) and session.status in (
                'running',
                'verifying',
            ):
                raise ValueError('Cancel the agent search before resetting it.')
            session = self.sessions.pop(session_id, None)
        if isinstance(session, ManualSession):
            session.close()
        return {'ok': True}

    def get(self, session_id: str) -> BaseSession:
        with self.lock:
            session = self.sessions.get(session_id)
        if session is None:
            raise ValueError(f'Unknown session id: {session_id}')
        return session

    def close_all(self) -> None:
        with self.lock:
            sessions = list(self.sessions.values())
            self.sessions.clear()
        for session in sessions:
            if isinstance(session, ManualSession):
                session.close()
            elif isinstance(session, AgentSession):
                session.cancel()

    def _store(self, session: BaseSession) -> dict[str, Any]:
        session_id = uuid.uuid4().hex
        with self.lock:
            self.sessions[session_id] = session
        return session.snapshot(session_id)

    def _manual(self, session_id: str) -> ManualSession:
        session = self.get(session_id)
        if not isinstance(session, ManualSession):
            raise ValueError('This action requires a manual session.')
        return session

    def _finish_agent(self, session_id: str) -> None:
        with self.lock:
            if self.active_agent_id == session_id:
                self.active_agent_id = None


def make_handler(store: SessionStore) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = 'AlphaProofInteractive/0.2'

        def do_OPTIONS(self) -> None:
            self._send_json(200, {'ok': True})

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            try:
                if parsed.path == '/api/health':
                    self._send_json(200, {'ok': True})
                elif parsed.path == '/api/state':
                    query = parse_qs(parsed.query)
                    session_id = query.get('sessionId', [''])[0]
                    self._send_json(200, store.snapshot(session_id))
                else:
                    self._send_json(404, {'error': 'Not found'})
            except Exception as exc:
                self._send_json(400, {'error': str(exc)})

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            try:
                data = self._read_json()
                if parsed.path == '/api/manual/start':
                    self._send_json(200, store.create_manual(self._theorem(data)))
                elif parsed.path == '/api/agent/start':
                    self._send_json(200, store.create_agent(self._theorem(data)))
                elif parsed.path == '/api/action/create':
                    self._send_json(
                        200,
                        store.create_action(
                            str(data.get('sessionId', '')),
                            str(data.get('nodeId', '')),
                        ),
                    )
                elif parsed.path == '/api/action/update':
                    self._send_json(
                        200,
                        store.update_action(
                            str(data.get('sessionId', '')),
                            str(data.get('actionId', '')),
                            str(data.get('action', '')),
                        ),
                    )
                elif parsed.path == '/api/action/run':
                    self._send_json(
                        200,
                        store.run_action(
                            str(data.get('sessionId', '')),
                            str(data.get('actionId', '')),
                        ),
                    )
                elif parsed.path == '/api/cancel':
                    self._send_json(
                        200,
                        store.cancel(str(data.get('sessionId', ''))),
                    )
                elif parsed.path == '/api/reset':
                    self._send_json(
                        200,
                        store.reset(str(data.get('sessionId', ''))),
                    )
                else:
                    self._send_json(404, {'error': 'Not found'})
            except Exception as exc:
                self._send_json(400, {'error': str(exc)})

        def log_message(self, format: str, *args: Any) -> None:
            sys.stderr.write(f'{self.address_string()} - {format % args}\n')

        def _theorem(self, data: dict[str, Any]) -> str:
            theorem = str(data.get('theorem', '')).strip()
            if not theorem:
                raise ValueError('Enter a theorem.')
            if theorem.count('sorry') != 1:
                raise ValueError('The theorem must contain exactly one `sorry`.')
            return theorem

        def _read_json(self) -> dict[str, Any]:
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                return {}
            body = self.rfile.read(content_length)
            return json.loads(body.decode('utf-8'))

        def _send_json(self, status: int, data: dict[str, Any]) -> None:
            payload = json.dumps(data).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.end_headers()
            self.wfile.write(payload)

    return Handler


def parse_imports(imports: str) -> tuple[str, ...]:
    return tuple(module.strip() for module in imports.split(',') if module.strip())


def validate_run_dir(parser: argparse.ArgumentParser, run_dir: Path) -> None:
    if not run_dir.is_dir():
        parser.error(f'Run does not exist: {run_dir}')
    has_sft_params = (run_dir / 'network_params.pt').is_file()
    has_rl_params = any((run_dir / 'checkpoints').glob('step_*.pt'))
    if not has_sft_params and not has_rl_params:
        parser.error(f'Run contains no network parameters: {run_dir}')


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    defaults = load_experiment_config(DEFAULT_EXPERIMENT_PATH).rl
    default_run_dir = defaults.sft_run_dir
    parser = argparse.ArgumentParser(
        description='Run the interactive AlphaProof backend.'
    )
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', default=8000, type=int)
    parser.add_argument(
        '--run-dir',
        type=Path,
        default=default_run_dir,
        help=(
            'SFT or RL run containing trained network parameters '
            f'(default: {default_run_dir}).'
        ),
    )
    parser.add_argument(
        '--num-simulations', type=int, default=defaults.num_simulations
    )
    parser.add_argument(
        '--num-sampled-actions', type=int, default=defaults.num_sampled_actions
    )
    parser.add_argument(
        '--tactic-timeout', type=float, default=defaults.tactic_timeout
    )
    parser.add_argument('--seed', type=int, default=defaults.seed)
    parser.set_defaults(
        parallel_searches=defaults.num_actors,
        inference_batch_size=defaults.inference_batch_size,
        inference_batch_timeout=defaults.inference_batch_timeout,
    )
    parser.add_argument(
        '--imports',
        default='Mathlib',
        help='Comma-separated Lean modules to import before each theorem.',
    )
    args = parser.parse_args(argv)
    validate_run_dir(parser, args.run_dir)
    if args.num_simulations < 1:
        parser.error('--num-simulations must be positive')
    if args.num_sampled_actions < 1:
        parser.error('--num-sampled-actions must be positive')
    if args.tactic_timeout <= 0:
        parser.error('--tactic-timeout must be positive')
    return args


def build_agent_runtime(args: argparse.Namespace) -> AgentRuntime:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = make_config(args)
    network = Network(config)
    checkpoint_path = load_network_checkpoint(args.run_dir, network)
    print(f'Loaded agent checkpoint: {checkpoint_path}')
    return AgentRuntime(config, network, args.seed)


def run_server(
    host: str,
    port: int,
    imports: tuple[str, ...],
    agent_runtime: AgentRuntime,
) -> None:
    store = SessionStore(
        agent_runtime.config.tactic_timeout,
        imports=imports,
        agent_runtime=agent_runtime,
    )
    server = HTTPServer((host, port), make_handler(store))
    print(f'Interactive AlphaProof backend running at http://{host}:{port}')
    try:
        server.serve_forever()
    finally:
        store.close_all()
        server.server_close()


def main() -> None:
    args = parse_args()
    run_server(
        args.host,
        args.port,
        parse_imports(args.imports),
        build_agent_runtime(args),
    )


if __name__ == '__main__':
    main()
