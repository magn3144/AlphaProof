import asyncio
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from threading import Barrier, Condition, Lock, Semaphore, Thread, local
from time import perf_counter
from typing import Protocol

from alphaproof.core.actors import ObjectiveRejection, play_game
from alphaproof.core.config import Config
from alphaproof.core.environment import Environment
from alphaproof.core.game import Game
from alphaproof.core.network import Network, NetworkSamplingOutput


@dataclass(frozen=True)
class SearchRequest:
    """One independent theorem-search attempt."""

    request_id: str
    theorem: str
    disprove: bool
    num_simulations: int
    stop_on_solution: bool


@dataclass
class SearchResult:
    """Result of one independent theorem-search attempt."""

    request: SearchRequest
    game: Game
    rejection: ObjectiveRejection | None
    duration_seconds: float


@dataclass(frozen=True)
class InferenceBatchStats:
    """Aggregate measurements for the shared inference batcher."""

    batch_count: int
    request_count: int
    average_batch_size: float
    queue_wait_seconds: float
    model_seconds: float


@dataclass
class _InferenceRequest:
    observation: str
    queued_at: float
    future: Future[NetworkSamplingOutput]


class InferenceBatchLogger(Protocol):
    """Logger notified after each model inference batch."""

    def log_inference_batch(self, batch_size: int) -> None:
        """Log the number of requests processed in one batch."""


class _InferenceBatcher:
    """Owns the GPU model and batch synchronous requests from search workers."""

    def __init__(
        self,
        network: Network,
        max_batch_size: int,
        timeout: float,
        logger: InferenceBatchLogger | None,
    ):
        if max_batch_size < 1:
            raise ValueError('Inference batch size must be positive.')
        if timeout < 0:
            raise ValueError('Inference batch timeout cannot be negative.')
        self.network = network
        self.max_batch_size = max_batch_size
        self.timeout = timeout
        self.logger = logger
        self._condition = Condition()
        self._pending: deque[_InferenceRequest] = deque()
        self._paused = False
        self._running = False
        self._closed = False
        self._error: BaseException | None = None
        self._batch_count = 0
        self._request_count = 0
        self._queue_wait_seconds = 0.0
        self._model_seconds = 0.0
        self._thread = Thread(
            target=self._run,
            name='AlphaProofInferenceBatcher',
            daemon=True,
        )
        self._thread.start()

    def sample(self, observation: str) -> NetworkSamplingOutput:
        """Queue one state and wait for its batched network result."""
        future: Future[NetworkSamplingOutput] = Future()
        request = _InferenceRequest(observation, perf_counter(), future)
        with self._condition:
            if self._error is not None:
                raise self._error
            if self._closed:
                raise RuntimeError('Inference batcher is closed.')
            self._pending.append(request)
            self._condition.notify()
        return future.result()

    def close(self) -> None:
        """Stop the batch thread after all searches have drained."""
        with self._condition:
            if self._closed:
                return
            self._closed = True
            error = RuntimeError('Inference batcher closed with pending requests.')
            while self._pending:
                self._pending.popleft().future.set_exception(error)
            self._condition.notify_all()
        self._thread.join()

    def pause(self) -> None:
        """Pause before the next model batch and wait for inference to stop."""
        with self._condition:
            self._paused = True
            self._condition.notify_all()
            while self._running:
                self._condition.wait()

    def resume(self) -> None:
        """Allow queued inference requests to continue."""
        with self._condition:
            resumed_at = perf_counter()
            for request in self._pending:
                request.queued_at = resumed_at
            self._paused = False
            self._condition.notify_all()

    def stats(self) -> InferenceBatchStats:
        """Return aggregate batching measurements."""
        with self._condition:
            return self._stats()

    def take_stats(self) -> InferenceBatchStats:
        """Return and clear measurements for one completed search phase."""
        with self._condition:
            stats = self._stats()
            self._batch_count = 0
            self._request_count = 0
            self._queue_wait_seconds = 0.0
            self._model_seconds = 0.0
            return stats

    def _stats(self) -> InferenceBatchStats:
        average = (
            self._request_count / self._batch_count
            if self._batch_count
            else 0.0
        )
        return InferenceBatchStats(
            batch_count=self._batch_count,
            request_count=self._request_count,
            average_batch_size=average,
            queue_wait_seconds=self._queue_wait_seconds,
            model_seconds=self._model_seconds,
        )

    def _run(self) -> None:
        while True:
            with self._condition:
                while (not self._pending or self._paused) and not self._closed:
                    self._condition.wait()
                if self._closed:
                    return

                deadline = self._pending[0].queued_at + self.timeout
                while len(self._pending) < self.max_batch_size:
                    remaining = deadline - perf_counter()
                    if remaining <= 0:
                        break
                    self._condition.wait(remaining)
                    if self._closed:
                        return
                    if self._paused:
                        break

                if self._paused:
                    continue

                batch = [
                    self._pending.popleft()
                    for _ in range(min(self.max_batch_size, len(self._pending)))
                ]
                self._running = True
                started = perf_counter()
                self._queue_wait_seconds += sum(
                    started - request.queued_at for request in batch
                )

            try:
                model_started = perf_counter()
                outputs = self.network.sample_batch([
                    request.observation for request in batch
                ])
                model_seconds = perf_counter() - model_started
                if self.logger is not None:
                    self.logger.log_inference_batch(len(batch))
            except BaseException as error:
                self._fail(error, batch)
                return

            with self._condition:
                self._running = False
                self._batch_count += 1
                self._request_count += len(batch)
                self._model_seconds += model_seconds
                self._condition.notify_all()
            for request, output in zip(batch, outputs):
                request.future.set_result(output)

    def _fail(
        self,
        error: BaseException,
        batch: list[_InferenceRequest],
    ) -> None:
        with self._condition:
            self._running = False
            self._error = error
            for request in batch:
                request.future.set_exception(error)
            while self._pending:
                self._pending.popleft().future.set_exception(error)
            self._condition.notify_all()


@dataclass
class _SearchWorker:
    """Thread-affine Lean resources reused across proof games."""

    event_loop: asyncio.AbstractEventLoop
    environment: Environment

    def close(self) -> None:
        """Close Lean resources and the owning thread's event loop."""
        self.environment.close()
        asyncio.set_event_loop(None)
        self.event_loop.close()


class ParallelSearchEngine:
    """Run independent MCTS searches around one batched GPU model."""

    def __init__(
        self,
        config: Config,
        network: Network,
        inference_logger: InferenceBatchLogger | None,
    ):
        if config.num_actors < 1:
            raise ValueError('Parallel search count must be positive.')
        if config.max_concurrent_lean_imports < 1:
            raise ValueError('Lean import concurrency must be positive.')
        self.config = config
        self.parallel_searches = config.num_actors
        self._startup_semaphore = Semaphore(config.max_concurrent_lean_imports)
        self._worker_local = local()
        self._workers: list[_SearchWorker] = []
        self._workers_lock = Lock()
        self._closed = False
        self._continuous_active: set[Future[SearchResult]] = set()
        self._continuous_completed: deque[SearchResult] = deque()
        self._batcher = _InferenceBatcher(
            network,
            min(config.inference_batch_size, config.num_actors),
            config.inference_batch_timeout,
            inference_logger,
        )
        self._executor = ThreadPoolExecutor(
            max_workers=self.parallel_searches,
            thread_name_prefix='AlphaProofSearch',
            initializer=self._initialize_worker,
        )

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        self.close()

    @property
    def inference_stats(self) -> InferenceBatchStats:
        """Return aggregate inference measurements for this engine."""
        return self._batcher.stats()

    def take_inference_stats(self) -> InferenceBatchStats:
        """Return and clear inference measurements for the completed phase."""
        return self._batcher.take_stats()

    def pause(self) -> None:
        """Pause model inference while preserving in-flight searches."""
        self._batcher.pause()

    def resume(self) -> None:
        """Resume model inference for preserved searches."""
        self._batcher.resume()

    def close(self) -> None:
        """Release persistent actor resources and the inference thread."""
        if self._closed:
            return
        self._closed = True
        try:
            self._batcher.close()
        finally:
            try:
                self._close_workers()
            finally:
                self._executor.shutdown(wait=True, cancel_futures=True)

    def _initialize_worker(self) -> None:
        """Create one reusable Lean environment on an executor thread."""
        event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(event_loop)
        environment: Environment | None = None
        try:
            with self._startup_semaphore:
                environment = self.config.environment_ctor()
                environment.initialize()
            worker = _SearchWorker(event_loop, environment)
        except BaseException:
            if environment is not None:
                environment.close()
            asyncio.set_event_loop(None)
            event_loop.close()
            raise
        self._worker_local.worker = worker
        with self._workers_lock:
            self._workers.append(worker)

    def _close_workers(self) -> None:
        """Close every worker from the thread that owns its event loop."""
        with self._workers_lock:
            num_workers = len(self._workers)
        if num_workers == 0:
            return

        barrier = Barrier(num_workers)

        def close_worker() -> None:
            barrier.wait()
            self._current_worker().close()

        futures = [
            self._executor.submit(close_worker)
            for _ in range(num_workers)
        ]
        for future in futures:
            future.result()

    def _current_worker(self) -> _SearchWorker:
        worker = getattr(self._worker_local, 'worker', None)
        if not isinstance(worker, _SearchWorker):
            raise RuntimeError('Search worker was not initialized.')
        return worker

    def search(self, requests: list[SearchRequest]) -> list[SearchResult]:
        """Run requests with bounded concurrency and preserve input order."""
        if not requests:
            return []

        results: list[SearchResult | None] = [None] * len(requests)
        active: dict[Future[SearchResult], int] = {}
        next_index = 0
        while next_index < min(self.parallel_searches, len(requests)):
            future = self._executor.submit(
                self._search_one,
                requests[next_index],
            )
            active[future] = next_index
            next_index += 1

        try:
            while active:
                done, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in done:
                    results[active.pop(future)] = future.result()
                    if next_index < len(requests):
                        next_future = self._executor.submit(
                            self._search_one,
                            requests[next_index],
                        )
                        active[next_future] = next_index
                        next_index += 1
        except BaseException:
            for future in active:
                future.cancel()
            wait(active)
            raise

        if any(result is None for result in results):
            raise RuntimeError('Parallel search completed without every result.')
        return [result for result in results if result is not None]

    @property
    def num_searches(self) -> int:
        """Return the number of running or completed continuous searches."""
        return len(self._continuous_active) + len(self._continuous_completed)

    def submit(self, request: SearchRequest) -> None:
        """Submit one continuous search without exceeding actor capacity."""
        if self.num_searches >= self.parallel_searches:
            raise RuntimeError('All parallel search slots are occupied.')
        self._continuous_active.add(
            self._executor.submit(self._search_one, request)
        )

    def next_result(self) -> SearchResult:
        """Return the next continuous result, retaining other completed results."""
        if not self._continuous_completed:
            if not self._continuous_active:
                raise RuntimeError('No continuous searches have been submitted.')
            done, _ = wait(
                self._continuous_active,
                return_when=FIRST_COMPLETED,
            )
            for future in done:
                self._continuous_active.remove(future)
                self._continuous_completed.append(future.result())
        return self._continuous_completed.popleft()

    def _search_one(self, request: SearchRequest) -> SearchResult:
        started = perf_counter()
        worker = self._current_worker()
        game = Game(
            request.theorem,
            request.disprove,
            request.num_simulations,
        )
        rejection = play_game(
            self.config,
            game,
            self._batcher,
            worker.environment,
            stop_on_solution=request.stop_on_solution,
        )

        duration = perf_counter() - started
        game.timings.total_seconds = duration
        return SearchResult(request, game, rejection, duration)
