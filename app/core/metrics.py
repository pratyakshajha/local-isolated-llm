"""Low-cardinality in-process metrics for a single local server instance."""

from __future__ import annotations

from collections import Counter, defaultdict
from threading import Lock


class AppMetrics:
    def __init__(self) -> None:
        self._lock = Lock()
        self._http_requests: Counter[tuple[str, str, int]] = Counter()
        self._http_duration: dict[tuple[str, str], tuple[int, float]] = defaultdict(
            lambda: (0, 0.0)
        )
        self._generation_count: Counter[str] = Counter()
        self._generation_duration: dict[str, tuple[int, float]] = defaultdict(
            lambda: (0, 0.0)
        )

    def observe_http(
        self, method: str, route: str, status: int, duration_seconds: float
    ) -> None:
        key = (method, route)
        with self._lock:
            self._http_requests[(method, route, status)] += 1
            count, total = self._http_duration[key]
            self._http_duration[key] = (count + 1, total + duration_seconds)

    def observe_generation(self, outcome: str, duration_seconds: float) -> None:
        with self._lock:
            self._generation_count[outcome] += 1
            count, total = self._generation_duration[outcome]
            self._generation_duration[outcome] = (count + 1, total + duration_seconds)

    def render(self, model_loaded: bool, active_sessions: int, resident_memory: int) -> str:
        with self._lock:
            http_requests = dict(self._http_requests)
            http_duration = dict(self._http_duration)
            generation_count = dict(self._generation_count)
            generation_duration = dict(self._generation_duration)

        lines = [
            "# HELP local_llm_model_loaded Whether the configured model is loaded.",
            "# TYPE local_llm_model_loaded gauge",
            f"local_llm_model_loaded {int(model_loaded)}",
            "# HELP local_llm_active_sessions Number of active in-memory chat sessions.",
            "# TYPE local_llm_active_sessions gauge",
            f"local_llm_active_sessions {active_sessions}",
            "# HELP local_llm_process_resident_memory_bytes Resident process memory.",
            "# TYPE local_llm_process_resident_memory_bytes gauge",
            f"local_llm_process_resident_memory_bytes {resident_memory}",
            "# HELP local_llm_http_requests_total Completed HTTP requests.",
            "# TYPE local_llm_http_requests_total counter",
        ]
        for (method, route, status), count in sorted(http_requests.items()):
            lines.append(
                f'local_llm_http_requests_total{{method="{method}",route="{route}",status="{status}"}} {count}'
            )
        lines.extend(
            [
                "# HELP local_llm_http_request_duration_seconds HTTP duration sum and count.",
                "# TYPE local_llm_http_request_duration_seconds summary",
            ]
        )
        for (method, route), (count, total) in sorted(http_duration.items()):
            labels = f'method="{method}",route="{route}"'
            lines.append(f"local_llm_http_request_duration_seconds_sum{{{labels}}} {total:.6f}")
            lines.append(f"local_llm_http_request_duration_seconds_count{{{labels}}} {count}")
        lines.extend(
            [
                "# HELP local_llm_generations_total Completed generations by outcome.",
                "# TYPE local_llm_generations_total counter",
            ]
        )
        for outcome, count in sorted(generation_count.items()):
            lines.append(f'local_llm_generations_total{{outcome="{outcome}"}} {count}')
        lines.extend(
            [
                "# HELP local_llm_generation_duration_seconds Generation duration sum and count.",
                "# TYPE local_llm_generation_duration_seconds summary",
            ]
        )
        for outcome, (count, total) in sorted(generation_duration.items()):
            labels = f'outcome="{outcome}"'
            lines.append(f"local_llm_generation_duration_seconds_sum{{{labels}}} {total:.6f}")
            lines.append(f"local_llm_generation_duration_seconds_count{{{labels}}} {count}")
        return "\n".join(lines) + "\n"
