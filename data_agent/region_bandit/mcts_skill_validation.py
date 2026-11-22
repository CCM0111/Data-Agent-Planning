"""Explicit external low-fit budget for paired policy replay; full fits forbidden."""
from pathlib import Path
import time

from ..mcts_search import SearchBudgetExceeded
from ..coreset import file_hash
from .schemas import read_json, write_json


class IncompleteValidation(RuntimeError):
    """A frozen low-only method failed; the bounded comparison is incomplete."""


class ValidationRunner:
    """Minimal evaluation facade; never mutates the search tree or its budget.

One ledger is shared by all sources in a candidate validation round. The caller
serializes campaign execution. Recovering committed low fits spends zero fits.
"""
    def __init__(self, backend, ledger, *, max_low_fits=216, max_api_calls=48, max_wall_seconds=86400):
        self.backend, self.path = backend, Path(ledger)
        self.limits = dict(max_low_fits=max_low_fits, max_api_calls=max_api_calls,
                           max_wall_seconds=max_wall_seconds)
        if any(type(v) is not int or v <= 0 for v in self.limits.values()):
            raise ValueError('Validation budget must be positive integer limits')
        self.state = (read_json(self.path) if self.path.exists() else dict(
            protocol='external-low-only-validation-v1', limits=self.limits,
            low_fit_attempts=0, api_calls=0, full_fit_attempts=0, elapsed_seconds=0., skill_validation={}))
        if self.state['limits'] != self.limits:
            raise ValueError('Validation budget changed after freeze')
        self.state.setdefault('failed_probes', {})
        self.elapsed = self.state['elapsed_seconds']; self.started = None

    def __enter__(self):
        self.started = time.monotonic()
        self.previous = self.backend.before_fit, self.backend.request_guard
        self.backend.before_fit, self.backend.request_guard = self.before_fit, self.before_request
        self.checkpoint()
        return self

    def __exit__(self, *_):
        try:
            self.checkpoint()
        finally:
            self.backend.before_fit, self.backend.request_guard = self.previous
            self.elapsed = self.state['elapsed_seconds']; self.started = None

    def check_wall(self):
        current = self.elapsed+(time.monotonic()-self.started if self.started is not None else 0.)
        if current >= self.limits['max_wall_seconds']:
            raise SearchBudgetExceeded('external_validation_wall_budget')

    def checkpoint(self):
        self.state['elapsed_seconds'] = self.elapsed+(time.monotonic()-self.started if self.started is not None else 0.)
        write_json(self.path, self.state)

    def before_fit(self, stage, candidate):
        if stage not in ('low', 'coreset', 'evaluation'):
            raise ValueError('Paired skill replay cannot train a full model')
        self.check_wall()
        if self.state['low_fit_attempts'] >= self.limits['max_low_fits']:
            raise SearchBudgetExceeded('external_validation_low_fit_budget')
        self.state['low_fit_attempts'] += 1
        self.state['inflight'] = dict(kind='low_fit', candidate_id=candidate['id'])
        self.checkpoint()

    def before_request(self):
        self.check_wall()
        if self.state['api_calls'] >= self.limits['max_api_calls']:
            raise SearchBudgetExceeded('external_validation_api_budget')
        self.state['api_calls'] += 1
        self.state['inflight'] = dict(kind='api', **self.state.get('active_evaluation', {}))
        self.checkpoint()

    async def evaluate(self, candidate, stage):
        if stage != 'low' or not candidate.get('anchor_of'):
            raise ValueError('Validation runner only evaluates frozen low-only probes')
        if candidate.get('low_report'):
            saved = candidate['low_report']
            report = self.backend.recover_evaluation(candidate, 'low')
            if report is None or report != saved:
                raise ValueError('Cached validation probe report changed or disappeared')
            return report
        failed = self.state['failed_probes'].get(candidate['id'])
        if failed or candidate.get('low_error') or candidate.get('status') == 'failed':
            reason = failed or candidate.get('low_error') or candidate.get('error') or 'Frozen low method failed'
            self.state['failed_probes'][candidate['id']] = reason
            self.checkpoint()
            raise IncompleteValidation(reason)
        self.check_wall()
        # Recovery verifies artifact identities. Those errors must stay hard
        # failures, rather than being disguised as an inconclusive experiment.
        report = self.backend.recover_evaluation(candidate, 'low')
        if report is None:
            frozen = candidate.get('frozen_stages', {}).get('coreset')
            if frozen and file_hash(frozen['initial_pipeline_path']) != frozen['initial_pipeline_sha256']:
                raise ValueError('Frozen validation constructor artifact changed')
            try:
                report = await self.backend.evaluate(candidate, 'low')
            except Exception as error:
                if candidate.get('status') == 'failed' or candidate.get('low_error'):
                    reason = f'{type(error).__name__}: {error}'
                    self.state['failed_probes'][candidate['id']] = reason
                    self.state.pop('inflight', None)
                    self.checkpoint()
                    raise IncompleteValidation(reason) from error
                raise
        candidate['low_report'], candidate['low_score'] = report, float(report['score'])
        self.state.pop('inflight', None)
        self.checkpoint()
        return report
