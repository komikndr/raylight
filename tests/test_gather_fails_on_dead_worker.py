import time

import pytest

import raylight.comfy_extra_dist.nodes_custom_sampler as sampler


class _Ref:
    """A future that finishes `at` seconds into the test, or never when None."""

    def __init__(self, at, value=None, error=None):
        self.at, self.value, self.error = at, value, error

    def done(self, start):
        return self.at is not None and time.monotonic() - start >= self.at


class _Ray:
    def __init__(self):
        self.start = time.monotonic()

    def wait(self, refs, num_returns=1, timeout=None):
        end = time.monotonic() + (timeout or 0)
        while True:
            ready = [ref for ref in refs if ref.done(self.start)]
            if len(ready) >= num_returns or time.monotonic() >= end:
                return ready, [ref for ref in refs if ref not in ready]
            time.sleep(0.02)

    def get(self, ref, timeout=None):
        if isinstance(ref, list):
            return [self.get(item) for item in ref]
        if ref.error is not None:
            raise ref.error
        return ref.value


@pytest.fixture
def retired(monkeypatch):
    reasons = []
    monkeypatch.setattr(sampler, "ray", _Ray())
    monkeypatch.setattr(sampler, "retire_actors", lambda actors, reason: reasons.append(reason))
    monkeypatch.setattr(sampler, "CANCEL_PEER_MIN_SECONDS", 0.5)
    return reasons


def test_a_dead_worker_fails_the_job_and_ends_the_peers_it_stranded(retired):
    futures = [_Ref(None), _Ref(None), _Ref(0.1, error=RuntimeError("worker died")), _Ref(None)]

    with pytest.raises(RuntimeError, match="worker died"):
        sampler._gather_with_progress(futures, {"workers": [object()] * 4})

    assert len(retired) == 1 and "3 were left waiting" in retired[0]


def test_workers_that_all_fail_are_left_alone(retired):
    futures = [_Ref(0.1 + 0.05 * i, error=RuntimeError("out of memory")) for i in range(4)]

    with pytest.raises(RuntimeError, match="out of memory"):
        sampler._gather_with_progress(futures, {"workers": [object()] * 4})

    assert retired == []


def test_a_finished_sample_is_returned(retired):
    futures = [_Ref(0.1, value=rank) for rank in range(4)]

    assert sampler._gather_with_progress(futures, {"workers": [object()] * 4}) == [0, 1, 2, 3]
    assert retired == []
