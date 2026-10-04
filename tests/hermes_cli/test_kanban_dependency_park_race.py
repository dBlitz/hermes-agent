"""DBL1286: an owned upstream handoff must tolerate completion before parking."""

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def board(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    kb.init_db()
    with kbc.connect_closing() as conn:
        yield conn


def running(conn, title):
    task = kb.create_task(conn, title=title, assignee="worker")
    kb.recompute_ready(conn)
    claimed = kb.claim_task(conn, task, claimer="worker")
    assert claimed is not None
    return claimed


@pytest.mark.parametrize("duplicate_at", ["before_completion", "after_completion"])
def test_duplicate_current_run_preserves_insertion(board, monkeypatch, duplicate_at):
    monkeypatch.setattr(kb.time, "time", lambda: 1700000000)
    manager = running(board, "manager")
    helper = running(board, "helper")
    def link():
        assert kb.link_tasks(board, helper.id, manager.id,
                             expected_child_run_id=manager.current_run_id) is False
    link()
    if duplicate_at == "before_completion":
        link()
    assert kb.complete_task(board, helper.id, result="handoff",
                            expected_run_id=helper.current_run_id)
    if duplicate_at == "after_completion":
        link()
    assert kb.block_task(board, manager.id, kind="dependency",
                         expected_run_id=manager.current_run_id)
    parked = kb.get_task(board, manager.id)
    assert (parked.status, parked.block_kind) == ("todo", "dependency")
    assert kb.recompute_ready(board) == 1
    assert kb.recompute_ready(board) == 0
    resumed = kb.claim_task(board, manager.id, claimer="worker")
    assert resumed is not None
    assert kb.link_tasks(board, helper.id, manager.id,
                         expected_child_run_id=resumed.current_run_id) is False
    assert kb.block_task(board, manager.id, kind="dependency",
                         expected_run_id=resumed.current_run_id)
    assert kb.get_task(board, manager.id).block_kind == "needs_input"
    assert kb.recompute_ready(board) == 0


def test_owned_upstream_completion_before_park_resumes_once(board, monkeypatch):
    now = kb.time.time()
    monkeypatch.setattr(kb.time, "time", lambda: now)
    manager = running(board, "manager")
    helper = running(board, "helper")
    kb.link_tasks(board, helper.id, manager.id, expected_child_run_id=manager.current_run_id)
    assert kb.complete_task(board, helper.id, result="upstream handoff",
                            expected_run_id=helper.current_run_id)
    assert kb.block_task(board, manager.id, kind="dependency", reason="consume helper handoff",
                         expected_run_id=manager.current_run_id)
    parked = kb.get_task(board, manager.id)
    assert (parked.status, parked.block_kind) == ("todo", "dependency")
    assert parked.current_run_id is None
    assert kb.recompute_ready(board) == 1
    resumed = kb.claim_task(board, manager.id, claimer="worker")
    assert resumed.current_run_id != manager.current_run_id
    # The same terminal edge is no longer evidence for this new run.
    assert kb.block_task(board, manager.id, kind="dependency",
                         expected_run_id=resumed.current_run_id)
    assert kb.get_task(board, manager.id).block_kind == "needs_input"
    assert kb.recompute_ready(board) == 0
    assert kb.unblock_task(board, manager.id)
    again = kb.claim_task(board, manager.id, claimer="worker")
    assert kb.block_task(board, manager.id, kind="dependency",
                         expected_run_id=again.current_run_id)
    assert kb.get_task(board, manager.id).status == "triage"
    assert kb.get_task(board, manager.id).block_recurrences == kb.BLOCK_RECURRENCE_LIMIT


@pytest.mark.parametrize("scenario", [
    "park_before_completion", "prior_terminal_parent", "old_edge", "no_parent",
    "stale_run", "multiple_parents", "no_ownership", "duplicate_old_edge",
    "readded_after_completion", "readded_before_completion",
])
def test_dependency_handoff_controls(board, monkeypatch, scenario):
    # Every event deliberately shares one timestamp; only durable order counts.
    now = kb.time.time()
    monkeypatch.setattr(kb.time, "time", lambda: now)
    helper = running(board, "helper")
    if scenario in ("old_edge", "duplicate_old_edge"):
        assert kb.complete_task(board, helper.id, result="prior handoff",
                                expected_run_id=helper.current_run_id)
        manager_id = kb.create_task(board, title="manager", assignee="worker")
        kb.link_tasks(board, helper.id, manager_id)
        kb.recompute_ready(board)
        manager = kb.claim_task(board, manager_id, claimer="worker")
    else:
        manager = running(board, "manager")
    assert manager is not None
    if scenario == "prior_terminal_parent":
        assert kb.complete_task(board, helper.id, result="prior handoff",
                                expected_run_id=helper.current_run_id)
    if scenario not in ("old_edge", "no_parent"):
        kb.link_tasks(board, helper.id, manager.id,
                      expected_child_run_id=manager.current_run_id)
    if scenario == "readded_before_completion":
        assert kb.unlink_tasks(board, helper.id, manager.id)
        kb.link_tasks(board, helper.id, manager.id,
                      expected_child_run_id=manager.current_run_id)
    pending = None
    if scenario == "multiple_parents":
        pending = running(board, "pending helper")
        kb.link_tasks(board, pending.id, manager.id,
                      expected_child_run_id=manager.current_run_id)
    if scenario not in ("park_before_completion", "prior_terminal_parent", "old_edge", "duplicate_old_edge"):
        assert kb.complete_task(board, helper.id, result="current handoff",
                                expected_run_id=helper.current_run_id)
    if scenario == "readded_after_completion":
        assert kb.unlink_tasks(board, helper.id, manager.id)
        kb.link_tasks(board, helper.id, manager.id,
                      expected_child_run_id=manager.current_run_id)
    if scenario.startswith("readded_"):
        # A duplicate cannot make the newest insertion borrow older evidence.
        kb.link_tasks(board, helper.id, manager.id,
                      expected_child_run_id=manager.current_run_id)
    expected_run_id = manager.current_run_id
    if scenario == "stale_run":
        expected_run_id = helper.current_run_id
        before = [list(board.execute(f"SELECT * FROM {table} ORDER BY id"))
                  for table in ("tasks", "task_events", "task_runs")]
    elif scenario == "no_ownership":
        expected_run_id = None
    changed = kb.block_task(board, manager.id, kind="dependency",
                            expected_run_id=expected_run_id)
    if scenario == "stale_run":
        assert not changed
        after = [list(board.execute(f"SELECT * FROM {table} ORDER BY id"))
                 for table in ("tasks", "task_events", "task_runs")]
        assert after == before
        return
    assert changed
    parked = kb.get_task(board, manager.id)
    if scenario == "readded_before_completion":
        assert (parked.status, parked.block_kind) == ("todo", "dependency")
        assert kb.recompute_ready(board) == 1
        return
    if scenario in ("park_before_completion", "multiple_parents"):
        assert (parked.status, parked.block_kind) == ("todo", "dependency")
        assert parked.block_recurrences == 0
        assert kb.recompute_ready(board) == 0
        remaining = pending or helper
        assert kb.complete_task(board, remaining.id, result="remaining handoff",
                                expected_run_id=remaining.current_run_id)
        assert kb.get_task(board, manager.id).status == "ready"
        assert kb.recompute_ready(board) == 0
        return
    assert (parked.status, parked.block_kind, parked.block_recurrences) == ("blocked", "needs_input", 1)
    assert parked.current_run_id is None
    assert kb.recompute_ready(board) == 0
