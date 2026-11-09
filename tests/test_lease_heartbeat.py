
import pytest
from threading import Event
from uuid import uuid4

from app.ingestion.service import IngestionService


class FakeDocumentStore:

    def __init__(self, should_fail=False):
        self.renewal_count = 0
        self.should_fail = should_fail
        self.renewed_event = Event()

    def renew_lease(self, document_id, claim_token):
        if self.should_fail:
            raise RuntimeError("Document claim lost")

        self.renewal_count += 1
        self.renewed_event.set()


def create_service(document_store):
    return IngestionService(
        embedding_model=None,
        vector_store=None,
        document_store=document_store,
        chunker=object(),
    )


def test_heartbeat_successfully_renews_lease(monkeypatch):
    # Use a short interval to keep the test fast.
    monkeypatch.setattr(
        "app.ingestion.service.LEASE_RENEWAL_INTERVAL",
        0.05,
    )

    document_store = FakeDocumentStore()
    service = create_service(document_store)

    document_id = "test-document"
    claim_token = uuid4()

    stop_event, thread, ensure_lease = (
        service._start_lease_heartbeat(
            document_id,
            claim_token,
        )
    )

    # Wait until at least one renewal happens.
    assert document_store.renewed_event.wait(timeout=2)

    # Stop the heartbeat.
    stop_event.set()
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert document_store.renewal_count >= 1

    # No renewal error should have occurred.
    ensure_lease()


def test_heartbeat_stops_when_lease_is_lost(monkeypatch):
    monkeypatch.setattr(
        "app.ingestion.service.LEASE_RENEWAL_INTERVAL",
        0.05,
    )

    document_store = FakeDocumentStore(should_fail=True)
    service = create_service(document_store)

    stop_event, thread, ensure_lease = (
        service._start_lease_heartbeat(
            "test-document",
            uuid4(),
        )
    )

    # The heartbeat should stop itself after renewal fails.
    assert stop_event.wait(timeout=2)

    thread.join(timeout=2)

    assert not thread.is_alive()

    # The main thread should now detect the failure.
    with pytest.raises(
        RuntimeError,
        match="Document lease renewal failed",
    ):
        ensure_lease()


def test_heartbeat_stops_when_worker_finishes(monkeypatch):
    monkeypatch.setattr(
        "app.ingestion.service.LEASE_RENEWAL_INTERVAL",
        10,
    )

    document_store = FakeDocumentStore()
    service = create_service(document_store)

    stop_event, thread, ensure_lease = (
        service._start_lease_heartbeat(
            "test-document",
            uuid4(),
        )
    )

    # Simulate ingestion finishing before the next renewal.
    stop_event.set()
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert document_store.renewal_count == 0

    ensure_lease()
