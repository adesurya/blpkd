"""
core/abstractions/queue.py

Queue abstraction layer — swap Redis → RabbitMQ → Kafka
by changing QUEUE_BACKEND env var only. No business logic changes needed.

Usage:
    from core.abstractions.queue import get_queue
    queue = get_queue()
    await queue.publish("detection.jobs", payload)
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

import structlog

log = structlog.get_logger(__name__)


class QueueBackend(ABC):
    """Abstract queue interface. All backends must implement this."""

    @abstractmethod
    async def publish(self, topic: str, message: dict[str, Any]) -> str:
        """Publish a message. Returns message ID."""
        ...

    @abstractmethod
    async def subscribe(self, topic: str, handler: Any) -> None:
        """Subscribe to a topic with a handler coroutine."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the queue backend is healthy."""
        ...


# ─────────────────────────────────────────────────────────────
# MVP: Redis Backend (using Redis Streams for ordered delivery)
# ─────────────────────────────────────────────────────────────
class RedisQueueBackend(QueueBackend):
    """
    Redis Streams-based queue.
    Perfect for MVP — low latency, easy to set up.
    """

    def __init__(self, redis_url: str) -> None:
        import redis.asyncio as aioredis
        self._client = aioredis.from_url(redis_url, decode_responses=True)

    async def publish(self, topic: str, message: dict[str, Any]) -> str:
        msg_id = await self._client.xadd(
            topic,
            {"data": json.dumps(message)},
            maxlen=10000,  # keep last 10k messages
        )
        log.debug("queue.published", topic=topic, msg_id=msg_id)
        return str(msg_id)

    async def subscribe(self, topic: str, handler: Any) -> None:
        # Used directly via Celery for task dispatch in MVP
        # Direct consume pattern for non-Celery use cases
        last_id = ">"
        while True:
            try:
                messages = await self._client.xreadgroup(
                    "vision-workers", "worker-1",
                    {topic: last_id},
                    count=1, block=1000
                )
                for stream, entries in (messages or []):
                    for msg_id, data in entries:
                        await handler(json.loads(data["data"]))
                        await self._client.xack(topic, "vision-workers", msg_id)
            except Exception as e:
                log.error("queue.subscribe_error", error=str(e))

    async def health_check(self) -> bool:
        try:
            return await self._client.ping()
        except Exception:
            return False


# ─────────────────────────────────────────────────────────────
# GROWTH: RabbitMQ Backend
# ─────────────────────────────────────────────────────────────
class RabbitMQBackend(QueueBackend):
    """
    RabbitMQ AMQP backend.
    Better for: routing, dead letter queues, message TTL.
    Swap in Growth phase by setting QUEUE_BACKEND=rabbitmq.
    """

    def __init__(self, amqp_url: str) -> None:
        self._url = amqp_url
        self._connection = None

    async def _get_connection(self):
        if self._connection is None or self._connection.is_closed:
            import aio_pika
            self._connection = await aio_pika.connect_robust(self._url)
        return self._connection

    async def publish(self, topic: str, message: dict[str, Any]) -> str:
        import aio_pika
        conn = await self._get_connection()
        async with conn.channel() as channel:
            await channel.declare_queue(topic, durable=True)
            msg = aio_pika.Message(
                json.dumps(message).encode(),
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            )
            await channel.default_exchange.publish(msg, routing_key=topic)
        return "rabbitmq-published"

    async def subscribe(self, topic: str, handler: Any) -> None:
        import aio_pika
        conn = await self._get_connection()
        async with conn.channel() as channel:
            queue = await channel.declare_queue(topic, durable=True)
            async for msg in queue:
                async with msg.process():
                    await handler(json.loads(msg.body.decode()))

    async def health_check(self) -> bool:
        try:
            conn = await self._get_connection()
            return not conn.is_closed
        except Exception:
            return False


# ─────────────────────────────────────────────────────────────
# ENTERPRISE: Kafka Backend
# ─────────────────────────────────────────────────────────────
class KafkaBackend(QueueBackend):
    """
    Apache Kafka backend.
    Best for: high throughput, replay, partitioned processing.
    Swap in Enterprise phase by setting QUEUE_BACKEND=kafka.
    """

    def __init__(self, bootstrap_servers: str) -> None:
        self._servers = bootstrap_servers

    async def publish(self, topic: str, message: dict[str, Any]) -> str:
        from aiokafka import AIOKafkaProducer
        producer = AIOKafkaProducer(bootstrap_servers=self._servers)
        await producer.start()
        try:
            result = await producer.send_and_wait(
                topic,
                json.dumps(message).encode()
            )
            return str(result.offset)
        finally:
            await producer.stop()

    async def subscribe(self, topic: str, handler: Any) -> None:
        from aiokafka import AIOKafkaConsumer
        consumer = AIOKafkaConsumer(
            topic,
            bootstrap_servers=self._servers,
            group_id="vision-workers",
        )
        await consumer.start()
        try:
            async for msg in consumer:
                await handler(json.loads(msg.value.decode()))
        finally:
            await consumer.stop()

    async def health_check(self) -> bool:
        try:
            from aiokafka.admin import AIOKafkaAdminClient
            client = AIOKafkaAdminClient(bootstrap_servers=self._servers)
            await client.start()
            await client.close()
            return True
        except Exception:
            return False


# ─────────────────────────────────────────────────────────────
# Factory — returns the right backend based on settings
# ─────────────────────────────────────────────────────────────
_queue_instance: QueueBackend | None = None


def get_queue() -> QueueBackend:
    """
    Factory function. Returns singleton queue backend.
    Backend is determined by QUEUE_BACKEND env var.

    MVP:        QUEUE_BACKEND=redis     → RedisQueueBackend
    Growth:     QUEUE_BACKEND=rabbitmq  → RabbitMQBackend
    Enterprise: QUEUE_BACKEND=kafka     → KafkaBackend
    """
    global _queue_instance
    if _queue_instance is not None:
        return _queue_instance

    from core.config.settings import settings

    backend = settings.QUEUE_BACKEND
    if backend == "redis":
        _queue_instance = RedisQueueBackend(settings.REDIS_URL)
    elif backend == "rabbitmq":
        _queue_instance = RabbitMQBackend(settings.RABBITMQ_URL)
    elif backend == "kafka":
        _queue_instance = KafkaBackend(settings.KAFKA_BOOTSTRAP_SERVERS)
    else:
        raise ValueError(f"Unknown queue backend: {backend}")

    log.info("queue.initialized", backend=backend)
    return _queue_instance
