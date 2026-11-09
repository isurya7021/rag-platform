
import sys
from pathlib import Path

from alembic import context
from sqlalchemy import pool

# Add project root to Python path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import existing database configuration.
from app.database.connection import DATABASE_URL
from app.database.models import Base

# Alembic uses this metadata to detect model changes.
target_metadata = Base.metadata

config = context.config


def run_migrations_offline() -> None:
    """Run migrations without creating a database connection."""

    context.configure(
        url=DATABASE_URL.render_as_string(hide_password=False),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations using a live database connection."""

    from app.database.connection import engine

    with engine.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
