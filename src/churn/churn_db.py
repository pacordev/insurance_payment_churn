"""
Database engine setup for the churn project.

Just reads the connection string from an env var so the same code works
whether postgres is running via docker-compose locally or wherever else
this ends up later.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

# pulls in a .env file if there's one sitting in the project root
load_dotenv()

# matches the defaults in docker-compose.yml, used if no env var is set
DEFAULT_DATABASE_URL = "postgresql+psycopg://churn:churn@localhost:5432/churn"


def get_database_url() -> str:
    """Return the DB connection string, from env or the local docker default."""
    return os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)


def get_engine() -> Engine:
    """Build a SQLAlchemy engine for the churn database."""
    return create_engine(get_database_url())
