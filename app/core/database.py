from __future__ import annotations

from typing import Optional

from neo4j import Driver, GraphDatabase

from app.core.config import Settings


def create_neo4j_driver(settings: Settings) -> Driver:
    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password),
    )
    driver.verify_connectivity()
    return driver


def close_neo4j_driver(driver: Optional[Driver]) -> None:
    if driver is None:
        return
    driver.close()

