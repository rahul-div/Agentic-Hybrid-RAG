"""
Multi-Tenant RAG System - Main Application Entry Point
Project-per-Tenant Architecture with Neon PostgreSQL and Graphiti
"""

import os
import logging
import uvicorn
from typing import Optional

# Import our corrected multi-tenant components
from multi_tenant_api import create_app


# Configuration class for project-per-tenant architecture
class Config:
    """Configuration management for the multi-tenant RAG system."""

    # Catalog Database (Control Plane)
    CATALOG_DATABASE_URL: Optional[str] = os.getenv("CATALOG_DATABASE_URL")

    # Neon API for project creation
    NEON_API_KEY: Optional[str] = os.getenv("NEON_API_KEY")

    # Neo4j (Shared instance with namespacing)
    NEO4J_URI: Optional[str] = os.getenv("NEO4J_URI")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: Optional[str] = os.getenv("NEO4J_PASSWORD")

    # JWT Authentication
    JWT_SECRET_KEY: Optional[str] = os.getenv("JWT_SECRET_KEY")
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")

    # Gemini API (using Google API Key)
    GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")

    # For backward compatibility, get either Google or Gemini API key
    @property
    def API_KEY(self) -> Optional[str]:
        return self.GOOGLE_API_KEY or self.GEMINI_API_KEY

    # Application Settings
    APP_ENV: str = os.getenv("APP_ENV", "development")
    APP_HOST: str = os.getenv("APP_HOST", "127.0.0.1")
    APP_PORT: int = int(os.getenv("APP_PORT", 8000))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "info")

    # Neon Configuration
    DEFAULT_REGION: str = os.getenv("DEFAULT_NEON_REGION", "aws-us-east-1")

    @classmethod
    def validate(cls) -> None:
        """Validate required configuration variables for project-per-tenant."""
        required_vars = [
            ("CATALOG_DATABASE_URL", cls.CATALOG_DATABASE_URL),
            ("NEON_API_KEY", cls.NEON_API_KEY),
            ("NEO4J_URI", cls.NEO4J_URI),
            ("NEO4J_PASSWORD", cls.NEO4J_PASSWORD),
            ("JWT_SECRET_KEY", cls.JWT_SECRET_KEY),
            ("GOOGLE_API_KEY or GEMINI_API_KEY", cls().API_KEY),
        ]

        missing = [var_name for var_name, var_value in required_vars if not var_value]
        if missing:
            raise ValueError(
                f"Missing required environment variables: {missing}. "
                f"Please check your .env file and ensure all variables are set."
            )

    @classmethod
    def get_summary(cls) -> dict:
        """Get configuration summary for logging."""
        return {
            "app_env": cls.APP_ENV,
            "app_host": cls.APP_HOST,
            "app_port": cls.APP_PORT,
            "log_level": cls.LOG_LEVEL,
            "default_region": cls.DEFAULT_REGION,
            "has_catalog_db": bool(cls.CATALOG_DATABASE_URL),
            "has_neon_api_key": bool(cls.NEON_API_KEY),
            "has_neo4j_config": bool(cls.NEO4J_URI and cls.NEO4J_PASSWORD),
            "has_jwt_secret": bool(cls.JWT_SECRET_KEY),
            "has_gemini_key": bool(cls().API_KEY),
        }


def setup_logging(level: str) -> None:
    """Setup application logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("multi_tenant_rag.log")],
    )


def main():
    """Main application entry point."""
    try:
        # Setup logging
        setup_logging(Config.LOG_LEVEL)
        logger = logging.getLogger(__name__)

        # Validate configuration
        Config.validate()

        logger.info("Starting Multi-Tenant RAG System")
        logger.info(f"Configuration: {Config.get_summary()}")

        # Create FastAPI app with project-per-tenant architecture
        app = create_app(
            neon_api_key=Config.NEON_API_KEY,
            catalog_database_url=Config.CATALOG_DATABASE_URL,
            neo4j_uri=Config.NEO4J_URI,
            neo4j_user=Config.NEO4J_USER,
            neo4j_password=Config.NEO4J_PASSWORD,
            jwt_secret_key=Config.JWT_SECRET_KEY,
            gemini_api_key=Config().API_KEY,
            default_region=Config.DEFAULT_REGION,
            title="Multi-Tenant RAG API (Project-per-Tenant)",
            version="2.0.0",
            description="Production-ready multi-tenant RAG system following official Neon and Graphiti best practices",
        )

        # Start the server
        uvicorn.run(
            app,
            host=Config.APP_HOST,
            port=Config.APP_PORT,
            reload=(Config.APP_ENV == "development"),
            log_level=Config.LOG_LEVEL.lower(),
            access_log=True,
        )

    except ValueError as e:
        print(f"Configuration Error: {e}")
        return 1
    except Exception as e:
        print(f"Startup Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
