#!/usr/bin/env python3
"""
Aegis Health Dashboard Server Startup Script

This script starts the FastAPI backend server for the Aegis Health Dashboard.
The React frontend should be started separately with 'npm start' in the dashboard/ directory.

Usage:
    python start_dashboard_server.py

The server will be available at:
    - API: http://127.0.0.1:8000
    - API Docs: http://127.0.0.1:8000/docs
    - WebSocket: ws://127.0.0.1:8000/ws/health/live

Requirements: 13.1, 13.17
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.dashboard_api import create_app
from core.health_db import HealthDatabase
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Start the dashboard server."""
    try:
        # Initialize database
        logger.info("Initializing health database...")
        db = HealthDatabase()
        
        # Create FastAPI app
        logger.info("Creating FastAPI application...")
        app = create_app(db)
        
        # Start server
        logger.info("Starting Aegis Health Dashboard Server...")
        logger.info("API will be available at: http://127.0.0.1:8000")
        logger.info("API documentation at: http://127.0.0.1:8000/docs")
        logger.info("WebSocket endpoint at: ws://127.0.0.1:8000/ws/health/live")
        logger.info("")
        logger.info("To start the React frontend:")
        logger.info("  cd dashboard")
        logger.info("  npm install  # (first time only)")
        logger.info("  npm start")
        logger.info("")
        logger.info("Press Ctrl+C to stop the server")
        
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=8000,
            log_level="info"
        )
        
    except KeyboardInterrupt:
        logger.info("\nShutting down server...")
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
