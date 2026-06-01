"""CLI entry point for visionagent."""
import uvicorn


def main():
    """Start the visionagent web server."""
    uvicorn.run(
        "src.interface.web.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )


if __name__ == "__main__":
    main()
