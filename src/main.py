import threading
import uvicorn
import asyncio
import webbrowser

from src.core.registry import ServiceRegistry
from src.interface.app import app as fastapi_app


def run_backend(host: str, port: int):
    uvicorn.run(fastapi_app, host=host, port=port, log_level="warning")


async def main_async():
    registry = ServiceRegistry()
    await registry.initialize()
    
    settings = registry.settings
    
    backend_thread = threading.Thread(
        target=run_backend, 
        args=("127.0.0.1", 8000), 
        daemon=True
    )
    backend_thread.start()
    print("[Main] Backend server started successfully on http://127.0.0.1:8000")

    if settings.ui_mode == "tray":
        print("[Main] Launching native Tray Application...")
        from src.interface.tray.tray_app import TrayController
        controller = TrayController()
        controller.run() 
        
    elif settings.ui_mode == "web":
        print("[Main] Launching Web Interface...")
        webbrowser.open("http://127.0.0.1:8000/")
        await _keep_alive()
            
    else:
        print("[Main] Headless mode: No UI requested.")
        await _keep_alive()

    await registry.shutdown()


async def _keep_alive():
    """Keeps the main thread alive for headless/web modes."""
    try:
        while True:
            await asyncio.sleep(3600)
    except KeyboardInterrupt:
        print("\n[Main] Shutting down gracefully...")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()