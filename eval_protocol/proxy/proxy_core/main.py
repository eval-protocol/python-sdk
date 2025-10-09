import os
from .app import create_app

if __name__ == "__main__":
    import uvicorn

    # Build app with default NoAuth for local runs
    application = create_app()
    port = int(os.getenv("PORT", "4000"))
    uvicorn.run(application, host="0.0.0.0", port=port)
