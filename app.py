# Ensure .env is loaded before anything else
import src.utils.load_env
# Web UI launcher
from src.ui.flask_app import create_app
import config

app = create_app()

if __name__ == "__main__":
    app.run(host=config.GRADIO_SERVER_NAME, port=config.GRADIO_SERVER_PORT, debug=config.GRADIO_DEBUG)
