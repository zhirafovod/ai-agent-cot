import os

# LM Studio Configuration
LM_STUDIO_API_BASE = "http://localhost:1234/v1"
os.environ["OPENAI_API_BASE"] = LM_STUDIO_API_BASE
os.environ["OPENAI_API_KEY"] = "dummy_key"