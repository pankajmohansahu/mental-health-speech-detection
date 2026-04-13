"""
config.py — API keys and global configuration
AI-Based Early Mental Health Breakdown Detection from Speech Patterns
Group 6 | IIT Madras — BSDA4001
"""

import os

# Set GROQ_API_KEY as an environment variable before running the app.
# Local:        set GROQ_API_KEY=gsk_... (Windows) or export GROQ_API_KEY=gsk_... (Linux/Mac)
# HF Spaces:   add it under Settings → Variables and secrets → New secret
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

GROQ_MODEL = "llama-3.3-70b-versatile"
