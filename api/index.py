"""Entrada WSGI/ASGI para plataformas Python serverless (incluido Vercel)."""

from api.server import create_app

app = create_app()
