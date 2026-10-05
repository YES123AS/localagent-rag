.PHONY: up down logs restart

up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f app

restart:
	docker compose restart app
