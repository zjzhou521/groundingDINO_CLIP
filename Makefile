.PHONY: infra-up infra-down dev db-generate db-migrate db-current

infra-up:
	docker compose up -d

infra-down:
	docker compose down

dev:
	uv run uvicorn app.main:app --reload

db-generate:
	@test -n "$(m)" || (echo 'Usage: make db-generate m="describe_change"' && exit 1)
	uv run alembic revision --autogenerate -m "$(m)"

db-migrate:
	uv run alembic upgrade head

db-current:
	uv run alembic current
