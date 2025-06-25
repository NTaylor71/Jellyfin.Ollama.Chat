#!/bin/bash
set -e

echo "⏳ Waiting for Postgres..."
until pg_isready -h "$POSTGRES_HOST" -U "$POSTGRES_USER"; do
    sleep 1
done

echo "🚀 Running Django migrations..."
python manage.py migrate

echo "🎯 Starting Django dev server..."
exec python manage.py runserver 0.0.0.0:8000
