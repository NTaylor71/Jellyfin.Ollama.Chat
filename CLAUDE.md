# CLAUDE.md

## RULES 

1. First think through the problem, read the codebase for relevant files, and write a plan to tasks/todo.md.
2. The plan should have a list of todo items that you can check off as you complete them
3. Before you begin working, check in with me and I will verify the plan.
4. Then, begin working on the todo items, marking them as complete as you go.
5. Please every step of the way just give me a high level explanation of what changes you made
6. Make every task and code change you do as simple as possible. We want to avoid making any massive or complex changes. Every change should impact as little code as possible. Everything is about simplicity.
7. Finally, add a review section to the [todo.md](http://todo.md/) file with a summary of the changes you made and any other relevant information.
8. im in wsl and I've been running test_<thing>.py or test<thing>.ps1 or debug_<thing>.py scripts in my host and reporting    │
│   back to you that way
9. remember we use only toml and dev_setup.ps1 to set local dev .venv and rebuild docker stack too
10. when sub-classing, be sure to implement base class required methods
11. always test before declaring stages complete - i run the scripts on the host and give you the console feedback
12. whilst we're building a movie search tool, great care should be taken to keep an open mind on other media types that will be added in the future : books, music, tv shows, comics, audiobooks - nothing too brittle and aimed at only movies should be constructed. we want to avoid brittle.

## Project Overview

**Jellyfin.Ollama.Chat** is a production-grade RAG (Retrieval-Augmented Generation) system with microservices architecture. The system provides intelligent chat capabilities with FAISS vector search and Redis-based queue processing.

simple, focused implementations that maintain existing patterns while adding powerful new capabilities

RAW EXAMPLE JELLYFIN movie data can be found here : data/example_movie_data.py
