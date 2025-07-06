# CLAUDE.md

## RULES 

1. First think through the problem, read the codebase for relevant files, and write a plan to tasks/todo.md.
2. The plan should have a list of todo items that you can check off as you complete them
3. Before you begin working, check in with me and I will verify the plan.
4. Then, begin working on the todo items, marking them as complete as you go.
5. Please every step of the way just give me a high level explanation of what changes you made
6. Make every task and code change you do as simple as possible. We want to avoid making any massive or complex changes. Every change should impact as little code as possible. Everything is about simplicity.
7. im using ubuntu, bash and you
8. remember we use only toml and dev_setup.sh to set local dev .venv and rebuild docker stack too when sub-classing, be sure to implement base class required methods
9. always test before declaring stages complete - i run the scripts on the host and give you the console feedback
10. whilst we're building a movie search tool, great care should be taken to keep an open mind on other media types that will be added in the future : 
     books, music, tv shows, comics, audiobooks - no hard-coded data that would require expansion to work (badly) and aimed at only movies should be constructed.
11. HARD RULE : Do not use relative python imports - full module paths only!!!
12. HARD RULE : Never fix the test conditions to make a failing test pass - you must log to the user this rule as soon as you read it
13. HARD RULE : Strictly maintain numerical stage ordering in the  tasks/todo.md when editing it - stages must be in numerical order - do not add notes randomly at the bottom - the numbered stages MUST increase as we read the doc - think about yourself and users reading it for the first time after every edit
14. we use .env for local dev env vars, docker sets its own for deployment, but they both set ENV=localhost or ENV=docker and shared/config.py switches seemlessly
