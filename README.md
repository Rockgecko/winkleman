# Winkleman: A home made claude chat app

A standalone chat app that uses Claude models via the Anthropic API.

To run, ensure ANTHROPIC_API_KEY is set in the environment and .venv is
active, then run:

```bash
uv run streamlit run app.py
```

TODO:

- [x] add extended thinking option & downloads (see email)
- [ ] consider adding search tool
- [ ] look into whether it's possible to display remaining credit
- [ ] also conversation tokens/cost
