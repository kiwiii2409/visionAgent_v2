How to run:
1. activate virtual environment and install requirements: `pip install -e .`
2. `python src/interface/web/app.py`
3. in a separate terminal: `websockify 6080 localhost:5900`
4. visit: `http://127.0.0.1:8000/` and type request

Check tools for available actions (open program, move mouse etc.)
Currently we're still using the ready-to-use agent by langchain, can only perform a single step