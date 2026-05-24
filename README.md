How to run:
1. activate virtual environment and install requirements: `pip install -e .`
2. set the folder you want to index in the `.env`
3. `python src/interface/web/app.py`
4. in a separate terminal to stream vnc to browser: `websockify 6080 localhost:5900`
5. visit: `http://127.0.0.1:8000/` and type request (e.g. open chrome or ask about context of file in indexed folder)

Currently we're still using the ready-to-use agent by langchain for testing (will be replaced once custom ones have been implemented)