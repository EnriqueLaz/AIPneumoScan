.PHONY: install run-api run-app streamlit

install:
	pip install -r requirements.txt
	pip install -e .

run-api:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

run-app streamlit:
	streamlit run app/app.py
