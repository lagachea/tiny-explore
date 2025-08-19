.POSIX:

install:
	uv venv --python=3.10
	uv pip install -r requirements.txt

test:
	python test.py
