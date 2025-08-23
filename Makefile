.POSIX:

install:
	uv venv --python=3.10
	uv pip install -r requirements.txt

compile:
	python compile_small.py

test:
	python test.py

fclean:
	rm -rf .venv

re: fclean install
