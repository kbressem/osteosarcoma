all:
	@echo "Hi :). Nothing is implemented in here yet."

install:
	install flake8 black[jupyter] isort parameterized
	pip install -e .

pretty:
	isort .
	black --line-length 100 .

lint:
	flake8 .
	mypy .

test:
	make lint
	cd tests && python -m unittest discover
	make clean

uninstall:
	pip install pip-autoremove
	pip-autoremove trainlib -y
	pip uninstall pip-autoremove -y

clean:
	python3 -Bc "import pathlib; [p.unlink() for p in pathlib.Path('.').rglob('*.py[co]')]"
	python3 -Bc "import pathlib; [p.rmdir() for p in pathlib.Path('.').rglob('__pycache__')]"
	python3 -Bc "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('.ipynb_checkpoints')]"