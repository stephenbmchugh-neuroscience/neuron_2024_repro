.PHONY: env run test clean

env:
\tconda env create -f environment.yml

run:
\tpython reproduce_figures.py

test:
\tpytest -q

clean:
\trm -rf __pycache__ .pytest_cache
