# Release Procedure

2025-06-25

## Pre release
- update docs (readme and examples)
    - run examples.ipynb

- Update version in pyproject.toml
- Update version in __init__.py
- Update CHANGELOG.md
- Merge changes into main
- tag release: git tag vX.X.X
- push tag: git push origin vX.X.X
- build: python -m build      
- publish: python -m twine upload dist/* 


first time package upload you will have to `pip install build` and `pip install twine`