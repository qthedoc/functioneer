# Release Procedure

2025-06-25

- Update version in pyproject.toml
- Update version in __init__.py
- Update CHANGELOG.md
- Merge branch
- tag release: git tag vX.X.X
- push tag: git push origin vX.X.X
- build: python -m build      
- publish: python -m twine upload dist/* 