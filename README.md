# Domain Matcher

[![](https://img.shields.io/badge/Read_our_Blog-blue?logo=readdotcv)](https://dref360.github.io/domainmatching/)

Domain Matcher is a library that aims at matching a pre-defined domain to your input data.
Input without domain are deemed not important and thus can be safely filtered out.

> Domain Matching performs very cheap OoD detection using topic modeling and keyword extraction.

`pip install domain-matcher`

## Usage

```python
from datasets import load_dataset
from domain_matcher.core import DomainMatcher, DMConfig

# Custom version of `clinc-oos` where non-banking classes are assigned to oos.
ds = load_dataset("GlowstickAI/banking-clinc-oos", "plus")
config = DMConfig(text_column='text', label_column='intent', oos_class='oos')
dmatcher = DomainMatcher(config)
# Fit DM on your train data see our blog to see what's happening!
dmatcher.fit(ds['train'])

# Predict: You can predict on a string, List[str] or Dataset
dmatcher.transform("Can you cancel my credit card?")['in_domain']
# >>> True
dmatcher.transform("Can you cancel my reservation at Giorgi's?")['in_domain']
# >>> False
```

### Troubleshooting

For troubleshooting, please see our [wiki](https://github.com/GlowstickAI/domain-matcher/wiki) or [submit an issue](https://github.com/GlowstickAI/domain-matcher/issues) if you can't find what you're looking for.

## Development

* Install Pyenv
  * `curl https://pyenv.run | bash`
  * `pyenv install 3.9.13 && pyenv global 3.9.13`
* [Install Poetry](https://python-poetry.org/docs/master/#installing-with-the-official-installer)
* `poetry install`
* Add precommits
  * `poetry run pre-commit install`

### Tooling

* `make format`: format the code with Ruff
* `make test`: run unit tests and mypy.