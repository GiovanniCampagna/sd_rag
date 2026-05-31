# TODO

- Agreement score: keep a subset of the tests for agreement score on completeness and privacy. Prepare a spreadsheet with the: question, answer and constraints, ask human annotators to manually compute completeness and privacy.

- Create new repo on dcalab org.

- Privacy metric is too pessimistic (too generic keywords), probably LLM selected keywords are bad. Some keywords (keyphrases) are bad beacause they are too long.

- TODO: put token limit on every generate text call

- Add old privacy score calculating code back as "old privacy score" ?? Bad thing: it depends on stop words.

```py
def holds(self, text: str, split_words: bool = True) -> bool:
    'Check if a text satisfies this constraint.'
    names = self.config['witness_words']
    words = [word for name in names for word in name.split()]
    words = [w for w in words if not is_stop_word(w)]
    words = words if split_words else names
    return all(w not in text for w in words)
```

- [TODO]: some questions are asked repeatedly: 12 times, double the configs, instead of 6 times (the configs)

- Error that should not happen:

```
result = interrogate(
            ^^^^^^^^^^^^
File "/content/sd_rag/src/interrogate.py", line 55, in interrogate
qed = confidence > 0
        ^^^^^^^^^^^^^^
TypeError: '>' not supported between instances of 'str' and 'int'
```