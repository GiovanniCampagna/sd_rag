You are an expert at finding the atomic facts of a text; a text can always be decomposed into one or more atomic facts, where each atomic fact is a short sentence (MAX 15 words, no more) that uses explicit names and avoids pronouns and other implicit references. You will be given a text, and your task is to decompose it into a list of atomic facts. Please provide your output as a markdown list.

For example, the text:

Alice moved the laundry to the dryer this morning and realized she'd been using the same slightly bent hanger for years without ever bothering to replace it.

Should be decomposed into the following atomic facts:

- Alice moved the laundry to the dryer this morning.
- Alice realized that Alice had been using the same slightly bent hanger for years.
- Alice never replaced the slightly bent hanger.

Another example, the text:

Bob baked a chocolate cake yesterday and accidentally left the oven on all night.

Should be decomposed into the following atomic facts:

- Bob baked a chocolate cake yesterday.
- Bob accidentally left the oven on all night.

End of the examples.

The text is:

{text}

The atomic facts are:
