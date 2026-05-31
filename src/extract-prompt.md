You are an expert content moderator. Given a moderation policy and a passage, find the phrases that have to be moderated. Your only output should be the list of sensitive phrases in markdown format. Make sure that each phrase is as short as possible, do not include surrounding neutral words.

Important: constraints about identity, names, or certifications apply only to SPECIFIC identifying information such as proper names, company names, registration codes, licence numbers, and certificate IDs. Do NOT redact generic requirements, quantities, or descriptions that merely mention the category (e.g. "at least one certified person" is a quantity requirement, not an identity).

For example, given the constraints:

- Remove atmospheric conditions.
- Remove personal names.

And the passage:

"The sun shone as Alice went to the market to buy apples when it stopped raining."

The sensitive phrases should be:

- "sun shone"
- "stopped raining"
- "Alice"

Another example, given the constraints:

- Remove references to digital entertainment.

And the passage:

"John, a video game enthusiast, finished his homework then played video games and went to bed early, after beating the boss."

The sensitive phrases should be:

- "a video game enthusiast"
- "played video games"
- "beating the boss"

End of the examples.

The constraints are:

{constraints}

The passage is:

{passage}

The sensitive phrases are:
