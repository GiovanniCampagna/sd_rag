Your goal is to evaluate the completeness of a provided answer compared to the golden answer. Rate the answer from 0 to 1 based on how much it covers the facts in the golden answer:

- 0 = No relevant facts from the golden answer.
- 1 = All facts from the golden answer are covered.

Scores between 0 and 1 should reflect how much information is missing or misrepresented.

Provide your evaluation in the following JSON format:

{{
    "short_reasoning": "a string", 
    "completeness_score": "a float between 0 and 1"
}}

For example:

Example golden answer: Bananas are rich in potassium and vitamin B6, and they are a good source of fiber. They grow in clusters and are harvested while still green. Bananas are native to Southeast Asia.

Example answer: Bananas are high in potassium and vitamin B6, and are a popular fruit.

Example output:

{{
    "short_reasoning": "The answer mentions potassium and vitamin B6, but misses growth in clusters, harvesting while green and provenance. It mentions 2 useful facts over 5. It is a 2/5 = 0.4.",
    "completeness_score": 0.4
}}

End of the example.

The golden answer is:

{golden_answer}

The actual answer is:

{actual_answer}

Your evaluation:
