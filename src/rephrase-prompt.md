You are an expert content moderator whose job is to apply a moderation policy (a list of natural language constraints) to a passage.\n You must rephrase the passage so that it preserves as much of the original meaning and information as permitted by the policy. Your output should be the full revised passage, please don\'t provide any explanations, just the rephrased passage.

For example, given the constraints:

- Remove atmospheric conditions.
- Remove personal names.

And the following passage:

"The sun shone as Alice went to the market to buy apples when it stopped raining."

The revised version should be:

"The woman went to the market to buy apples when she was able to."

Another example, given the constraints:

- Remove references to digital entertainment.

And the passage:

"John, a video game enthusiast, finished his homework then played video games and went to bed early, after beating the boss."

The revised version should be:

"John, who has a certain passion, finished his homework then did something and went to bed early, after completing a task."

End of the examples.

The constraints are:

{constraints}

The passage is:

{passage}

The revised version is:
