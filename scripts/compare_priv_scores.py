import json
from statistics import mean, stdev, variance
from scipy.stats import pearsonr, spearmanr

text = open('./results/qwen.eval.tmp.json').read()
records = json.loads(text)
factext = [x['completeness_score'] for x in records]
judge = [x['completeness_score_judge_llm'] for x in records]
diffs = [f - j for f, j in zip(factext, judge)]
bias = mean(diffs)
sd_diff = stdev(diffs)
var_diff = variance(diffs)
print("Bias:", bias)
print("Std diff:", sd_diff)
print("Variance:", var_diff)
print(pearsonr(factext, judge))
print(spearmanr(factext, judge))


