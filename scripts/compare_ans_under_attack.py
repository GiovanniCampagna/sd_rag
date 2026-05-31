import json

import sys

sys.path.append('src/')
records = json.loads(open('./results/qwen.eval.tmp.json').read())
inj = [x for x in records if x['generate'] == 'extract' and x['inject']]
noinj = [x for x in records if x['generate'] == 'extract' and not x['inject']]

c1 = 0
c2 = 0

for with_inj, without_inj in zip(inj , noinj):

    # When answer to benign query breaks more constraints
    if len(with_inj['broken_constraints']) < len(without_inj['broken_constraints']):

        are_using_same_constraints = len({x['sensitive_category'] for x in with_inj['retrieved_constraints']} & {x['sensitive_category'] for x in without_inj['retrieved_constraints']}) == 3 #len({x['sensitive_category'] for x in with_inj['retrieved_constraints']} )

        if not are_using_same_constraints:
            c1+=1
            continue

        c2+=1

        print('TO MALICIOUS:')
        # print({x['sensitive_category'] for x in with_inj['retrieved_constraints']})
        print(with_inj['answer_text'])
        print([ x['broken_by'] for x in with_inj['broken_constraints']])
        print('TO BENIGN:')
        print(without_inj['answer_text'])
        print([ x['broken_by'] for x in without_inj['broken_constraints']])
        # print({x['sensitive_category'] for x in without_inj['retrieved_constraints']})
        print('='*8)

print(c1)
print(c2)
