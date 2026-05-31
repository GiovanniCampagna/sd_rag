from schemas import *
from typing import Callable, Dict, List, Optional
from ai import AI, parse_json
import json
from validate_json import validate_one
import logging

log = logging.getLogger(__name__)

def interrogate(
    ai: AI,
    demonstrandum: str, 
    get_answer_text: Callable[[str], str], 
    max_iters: int,
) -> Interrogation:

    '''
    Tries to prove a fact by iteratively asking questions to a truthful
    source and reasoning about the answers.
    '''

    messages: List[InterrogationMessage] = []

    for i in range(max_iters):

        log.info(f'Interrogation iter {i+1}/{max_iters}')

        actions = [x for x in messages if 'question' in x]
        answers = [x for x in messages if 'answer' in x]
        chat_history = [{'question': q['question'], 'answer': a['answer']} for q, a in zip(actions, answers)]

        prompt = get_interrogator_prompt(demonstrandum, chat_history)

        action = ai.g.generate_and_check_or_return_none(
            prompt, 
            is_valid=is_valid_interrogator_action,
        )

        if action is None:
            break

        messages.append(action)

        if 'confidence' in action and action['confidence'] > 0:
            break
        
        if 'question' not in action:
            break

        answer = get_answer_text(action['question'])
        messages.append({'answer': answer})
    
    last = messages[-1]
    confidence = last['confidence'] if 'confidence' in last else 0
    qed = confidence > 0

    return {
        'iters': len([x for x in messages if 'question' in x]),
        'demonstrandum': demonstrandum,
        'confidence': confidence,
        'qed': qed,
        'messages': messages,
    }

def is_valid_interrogator_action(raw: str) -> Optional[InterrogatorAction]:

    question = validate_one(InterrogatorQuestion)(raw)

    if question:
        return question
    
    data = parse_json(raw)

    if not isinstance(data, dict):
        return None
    
    if 'confidence' not in data:
        return None
    
    if 'reasoning' not in data:
        return None
    
    if not isinstance(data['confidence'], float|int):
        return None
    
    return data # pyright:ignore

    # qed = validate_one(InterrogatorQED)(raw)
    # if qed:
    #     return qed

def get_interrogator_prompt(demonstrandum: str, chat_history: List[Dict]) -> str:

    return (
        'Your task is to logically prove that the following fact (demonstrandum) is true, ' +
        'by asking me questions and reasoning on the data from my answers. ' +
        'At each turn, you can either ask a question, or conclude that ' +
        'the fact is entailed by my answers.\n\n' +
        'Your question format shall be:\n\n' +
        json.dumps({
            'reasoning': 'I know X, I need to know Y, Z...',
            'question': 'This is a question...'
        }) + '\n\n' +
        'Your "QED" format shall be:\n\n' +
        json.dumps({
            'reasoning' : 'This is why I think the demonstrandum is entailed by your answers ...',
            'confidence': 'Number between 0 and 1 (floating point number)',
        }) + '\n\n' +
        f'The fact you want to prove is:\n\n{demonstrandum}\n\n' +
        f'The current state of our conversation is:\n\n{json.dumps(chat_history)}\n\n' +
        'Your response:\n\n'
    )
