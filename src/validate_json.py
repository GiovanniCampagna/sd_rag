from typeddict_validator import DictMissingKeyException, DictValueTypeMismatchException, validate_typeddict
from typing import TypeVar, Optional, Type, List
from ai import parse_json
import logging


log = logging.getLogger(__name__)

T = TypeVar('T')

def validate_many(typed_dict: Type[T]):

    def run(raw: str) -> Optional[List[T]]:
        
        data = parse_json(raw)

        if not isinstance(data, list):
            return None

        for record in data:

            try:
                if not validate_typeddict(record, typed_dict, silent=False):
                    return None
            except DictMissingKeyException as e:
                log.info('MISSING '+ e.key)
                return None
            except DictValueTypeMismatchException as e:
                return None
                
        return data
    
    return run

def validate_one(typed_dict: Type[T]):

    def run(raw: str | dict) -> Optional[T]:

        if isinstance(raw, str):
            data = parse_json(raw)
        else:
            data = raw

        if not isinstance(data, dict):
            return None
        
        try:
            if validate_typeddict(data, typed_dict, silent=False):
                return data
        except DictMissingKeyException as e:
            return None
        except DictValueTypeMismatchException as e:
            return None
            # Note that: library has a "problem", int is not considered instance of float.
            # TODO: this is a tmp fix, if there is more than one problem it will not work!
            # if e.expected == float and e.actual == int:
            #     return data # pyright:ignore

        return None
    
    return run


def validate_one_from_path(typed_dict: Type[T], path: str) -> T:
    output = validate_one(typed_dict)(open(path, encoding="utf-8").read())
    assert output
    return output


def validate_many_from_path(typed_dict: Type[T], path: str) -> List[T]:
    output = validate_many(typed_dict)(open(path, encoding="utf-8").read())
    assert output
    return output
