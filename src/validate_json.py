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
            except DictValueTypeMismatchException:
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

        # The validator treats an int as NOT a float (library quirk). An LLM judge that
        # must return a float in [0, 1] often emits the bare integers 0 or 1 instead of
        # 0.0 / 1.0, which fails validation and forces a retry. Coerce ONLY these edge
        # values for float-typed fields; a genuinely malformed integer (e.g. 50) still
        # fails and retries as before. (type(value) is int excludes bool.)
        for key, expected in getattr(typed_dict, '__annotations__', {}).items():
            value = data.get(key)
            if expected is float and type(value) is int and value in (0, 1):
                data[key] = float(value)

        try:
            if validate_typeddict(data, typed_dict, silent=False):
                return data
        except DictMissingKeyException:
            return None
        except DictValueTypeMismatchException:
            return None

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
