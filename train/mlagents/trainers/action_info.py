from typing import NamedTuple, Any, Dict, Optional

ActionInfo = NamedTuple('ActionInfo', [('action', Any), ('memory', Any), ('text', Any), ('value', Any), ('outputs', Optional[Dict[str, Any]])])
