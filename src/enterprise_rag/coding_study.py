from dataclasses import dataclass
from pathlib import Path

from typing import Any

@dataclass(frozen=True)
class CodingStudy:
    name: str
    age : Path


data = CodingStudy(name="John Doe", age=Path('src/enterprise_rag'))

print(data.name)  # Output: John Doe
if data.age.exists():
    print(f"The path {data.age.parent} exists.")
# print(data.age)   # Output: src/enterprise_rag


my_dict: dict[str, Any] = {
    "name": "John Doe",
    "age": [30, 50]}
print(my_dict["name"])  # Output: John Doe
print(my_dict["age"])   # Output: [30, 50]