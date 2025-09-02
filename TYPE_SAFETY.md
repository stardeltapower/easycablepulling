# Type Safety Guidelines

## Always Use Type Annotations

1. **Function return types**: Always annotate function return types
   ```python
   def process_data() -> List[str]:  # Good
   def process_data():              # Bad
   ```

2. **Numpy conversions**: Cast numpy types to Python types when needed
   ```python
   result = float(np_value)         # Good
   result = np_value               # Bad - may cause mypy errors
   ```

3. **Generic types**: Use proper generic typing for collections
   ```python
   primitives: List[Primitive] = cast(List[Primitive], straight_list)  # Good
   primitives = straight_list      # Bad - type mismatch
   ```

4. **Optional handling**: Explicitly handle Optional types
   ```python
   max_val = float(max(values)) if values else 0.0  # Good
   max_val = max(values) if values else 0.0         # Bad - type ambiguity
   ```

## Common Patterns

- Use `cast()` from typing module for safe type conversions
- Use `float()`, `int()`, `str()` to convert numpy types
- Always annotate return types, especially for class methods
- Handle Optional types explicitly with proper null checks

This ensures clean mypy type checking and better code reliability.
