#!/usr/bin/env python3
"""Script to systematically fix common mypy error patterns."""

import re
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple


def get_mypy_errors() -> List[str]:
    """Get all mypy errors."""
    try:
        result = subprocess.run(
            ["python", "-m", "mypy", "easycablepulling", "--show-error-codes"],
            capture_output=True,
            text=True,
            cwd=".",
        )
        return result.stdout.split("\n")
    except Exception as e:
        print(f"Error running mypy: {e}")
        return []


def fix_missing_return_annotations(file_path: Path) -> int:
    """Fix missing return type annotations."""
    content = file_path.read_text()
    original_content = content

    # Pattern: def function(...): without -> return type
    patterns = [
        (r"(\n    def __init__\([^)]*\)):", r"\1 -> None:"),
        (
            r"(\n    def [a-zA-Z_][a-zA-Z0-9_]*\([^)]*\)):\s*\n\s*\"\"\"[^\"]*\"\"\"",
            lambda m: f"{m.group(1)} -> None:\n        {m.group(0).split(':', 1)[1].strip()}",
        ),
    ]

    fixes = 0
    for pattern, replacement in patterns:
        new_content = re.sub(pattern, replacement, content)
        if new_content != content:
            fixes += len(re.findall(pattern, content))
            content = new_content

    if content != original_content:
        file_path.write_text(content)
        print(f"  Fixed {fixes} missing return annotations in {file_path}")

    return fixes


def fix_int_float_assignments() -> int:
    """Fix int/float assignment errors by changing int initializers to float."""
    fixes = 0

    # Common patterns from mypy output
    files_and_fixes = [
        (
            "easycablepulling/reporting/json_reporter.py",
            [
                ("total_primitives = 0", "total_primitives = 0"),  # Keep as int
                ("total_straight_length = 0", "total_straight_length = 0.0"),
                ("total_bend_angle = 0", "total_bend_angle = 0.0"),
            ],
        ),
        (
            "easycablepulling/reporting/csv_reporter.py",
            [
                ("cumulative_length = 0", "cumulative_length = 0.0"),
            ],
        ),
    ]

    for file_path, replacements in files_and_fixes:
        path = Path(file_path)
        if path.exists():
            content = path.read_text()
            original_content = content

            for old, new in replacements:
                content = content.replace(old, new)

            if content != original_content:
                path.write_text(content)
                fixes += len(replacements)
                print(f"  Fixed int/float assignments in {path}")

    return fixes


def fix_primitive_attribute_access() -> int:
    """Fix Primitive attribute access by adding proper isinstance checks."""
    fixes = 0

    files = [
        "easycablepulling/reporting/csv_reporter.py",
        "easycablepulling/reporting/json_reporter.py",
    ]

    for file_path in files:
        path = Path(file_path)
        if not path.exists():
            continue

        content = path.read_text()
        original_content = content

        # Add imports if missing
        if "from ..core.models import" in content and "Straight" not in content:
            content = re.sub(
                r"from \.\.core\.models import ([^\\n]+)",
                r"from ..core.models import Bend, Straight, \1",
                content,
            )
            fixes += 1

        # Replace hasattr with isinstance
        replacements = [
            (r'hasattr\(primitive, "length_m"\)', "isinstance(primitive, Straight)"),
            (r'hasattr\(primitive, "radius_m"\)', "isinstance(primitive, Bend)"),
            (r'hasattr\(primitive, "angle_deg"\)', "isinstance(primitive, Bend)"),
        ]

        for old_pattern, new_text in replacements:
            if re.search(old_pattern, content):
                content = re.sub(old_pattern, new_text, content)
                fixes += 1

        if content != original_content:
            path.write_text(content)
            print(f"  Fixed primitive attribute access in {path}")

    return fixes


def add_missing_imports() -> int:
    """Add missing imports based on usage."""
    fixes = 0

    import_fixes = {
        "easycablepulling/reporting/csv_reporter.py": [
            (
                "from ..core.models import Route",
                "from ..core.models import Bend, Route, Straight",
            ),
        ],
        "easycablepulling/reporting/json_reporter.py": [
            (
                "from ..core.models import Route",
                "from ..core.models import Bend, Route, Straight",
            ),
        ],
    }

    for file_path, replacements in import_fixes.items():
        path = Path(file_path)
        if path.exists():
            content = path.read_text()
            original_content = content

            for old_import, new_import in replacements:
                if old_import in content and new_import not in content:
                    content = content.replace(old_import, new_import)
                    fixes += 1

            if content != original_content:
                path.write_text(content)
                print(f"  Added missing imports in {path}")

    return fixes


def main():
    """Main function to fix mypy errors systematically."""
    print("🔧 Starting systematic mypy error fixes...")

    # Get initial error count
    print("\n📊 Getting initial error count...")
    errors_before = get_mypy_errors()
    error_count_before = len([e for e in errors_before if "error:" in e])
    print(f"   Initial errors: {error_count_before}")

    total_fixes = 0

    # Fix 1: Missing imports
    print("\n1️⃣  Adding missing imports...")
    fixes = add_missing_imports()
    total_fixes += fixes

    # Fix 2: Primitive attribute access
    print("\n2️⃣  Fixing primitive attribute access...")
    fixes = fix_primitive_attribute_access()
    total_fixes += fixes

    # Fix 3: Int/float assignments
    print("\n3️⃣  Fixing int/float assignment errors...")
    fixes = fix_int_float_assignments()
    total_fixes += fixes

    # Fix 4: Missing return annotations
    print("\n4️⃣  Adding missing return type annotations...")
    python_files = list(Path("easycablepulling").rglob("*.py"))
    for file_path in python_files:
        fixes = fix_missing_return_annotations(file_path)
        total_fixes += fixes

    # Get final error count
    print("\n📊 Getting final error count...")
    errors_after = get_mypy_errors()
    error_count_after = len([e for e in errors_after if "error:" in e])

    print(f"\n🎯 Results:")
    print(f"   Errors before: {error_count_before}")
    print(f"   Errors after:  {error_count_after}")
    print(f"   Errors fixed:  {error_count_before - error_count_after}")
    print(f"   Total fixes applied: {total_fixes}")

    if error_count_after < error_count_before:
        print("✅ Successfully reduced mypy errors!")
    else:
        print("⚠️  Error count didn't decrease - may need manual fixes")


if __name__ == "__main__":
    main()
