from __future__ import annotations

import subprocess
import sys
import unittest


class ServeModuleTest(unittest.TestCase):
    def test_missing_fastapi_prints_helpful_message(self) -> None:
        """When fastapi is not installed, the module should exit with a helpful message."""
        result = subprocess.run(
            [sys.executable, "-c", "import inside_airbnb_serve"],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("fastapi", result.stdout)

    def test_module_syntax_is_valid(self) -> None:
        """The module should compile without syntax errors."""
        import py_compile
        py_compile.compile("inside_airbnb_serve.py", doraise=True)


if __name__ == "__main__":
    unittest.main()
