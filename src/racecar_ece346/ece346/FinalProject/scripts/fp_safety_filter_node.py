#!/usr/bin/env python3
"""
Thin entry-point wrapper.
The actual implementation lives in safety_filter_node.py, which is
installed via ament_python_install_package and symlinked into the Python
path. This wrapper is installed directly (no CMake RENAME) so that
`--symlink-install` can also create a real symlink here. Together the two
symlinks mean edits to safety_filter_node.py are live without a rebuild.
"""
from ece346.FinalProject.scripts.safety_filter_node import main

if __name__ == '__main__':
    main()
