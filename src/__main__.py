# SPDX-License-Identifier: GNU GPL v3
# This file is dual licensed under the terms of the GNU General Public, Version
# 3.0.  See the LICENSE file in the root of this
# repository for complete details.

"""
A launcher for executing the application as a Window app or a Terminal app.
"""


from StructuralGTb.entrypoints import main_gui, main_cli

if __name__ == "__main__":
    main_gui()
    # main_cli()
