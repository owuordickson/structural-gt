# SPDX-License-Identifier: GNU GPL v3
# This file is dual licensed under the terms of the GNU General Public, Version
# 3.0.  See the LICENSE file in the root of this
# repository for complete details.

"""
Entry points that allow users to execute GUI or Cli programs
"""

import logging
from .apps.gui_main import pyqt_app
from .apps.cli_main import terminal_app


logger = logging.getLogger("SGT App")
FORMAT = '%(asctime)s; %(user)-8s. %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


def main_gui():
    """
    Start graphical user interface application.
    :return:
    """
    logging.basicConfig(filename='sgt_app.log', encoding='utf-8', level=logging.INFO,
                        format=FORMAT, datefmt=DATE_FORMAT)
    logging.info("SGT application started running...", extra={'user': 'SGT Logs'})
    pyqt_app()
    logging.info("SGT application stopped running.", extra={'user': 'SGT Logs'})


def main_cli():
    """
    Start terminal/CMD application.
    :return:
    """
    logging.basicConfig(filename='sgt_app.log', encoding='utf-8', level=logging.INFO,
                        format=FORMAT, datefmt=DATE_FORMAT)
    logging.info("SGT application started running...", extra={'user': 'SGT Logs'})
    terminal_app()
    logging.info("SGT application stopped running.", extra={'user': 'SGT Logs'})
