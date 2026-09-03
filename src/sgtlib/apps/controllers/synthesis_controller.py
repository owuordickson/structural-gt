# SPDX-License-Identifier: GNU GPL v3
"""
Pyside6 (GUI components) controller class for network synthesis.
"""

import os
import sys
from pathlib import Path
from PySide6.QtCore import Slot, QObject, QProcess, QProcessEnvironment

from ...utils.config_loader import load_synthesis_configs
from ...utils.sgt_utils import ProgressData, verify_path

# NetworkSynth is an installable package, so it is launched as a module rather than a
# script path. PACKAGE_DIR is what has to be on PYTHONPATH for that import to resolve,
# and it doubles as the check that the checkout really holds NetworkSynth.
ENTRY_MODULE = "networksynth.gui_app"

# NetworkSynth reads a network from stdin when handed this flag, so the extracted
# graph goes straight over the pipe and neither side writes a file for it.
STDIN_FLAG = "--graph-from-stdin"
IMAGE_FLAG = "--image"
PACKAGE_DIR = os.path.join("src", "networksynth")

# Where 'git submodule update --init' puts NetworkSynth, so a source checkout needs no
# configuration at all. Anchored on this file rather than the working directory, which
# for a GUI is wherever the user happened to start it from.
DEFAULT_REPO_DIR = Path(__file__).resolve().parents[4] / "networksynth"

# A frozen build has no interpreter to lend, so an environment inside the checkout is the
# only thing left to run NetworkSynth with. Either platform's layout.
VENV_PYTHON = (Path(".venv", "bin", "python"), Path(".venv", "Scripts", "python.exe"))

# The submodule is marked 'update = none', so it is fetched only when asked for by name:
# NetworkSynth is a private repository, and a plain clone of this one must not fail for
# somebody who has no access to it.
FETCH_COMMAND = "git submodule update --init --checkout networksynth"

# Qt finds its plugins through these, and a frozen build's point inside its own bundle,
# which the child cannot use. Sharing an interpreter makes our PySide6 its PySide6, so
# clearing them is harmless there and Qt's own discovery finds the same files.
QT_ENV_VARS = ("QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH", "QML_IMPORT_PATH", "QML2_IMPORT_PATH")

# Enough of the tail to carry a traceback's last frame and its message.
STDERR_TAIL_LINES = 5


class SynthesisController(QObject):
    """
    Opens NetworkSynth, which generates synthetic networks modelled on an extracted graph.

    NetworkSynth is a separate program, run by this application's own interpreter from a
    source checkout. The two never import each other: NetworkSynth starts worker processes
    of its own and pins thread counts at import, neither of which belongs inside a Qt
    application. The user picks the inputs in NetworkSynth's own window, and it writes its
    results to the output folder chosen there.
    """

    def __init__(self, controller_obj, parent: QObject|None = None):
        super().__init__(parent)
        self._ctrl = controller_obj
        self._process: QProcess|None = None

        configs = load_synthesis_configs()
        self._repo_dir = configs["repo_dir"] or self._submodule_dir()
        self._interpreter = configs["python_interpreter"] or self._resolve_interpreter()

    @staticmethod
    def _submodule_dir() -> str:
        """The bundled checkout, when there is one."""
        return str(DEFAULT_REPO_DIR) if DEFAULT_REPO_DIR.is_dir() else ""

    def _resolve_interpreter(self) -> str:
        """The interpreter to run NetworkSynth with, or an empty string when there is none.

        NetworkSynth needs the same Python 3.14 this application does and pins every
        dependency they share to the same version, so a source checkout runs it with our
        own interpreter. A frozen sys.executable is this application, not a Python, and
        would relaunch StructuralGT instead, so a frozen build falls back to a virtual
        environment inside the checkout.
        """
        if not getattr(sys, "frozen", False):
            return sys.executable
        if self._repo_dir == "":
            return ""
        for relative_path in VENV_PYTHON:
            candidate = Path(self._repo_dir) / relative_path
            if candidate.is_file():
                return str(candidate)
        return ""

    @property
    def package_dir(self) -> str:
        """The package inside the checkout, which is what we import."""
        return os.path.join(self._repo_dir, PACKAGE_DIR) if self._repo_dir else ""

    @Slot(result=str)
    def unavailable_reason(self) -> str:
        """Why synthesis cannot run, or an empty string when it can."""
        if self._repo_dir == "":
            return (f"NetworkSynth is not in {DEFAULT_REPO_DIR}. Fetch it with "
                    f"'{FETCH_COMMAND}' - it is a private repository, so this needs an "
                    "SSH key with access to it. Or name where it already is with "
                    "'repo_dir' under [synthesis-settings] in the config file.")
        if not verify_path(self.package_dir)[0]:
            if self._repo_dir == str(DEFAULT_REPO_DIR):
                return (f"{DEFAULT_REPO_DIR} holds no {PACKAGE_DIR}. "
                        f"Fetch NetworkSynth with '{FETCH_COMMAND}'.")
            return f"No {PACKAGE_DIR} in {self._repo_dir}."
        if self._interpreter == "":
            return (f"This build has no interpreter to run NetworkSynth with. Make one in "
                    f"{os.path.join(self._repo_dir, '.venv')}, or name one with "
                    "'python_interpreter' under [synthesis-settings] in the config file.")
        if not verify_path(self._interpreter)[0]:
            return f"No Python interpreter at {self._interpreter}."
        return ""

    @Slot(result=bool)
    def is_available(self) -> bool:
        """True when NetworkSynth and an interpreter to run it with are both in place."""
        return self.unavailable_reason() == ""

    @Slot(result=str)
    def tooltip_text(self) -> str:
        reason = self.unavailable_reason()
        return "Generate synthetic networks" if reason == "" else f"Synthesis unavailable: {reason}"

    def _extracted_graph(self):
        """The graph currently in view, or None if nothing has been extracted yet."""
        sgt_obj = self._ctrl.get_selected_sgt_obj()
        if sgt_obj is None:
            return None, ""
        ntwk_p = sgt_obj.ntwk_p
        graph_obj = ntwk_p.graph_obj
        graph = graph_obj.nx_graph if graph_obj is not None else None
        return graph, (ntwk_p.img_path or "")

    @staticmethod
    def _as_graphml(graph) -> bytes:
        """Serialise to a buffer. Positions live in 'o', which GraphML cannot hold,
        so they are written out as plain x and y attributes."""
        import io

        import networkx as nx

        exported = nx.Graph()
        for node in graph.nodes():

            y, x = graph.nodes[node]["o"][:2]
            exported.add_node(node, x=float(x), y=float(y))
        for source, target, data in graph.edges(data=True):
            weight = data.get("weight")
            if weight is None:
                exported.add_edge(source, target)
            else:
                exported.add_edge(source, target, weight=float(weight))

        buffer = io.BytesIO()
        nx.write_graphml(exported, buffer)
        return buffer.getvalue()

    @Slot()
    def open_synthesis_window(self):
        """Start NetworkSynth as a separate process and let it run on its own."""
        reason = self.unavailable_reason()
        if reason != "":
            self._ctrl.showAlertSignal.emit("Synthesis Unavailable", reason)
            return

        if self._process is not None and self._process.state() != QProcess.ProcessState.NotRunning:
            self._ctrl.showAlertSignal.emit("Synthesis Running", "The synthesis window is already open.")
            return

        env = QProcessEnvironment.systemEnvironment()
        for var_name in QT_ENV_VARS:
            env.remove(var_name)
        # The checkout is not installed, so the import path is handed over explicitly.
        source_root = os.path.join(self._repo_dir, "src")
        existing = env.value("PYTHONPATH")
        env.insert("PYTHONPATH", f"{source_root}{os.pathsep}{existing}" if existing else source_root)

        graph, image = self._extracted_graph()
        arguments = ["-m", ENTRY_MODULE]
        handover = b""
        if graph is not None and graph.number_of_nodes() > 0:
            handover = self._as_graphml(graph)
            arguments.append(STDIN_FLAG)
            if image:
                arguments += [IMAGE_FLAG, image]

        self._process = QProcess(self)
        self._process.setProgram(self._interpreter)
        self._process.setArguments(arguments)
        self._process.setWorkingDirectory(self._repo_dir)
        self._process.setProcessEnvironment(env)
        self._process.finished.connect(self.handle_synthesis_finished)
        self._process.errorOccurred.connect(self.handle_synthesis_error)
        self._process.start()
        if handover:
            self._process.write(handover)
            self._process.closeWriteChannel()
        opened = "Opening the synthesis window with the extracted network..."
        self._report("info", opened if handover else "Opening the synthesis window...")

    def handle_synthesis_finished(self, exit_code: int, exit_status) -> None:
        """Report how NetworkSynth ended, with its own last words when it ended badly."""
        crashed = exit_status == QProcess.ExitStatus.CrashExit
        if exit_code == 0 and not crashed:
            self._report("info", "Synthesis window closed.")
            return

        error_output = bytes(self._process.readAllStandardError()).decode(errors="replace")
        tail = [line for line in error_output.splitlines() if line.strip()][-STDERR_TAIL_LINES:]
        detail = "\n".join(tail) if tail else "no output"
        self._report("error", f"Synthesis exited with code {exit_code}:\n{detail}")

    def handle_synthesis_error(self, error) -> None:
        """A process that never started reports here instead of in 'finished'."""
        if error != QProcess.ProcessError.FailedToStart:
            return
        message = f"Could not start {self._interpreter}."
        self._report("error", message)
        self._ctrl.showAlertSignal.emit("Synthesis Error", message)

    def _report(self, msg_type: str, message: str) -> None:
        self._ctrl.handle_progress_update(ProgressData(type=msg_type, sender="GT", message=message))
