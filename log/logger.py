import sys, os
from logging.handlers import RotatingFileHandler
import logging


class FTSyncLogger:
    """
    Automatically write logs and ensure that the log files are stored in a rolling manner.

    use example:

    with FTSyncLogger("test.log"):
        print("hello")
    """

    def __init__(self, filename=None):
        self.terminal = sys.stdout
        self.logfile = None
        self.loghandle = None
        if filename is not None:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            self.loghandle = RotatingFileHandler(
                filename, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
            )
            self.loghandle.setFormatter(
                logging.Formatter(
                    "[%(asctime)s %(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )

            self.logfile = logging.getLogger(filename)
            self.logfile.setLevel(logging.INFO)
            self.logfile.addHandler(self.loghandle)

            self.logfile.info(
                "\n\n===================== new log =====================\n"
            )

        self.pre_stdout = None

    def write(self, message: str):
        if len(message) < 1:
            return
        self.write_file(message)
        self.write_terminal(message)

    def write_file(self, message: str):
        if self.logfile is not None:
            if message[-1] == "\n":
                message = message[:-1]

            if len(message) < 1:
                return
            self.logfile.info(message.strip())

    def write_terminal(self, message: str):
        self.terminal.write(message)

    def flush(self):
        self.terminal.flush()
        if self.loghandle is not None:
            self.loghandle.flush()

    def __del__(self):
        if self.logfile is not None:
            self.loghandle.flush()
            self.loghandle.close()
            self.loghandle = None
            self.logfile = None

    def __enter__(self):
        self.pre_stdout = sys.stdout
        sys.stdout = self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.logfile is not None:
            self.loghandle.flush()
            self.loghandle.close()
            self.loghandle = None
            self.logfile = None
        sys.stdout = self.pre_stdout
