import sys, os


class FTSyncLogger:
    def __init__(self, filename=None, mode="a+"):
        self.terminal = sys.stdout
        if filename is not None:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            self.logfile = open(filename, mode)
        else:
            self.logfile = None

        self.pre_stdout = None

    def write(self, message):
        self.terminal.write(message)
        if self.logfile is not None:
            self.logfile.write(message)

    def flush(self):
        self.terminal.flush()
        if self.logfile is not None:
            self.logfile.flush()

    def __del__(self):
        if self.logfile is not None:
            self.logfile.flush()
            self.logfile.close()
            self.logfile = None

    def __enter__(self):
        self.pre_stdout = sys.stdout
        sys.stdout = self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.logfile is not None:
            self.logfile.flush()
            self.logfile.close()
            self.logfile = None
        sys.stdout = self.pre_stdout


class FTSyncLoggerParallel(FTSyncLogger):
    def __init__(
        self, filename=None, mode="a+", only_master_print=False, global_rank=-1
    ):
        super().__init__(filename, mode)

        self.rank = global_rank
        self.only_master_print = only_master_print
        self.prompt = f"\033[32m[node-{self.rank}]\033[0m "

        self.next_prompt = self.prompt

    def write(self, message):
        # whether print to terminal
        if len(message) < 1:
            return

        message = self.next_prompt + message
        if message[-1] == "\n":
            self.next_prompt = self.prompt
            print_message = message[:-1].replace("\n", "\n" + self.prompt) + "\n"
        else:
            self.next_prompt = ""
            print_message = message.replace("\n", "\n" + self.prompt)

        if self.only_master_print:
            if self.rank == 0:
                self.terminal.write(print_message)
        else:
            self.terminal.write(print_message)

        # only master node write log file
        if self.rank == 0:
            if self.logfile is not None:
                self.logfile.write(message)
