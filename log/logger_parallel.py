from .logger import FTSyncLogger

class FTSyncLoggerParallel(FTSyncLogger):
    """
    In a parallel environment, automatically distinguish the printed contents of different nodes, and the master node can automatically roll-over and store log files.

    use example:

    with FTSyncLoggerParallel("test.log", False, 0):    # global_rank need set to real rank
        print("hello")
    """

    def __init__(
        self,
        filename=None,
        only_master_print=False,
        global_rank=-1,
    ):
        super().__init__(filename)

        self.rank = global_rank
        self.only_master_print = only_master_print
        self.prompt = f"\033[32m[node-{self.rank}]\033[0m "

        self.next_prompt = self.prompt

    def write(self, message: str):
        # whether print to terminal
        if len(message) < 1:
            return

        # only master node write log file
        if self.rank == 0:
            self.write_file(message)

        message = self.next_prompt + message
        if message[-1] == "\n":
            self.next_prompt = self.prompt
            print_message = message[:-1].replace("\n", "\n" + self.prompt) + "\n"
        else:
            self.next_prompt = ""
            print_message = message.replace("\n", "\n" + self.prompt)

        if self.only_master_print:
            if self.rank == 0:
                self.write_terminal(print_message)
        else:
            self.write_terminal(print_message)
