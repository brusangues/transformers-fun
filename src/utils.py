from contextlib import redirect_stdout
import sys


class DualLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log_file = open(filename, "a", encoding="utf-8")  # Open in append mode

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        # This flush method is needed for Python 3 compatibility
        # and ensures that output is written to both destinations.
        self.terminal.flush()
        self.log_file.flush()


# Example usage:
if __name__ == "__main__":
    # Redirect sys.stdout to our custom DualLogger
    with redirect_stdout(DualLogger("test.txt")):
        print("This message will go to both the console and output.log")
        print("Another line of output.")
    print("This wont")
    print("End")

    sys.stdout = DualLogger("output.log")

    print("This message will go to both the console and output.log")
    print("Another line of output.")

    # You can also write directly to the log file if needed
    # sys.stdout.log_file.write("This is a direct write to the log file.\n")

    # To restore original stdout (optional)
    # sys.stdout = sys.stdout.terminal