import logging

from colorama import Fore, Style


class ColoredFormatter(logging.Formatter):
    def format(self, record):
        # Define colors for different log levels
        log_colors = {
            "DEBUG": Fore.BLUE,
            "INFO": Fore.GREEN,
            "WARNING": Fore.YELLOW,
            "ERROR": Fore.RED,
            "CRITICAL": Fore.RED + Style.BRIGHT,
        }

        levelname = record.levelname
        message = record.getMessage()

        log_color = log_colors.get(levelname, Fore.WHITE)

        message_parts = message.split(" ")
        message_parts[0] = f"{log_color}{message_parts[0]}{Style.RESET_ALL}"
        colored_message = " ".join(message_parts)

        record.msg = colored_message

        return super().format(record)
