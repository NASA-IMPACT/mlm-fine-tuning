from typing import Any, Optional, TextIO


def printd(*args: Any, **kwargs: Any) -> None:
    """
    Prints the provided arguments. If the 'file' keyword argument is provided,
    it prints to the file and then to the default standard output.

    Args:
        *args: Positional arguments to be printed.
        **kwargs: Keyword arguments for the `print` function.
    """
    file: Optional[TextIO] = kwargs.get("file", None)

    # Print to the specified file if provided
    print(*args, **kwargs)

    # If 'file' is provided, remove it and print to default stdout
    if file is not None:
        del kwargs["file"]
        print(*args, **kwargs)
