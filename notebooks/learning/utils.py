def printd(*args, **kwargs):
    if kwargs.get("file", None) is None:
        print(*args, **kwargs)
    else:
        print(*args, **kwargs)
        del kwargs["file"]
        print(*args, **kwargs)
