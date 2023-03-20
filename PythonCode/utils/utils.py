import importlib


def import_attr(module_and_attr_name):
    """
    Import an attr (class or function) from a (python) module, e.g. 'models.RNN' (taken from the Full Stack DL course)
    Args:
        module_and_attr_name: if list of str, try to import each attr in order and return the first that can be imported
    """
    module_name, class_name = module_and_attr_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    attr_ = getattr(module, class_name)

    return attr_
