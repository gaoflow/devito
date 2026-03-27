import inspect

__all__ = ['GenericVisitor']


class GenericVisitor:

    _handler_funcs_cache = {}
    _handlers_cache = {}

    """
    A generic visitor.

    To define handlers, subclasses should define :data:`visit_Foo`
    methods for each class :data:`Foo` they want to handle.
    If a specific method for a class :data:`Foo` is not found, the MRO
    of the class is walked in order until a matching method is found.

    The method signature is:

        .. code-block::
           def visit_Foo(self, o, [*args, **kwargs]):
               pass

    The handler is responsible for visiting the children (if any) of
    the node :data:`o`.  :data:`*args` and :data:`**kwargs` may be
    used to pass information up and down the call stack.  You can also
    pass named keyword arguments, e.g.:

        .. code-block::
           def visit_Foo(self, o, parent=None, *args, **kwargs):
               pass
    """

    def __init__(self):
        cls = type(self)

        try:
            self._handlers = self._handlers_cache[cls]
        except KeyError:
            self._handlers = {}
            self._handlers_cache[cls] = self._handlers

        try:
            self._handler_funcs = self._handler_funcs_cache[cls]
        except KeyError:
            self._handler_funcs = self._build_handler_funcs()
            self._handler_funcs_cache[cls] = self._handler_funcs

    @classmethod
    def _build_handler_funcs(cls):
        handlers = {}
        prefix = "visit_"
        for name, meth in inspect.getmembers(cls, predicate=inspect.isfunction):
            if not name.startswith(prefix):
                continue
            argspec = inspect.getfullargspec(meth)
            if len(argspec.args) < 2:
                raise RuntimeError("Visit method signature must be "
                                   "visit_Foo(self, o, [*args, **kwargs])")
            handlers[name[len(prefix):]] = meth
        return handlers

    """
    :attr:`default_args`. A dict of default keyword arguments for the visitor.
    These are not used by default in :meth:`visit`, however, a caller may pass
    them explicitly to :meth:`visit` by accessing :attr:`default_args`.
    For example::

        .. code-block::
           v = FooVisitor()
           v.visit(node, **v.default_args)
    """
    default_args = {}

    @classmethod
    def default_retval(cls):
        """
        A method that returns an object to use to populate return values.

        If your visitor combines values in a tree-walk, it may be useful to
        provide a object to combine the results into. :meth:`default_retval`
        may be defined by the visitor to be called to provide an empty object
        of appropriate type.
        """
        return None

    def lookup_method(self, instance):
        """
        Look up a handler method for a visitee.

        Parameters
        ----------
        instance : object
            The instance to look up a method for.
        """
        cls = instance.__class__
        try:
            return self._handlers[cls]
        except KeyError:
            cls_name = cls.__name__
            try:
                entry = self._handler_funcs[cls_name]
            except KeyError:
                for klass in cls.mro()[1:]:
                    entry = self._handler_funcs.get(klass.__name__)
                    if entry is not None:
                        self._handler_funcs[cls_name] = entry
                        break
                else:
                    raise RuntimeError("No handler found for class %s", cls_name)
            self._handlers[cls] = entry
            return entry

    def visit(self, o, *args, **kwargs):
        """
        Apply this Visitor to an object.

        Parameters
        ----------
        o : object
            The object to be visited.
        *args
            Optional arguments to pass to the visit methods.
        **kwargs
            Optional keyword arguments to pass to the visit methods.
        """
        ret = self._visit(o, *args, **kwargs)
        ret = self._post_visit(ret)
        return ret

    def _visit(self, o, *args, **kwargs):
        """Visit ``o``."""
        meth = self.lookup_method(o)
        return meth(self, o, *args, **kwargs)

    def _post_visit(self, ret):
        """Postprocess the visitor output before returning it to the caller."""
        return ret

    def visit_object(self, o, **kwargs):
        return self.default_retval()
