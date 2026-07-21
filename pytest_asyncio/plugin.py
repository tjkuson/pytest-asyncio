"""pytest-asyncio implementation."""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import dataclasses
import enum
import functools
import inspect
import socket
import sys
import traceback
import warnings
from asyncio import AbstractEventLoop
from collections.abc import (
    Callable,
    Coroutine,
    Generator,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from types import AsyncGeneratorType
from typing import (
    Any,
    Literal,
    ParamSpec,
    TypeAlias,
    TypeVar,
    overload,
)

import pluggy

import pytest
from _pytest.nodes import Node
from _pytest.scope import Scope
from pytest import (
    Config,
    FixtureDef,
    Function,
    Item,
    Mark,
    MonkeyPatch,
    Parser,
    PytestCollectionWarning,
    PytestPluginManager,
)

from ._pytest_compat import (
    FixtureInfo,
    get_fixture_info,
    get_requesting_item,
    replace_fixture_function,
)

if sys.version_info >= (3, 11):
    from asyncio import Runner
else:
    from backports.asyncio.runner import Runner

_ScopeName = Literal["session", "package", "module", "class", "function"]
_R = TypeVar("_R")
_P = ParamSpec("_P")
FixtureFunction = Callable[_P, _R]
LoopFactory: TypeAlias = Callable[[], AbstractEventLoop]
_LOOP_FACTORY_PARAM = "_pytest_asyncio_loop_factory"
_CollectResult: TypeAlias = (
    "pytest.Item | pytest.Collector | list[pytest.Item | pytest.Collector] | None"
)


@dataclasses.dataclass(frozen=True, eq=False, slots=True)
class _LoopFactoryVariant:
    """A stable, named factory parameter whose identity is part of loop identity."""

    name: str | None
    factory: LoopFactory | None


class PytestAsyncioError(Exception):
    """Base class for exceptions raised by pytest-asyncio"""


class PytestAsyncioWarning(pytest.PytestWarning):
    """Warning emitted for suspicious pytest-asyncio usage."""


class Mode(str, enum.Enum):
    AUTO = "auto"
    STRICT = "strict"


hookspec = pluggy.HookspecMarker("pytest")


class PytestAsyncioSpecs:
    @hookspec(firstresult=True)
    def pytest_asyncio_loop_factories(
        self,
        config: Config,
        item: Item,
    ) -> Mapping[str, LoopFactory] | None:
        raise NotImplementedError  # pragma: no cover


ASYNCIO_MODE_HELP = """\
'auto' - for automatically handling all async functions by the plugin
'strict' - for autoprocessing disabling (useful if different async frameworks \
should be tested together, e.g. \
both pytest-asyncio and pytest-trio are used in the same project)
"""


def pytest_addoption(parser: Parser, pluginmanager: PytestPluginManager) -> None:
    pluginmanager.add_hookspecs(PytestAsyncioSpecs)
    group = parser.getgroup("asyncio")
    group.addoption(
        "--asyncio-mode",
        dest="asyncio_mode",
        default=None,
        metavar="MODE",
        help=ASYNCIO_MODE_HELP,
    )
    group.addoption(
        "--asyncio-debug",
        dest="asyncio_debug",
        action="store_true",
        default=None,
        help="enable asyncio debug mode for the default event loop",
    )
    parser.addini(
        "asyncio_mode",
        help="default value for --asyncio-mode",
        default="strict",
    )
    parser.addini(
        "asyncio_debug",
        help="enable asyncio debug mode for the default event loop",
        type="bool",
        default="false",
    )
    parser.addini(
        "asyncio_default_fixture_loop_scope",
        type="string",
        help="default scope of the asyncio event loop used to execute async fixtures",
        default="function",
    )
    parser.addini(
        "asyncio_default_test_loop_scope",
        type="string",
        help="default scope of the asyncio event loop used to execute tests",
        default="function",
    )


@overload
def fixture(
    fixture_function: FixtureFunction[_P, _R],
    *,
    scope: _ScopeName | Callable[[str, Config], _ScopeName] = ...,
    loop_scope: _ScopeName | None = ...,
    params: Iterable[object] | None = ...,
    autouse: bool = ...,
    ids: (
        Iterable[str | float | int | bool | None]
        | Callable[[Any], object | None]
        | None
    ) = ...,
    name: str | None = ...,
) -> FixtureFunction[_P, _R]: ...


@overload
def fixture(
    fixture_function: None = ...,
    *,
    scope: _ScopeName | Callable[[str, Config], _ScopeName] = ...,
    loop_scope: _ScopeName | None = ...,
    params: Iterable[object] | None = ...,
    autouse: bool = ...,
    ids: (
        Iterable[str | float | int | bool | None]
        | Callable[[Any], object | None]
        | None
    ) = ...,
    name: str | None = None,
) -> Callable[[FixtureFunction[_P, _R]], FixtureFunction[_P, _R]]: ...


def fixture(
    fixture_function: FixtureFunction[_P, _R] | None = None,
    loop_scope: _ScopeName | None = None,
    **kwargs: Any,
) -> (
    FixtureFunction[_P, _R]
    | Callable[[FixtureFunction[_P, _R]], FixtureFunction[_P, _R]]
):
    if fixture_function is not None:
        wrapped_fixture = _create_asyncio_fixture_wrapper(
            fixture_function,
            loop_scope,
        )
        return pytest.fixture(wrapped_fixture, **kwargs)

    else:

        @functools.wraps(fixture)
        def inner(fixture_function: FixtureFunction[_P, _R]) -> FixtureFunction[_P, _R]:
            return fixture(fixture_function, loop_scope=loop_scope, **kwargs)

        return inner


def _is_asyncio_fixture_function(obj: Any) -> bool:
    obj = getattr(obj, "__func__", obj)  # instance method maybe?
    return getattr(obj, "_force_asyncio_fixture", False)


def _make_asyncio_fixture_function(obj: Any, loop_scope: _ScopeName | None) -> None:
    if hasattr(obj, "__func__"):
        # instance method, check the function object
        obj = obj.__func__
    obj._force_asyncio_fixture = True
    obj._loop_scope = loop_scope


def _is_coroutine_or_asyncgen(obj: Any) -> bool:
    return inspect.iscoroutinefunction(obj) or inspect.isasyncgenfunction(obj)


def _get_asyncio_mode(config: Config) -> Mode:
    val = config.getoption("asyncio_mode")
    if val is None:
        val = config.getini("asyncio_mode")
    try:
        return Mode(val)
    except ValueError as e:
        modes = ", ".join(m.value for m in Mode)
        raise pytest.UsageError(
            f"{val!r} is not a valid asyncio_mode. Valid modes: {modes}."
        ) from e


def _get_asyncio_debug(config: Config) -> bool:
    val = config.getoption("asyncio_debug")
    if val is None:
        val = config.getini("asyncio_debug")

    if isinstance(val, bool):
        return val
    else:
        return val == "true"


_INVALID_LOOP_FACTORIES = """\
pytest_asyncio_loop_factories must return a non-empty mapping of \
factory names to callables.
"""


def _collect_hook_loop_factories(
    config: Config,
    item: Item,
) -> dict[str, LoopFactory] | None:
    hook_caller = item.ihook.pytest_asyncio_loop_factories
    hookimpls = hook_caller.get_hookimpls()
    if not hookimpls:
        return None

    result = hook_caller(config=config, item=item)
    if result is None or not isinstance(result, Mapping):
        raise pytest.UsageError(_INVALID_LOOP_FACTORIES)
    # Copy into an isolated snapshot so later mutations of the hook's
    # original container do not affect parametrization.
    factories = dict(result)
    if not factories or any(
        not isinstance(name, str) or not name or not callable(factory)
        for name, factory in factories.items()
    ):
        raise pytest.UsageError(_INVALID_LOOP_FACTORIES)
    return factories


def _validate_scope(scope: str | None, option_name: str) -> None:
    if scope is None:
        return
    valid_scopes = [s.value for s in Scope]
    if scope not in valid_scopes:
        raise pytest.UsageError(
            f"{scope!r} is not a valid {option_name}. "
            f"Valid scopes are: {', '.join(valid_scopes)}."
        )


_RUNNER_TEARDOWN_WARNING = """\
An exception occurred during teardown of an asyncio.Runner. \
The reason is likely that you closed the underlying event loop in a test, \
which prevents the cleanup of asynchronous generators by the runner.
This warning will become an error in future versions of pytest-asyncio. \
Please ensure that your tests don't close the event loop. \
Here is the traceback of the exception triggered during teardown:
%s
"""


class _ManagedRunner:
    """An asyncio runner owned by the hook-based loop manager."""

    def __init__(self, *, debug: bool, factory: LoopFactory | None) -> None:
        self._runner = Runner(debug=debug, loop_factory=factory)
        self._runner.__enter__()

    @property
    def loop(self) -> AbstractEventLoop:
        return self._runner.get_loop()

    def run(
        self,
        awaitable: Coroutine[Any, Any, Any],
        *,
        context: contextvars.Context | None = None,
    ) -> Any:
        with _temporary_event_loop(self.loop):
            return self._runner.run(awaitable, context=context)

    def close(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", ".*BaseEventLoop.shutdown_asyncgens.*", RuntimeWarning
            )
            try:
                self._runner.__exit__(None, None, None)
            except RuntimeError:
                warnings.warn(
                    _RUNNER_TEARDOWN_WARNING % traceback.format_exc(),
                    RuntimeWarning,
                )


def _scope_root(node: Node, scope: _ScopeName) -> Node:
    if scope == "function":
        return node
    node_type: type[Node]
    if scope == "class":
        node_type = pytest.Class
    elif scope == "module":
        node_type = pytest.Module
    elif scope == "package":
        node_type = pytest.Package
    else:
        return node.session
    root = node.getparent(node_type)
    if root is not None:
        return root
    # Match pytest fixture-scope fallbacks: a missing class is item-local, while
    # a missing package behaves like session scope.
    return node if scope == "class" else node.session


class _LoopManager:
    """Own scoped runners independently from pytest fixtures."""

    def __init__(self, config: Config) -> None:
        self._debug = _get_asyncio_debug(config)
        self._active: dict[
            tuple[_ScopeName, Node],
            tuple[_LoopFactoryVariant, _ManagedRunner],
        ] = {}

    def get_runner(
        self,
        node: Node,
        scope: _ScopeName,
        variant: _LoopFactoryVariant,
    ) -> _ManagedRunner:
        root = _scope_root(node, scope)
        key = (scope, root)
        active = self._active.get(key)
        if active is not None:
            active_factory, runner = active
            if active_factory is variant:
                return runner
            del self._active[key]
            runner.close()
        runner = _ManagedRunner(debug=self._debug, factory=variant.factory)
        self._active[key] = (variant, runner)
        root.addfinalizer(functools.partial(self._close_runner, key, runner))
        return runner

    def _close_runner(
        self,
        key: tuple[_ScopeName, Node],
        runner: _ManagedRunner,
    ) -> None:
        active = self._active.get(key)
        if active is None or active[1] is not runner:
            return
        del self._active[key]
        runner.close()

    def close_all(self) -> None:
        for _, runner in reversed(tuple(self._active.values())):
            runner.close()
        self._active.clear()


_LOOP_MANAGER_KEY = pytest.StashKey[_LoopManager]()
_WARNED_SCOPE_EDGES_KEY = pytest.StashKey[set[tuple[object, FixtureDef]]]()
_LOOP_FACTORY_VARIANTS_KEY = pytest.StashKey[list[_LoopFactoryVariant]]()


def _get_loop_manager(config: Config) -> _LoopManager:
    return config.stash[_LOOP_MANAGER_KEY]


def pytest_configure(config: Config) -> None:
    default_fixture_loop_scope = config.getini("asyncio_default_fixture_loop_scope")
    _validate_scope(default_fixture_loop_scope, "asyncio_default_fixture_loop_scope")
    default_test_loop_scope = config.getini("asyncio_default_test_loop_scope")
    _validate_scope(default_test_loop_scope, "asyncio_default_test_loop_scope")
    config.addinivalue_line(
        "markers",
        "asyncio: "
        "mark the test as a coroutine, it will be "
        "run using an asyncio event loop",
    )
    config.stash[_LOOP_MANAGER_KEY] = _LoopManager(config)
    config.stash[_WARNED_SCOPE_EDGES_KEY] = set()
    config.stash[_LOOP_FACTORY_VARIANTS_KEY] = []


def pytest_unconfigure(config: Config) -> None:
    manager = config.stash.get(_LOOP_MANAGER_KEY, None)
    if manager is not None:
        manager.close_all()


@pytest.hookimpl(tryfirst=True)
def pytest_report_header(config: Config) -> list[str]:
    """Add asyncio config to pytest header."""
    mode = _get_asyncio_mode(config)
    debug = _get_asyncio_debug(config)
    default_fixture_loop_scope = config.getini("asyncio_default_fixture_loop_scope")
    default_test_loop_scope = _get_default_test_loop_scope(config)
    header = [
        f"mode={mode}",
        f"debug={debug}",
        f"asyncio_default_fixture_loop_scope={default_fixture_loop_scope}",
        f"asyncio_default_test_loop_scope={default_test_loop_scope}",
    ]
    return [
        "asyncio: " + ", ".join(header),
    ]


def _fixture_wrapper_signature(fixture_function: Callable) -> inspect.Signature:
    """Add pytest-asyncio's two internal fixture dependencies."""
    signature = inspect.signature(fixture_function)
    parameters = list(signature.parameters.values())
    parameter_names = signature.parameters
    if _LOOP_FACTORY_PARAM in parameter_names:
        raise pytest.UsageError(
            f"{_LOOP_FACTORY_PARAM!r} is reserved for use by pytest-asyncio."
        )
    if "request" not in parameter_names:
        request_parameter = inspect.Parameter(
            "request",
            kind=inspect.Parameter.KEYWORD_ONLY,
        )
        var_keyword_index = next(
            (
                index
                for index, parameter in enumerate(parameters)
                if parameter.kind is inspect.Parameter.VAR_KEYWORD
            ),
            len(parameters),
        )
        parameters.insert(var_keyword_index, request_parameter)
    factory_parameter = inspect.Parameter(
        _LOOP_FACTORY_PARAM,
        kind=inspect.Parameter.KEYWORD_ONLY,
    )
    var_keyword_index = next(
        (
            index
            for index, parameter in enumerate(parameters)
            if parameter.kind is inspect.Parameter.VAR_KEYWORD
        ),
        len(parameters),
    )
    parameters.insert(var_keyword_index, factory_parameter)
    return signature.replace(parameters=parameters)


def _fixture_runner(
    loop_scope: _ScopeName | None,
    fixture_accepts_request: bool,
    kwargs: dict[str, Any],
) -> tuple[_ManagedRunner, pytest.FixtureRequest]:
    request = kwargs["request"]
    variant = kwargs.pop(_LOOP_FACTORY_PARAM)
    if not isinstance(request, pytest.FixtureRequest):
        raise TypeError("pytest-asyncio received an invalid fixture request")
    if not isinstance(variant, _LoopFactoryVariant):
        raise TypeError("pytest-asyncio received an invalid loop-factory token")
    if not fixture_accepts_request:
        del kwargs["request"]
    effective_loop_scope = loop_scope or request.config.getini(
        "asyncio_default_fixture_loop_scope"
    )
    if Scope(request.scope) > Scope(effective_loop_scope):
        pytest.fail(
            f"ScopeMismatch: fixture {request.fixturename!r} has caching scope "
            f"{request.scope!r}, but its event loop scope is "
            f"{effective_loop_scope!r}. The event loop scope must be at least "
            "as wide as the fixture scope.",
            pytrace=False,
        )
    item = get_requesting_item(request)
    runner = _get_loop_manager(request.config).get_runner(
        item,
        effective_loop_scope,
        variant,
    )
    return runner, request


def _create_asyncio_fixture_wrapper(
    fixture_function: FixtureFunction[_P, _R],
    loop_scope: _ScopeName | None,
) -> FixtureFunction[_P, _R]:
    """Create a synchronous pytest fixture with explicit internal dependencies."""
    fixture_accepts_request = (
        "request" in inspect.signature(fixture_function).parameters
    )
    wrapper: Callable[..., Any]
    if inspect.isasyncgenfunction(fixture_function):

        @functools.wraps(fixture_function)
        def asyncgen_wrapper(*args: Any, **kwargs: Any) -> Any:
            runner, request = _fixture_runner(
                loop_scope,
                fixture_accepts_request,
                kwargs,
            )
            synchronized = _wrap_asyncgen_fixture(
                fixture_function,
                runner,
                request,
            )
            return synchronized(*args, **kwargs)

        wrapper = asyncgen_wrapper
    elif inspect.iscoroutinefunction(fixture_function):

        @functools.wraps(fixture_function)
        def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            runner, request = _fixture_runner(
                loop_scope,
                fixture_accepts_request,
                kwargs,
            )
            synchronized = _wrap_async_fixture(fixture_function, runner, request)
            return synchronized(*args, **kwargs)

        wrapper = async_wrapper
    elif inspect.isgeneratorfunction(fixture_function):

        @functools.wraps(fixture_function)
        def syncgen_wrapper(*args: Any, **kwargs: Any) -> Generator[Any]:
            runner, _ = _fixture_runner(
                loop_scope,
                fixture_accepts_request,
                kwargs,
            )
            yield from _wrap_syncgen_fixture(fixture_function, runner)(*args, **kwargs)

        wrapper = syncgen_wrapper
    else:

        @functools.wraps(fixture_function)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            runner, _ = _fixture_runner(
                loop_scope,
                fixture_accepts_request,
                kwargs,
            )
            return _wrap_sync_fixture(fixture_function, runner)(*args, **kwargs)

        wrapper = sync_wrapper

    wrapper_with_signature: Any = wrapper
    wrapper_with_signature.__signature__ = _fixture_wrapper_signature(fixture_function)
    _make_asyncio_fixture_function(wrapper, loop_scope)
    return wrapper  # type: ignore[return-value]


SyncGenFixtureParams = ParamSpec("SyncGenFixtureParams")
SyncGenFixtureYieldType = TypeVar("SyncGenFixtureYieldType")


def _wrap_syncgen_fixture(
    fixture_function: Callable[
        SyncGenFixtureParams, Generator[SyncGenFixtureYieldType]
    ],
    runner: _ManagedRunner,
) -> Callable[SyncGenFixtureParams, Generator[SyncGenFixtureYieldType]]:
    @functools.wraps(fixture_function)
    def _syncgen_fixture_wrapper(
        *args: SyncGenFixtureParams.args,
        **kwargs: SyncGenFixtureParams.kwargs,
    ) -> Generator[SyncGenFixtureYieldType]:
        with _temporary_event_loop(runner.loop):
            yield from fixture_function(*args, **kwargs)

    return _syncgen_fixture_wrapper


SyncFixtureParams = ParamSpec("SyncFixtureParams")
SyncFixtureReturnType = TypeVar("SyncFixtureReturnType")


def _wrap_sync_fixture(
    fixture_function: Callable[SyncFixtureParams, SyncFixtureReturnType],
    runner: _ManagedRunner,
) -> Callable[SyncFixtureParams, SyncFixtureReturnType]:
    @functools.wraps(fixture_function)
    def _sync_fixture_wrapper(
        *args: SyncFixtureParams.args,
        **kwargs: SyncFixtureParams.kwargs,
    ) -> SyncFixtureReturnType:
        with _temporary_event_loop(runner.loop):
            return fixture_function(*args, **kwargs)

    return _sync_fixture_wrapper


AsyncGenFixtureParams = ParamSpec("AsyncGenFixtureParams")
AsyncGenFixtureYieldType = TypeVar("AsyncGenFixtureYieldType")


def _wrap_asyncgen_fixture(
    fixture_function: Callable[
        AsyncGenFixtureParams, AsyncGeneratorType[AsyncGenFixtureYieldType, Any]
    ],
    runner: _ManagedRunner,
    request: pytest.FixtureRequest,
) -> Callable[AsyncGenFixtureParams, AsyncGenFixtureYieldType]:
    @functools.wraps(fixture_function)
    def _asyncgen_fixture_wrapper(
        *args: AsyncGenFixtureParams.args,
        **kwargs: AsyncGenFixtureParams.kwargs,
    ):
        gen_obj = fixture_function(*args, **kwargs)

        async def setup():
            res = await gen_obj.__anext__()
            return res

        context = contextvars.copy_context()
        result = runner.run(setup(), context=context)

        reset_contextvars = _apply_contextvar_changes(context)

        def finalizer() -> None:
            """Yield again, to finalize."""

            async def async_finalizer() -> None:
                try:
                    await gen_obj.__anext__()
                except StopAsyncIteration:
                    pass
                else:
                    msg = "Async generator fixture didn't stop."
                    msg += "Yield only once."
                    raise ValueError(msg)

            runner.run(async_finalizer(), context=context)
            if reset_contextvars is not None:
                reset_contextvars()

        request.addfinalizer(finalizer)
        return result

    return _asyncgen_fixture_wrapper


AsyncFixtureParams = ParamSpec("AsyncFixtureParams")
AsyncFixtureReturnType = TypeVar("AsyncFixtureReturnType")


def _wrap_async_fixture(
    fixture_function: Callable[
        AsyncFixtureParams, Coroutine[Any, Any, AsyncFixtureReturnType]
    ],
    runner: _ManagedRunner,
    request: pytest.FixtureRequest,
) -> Callable[AsyncFixtureParams, AsyncFixtureReturnType]:
    @functools.wraps(fixture_function)
    def _async_fixture_wrapper(
        *args: AsyncFixtureParams.args,
        **kwargs: AsyncFixtureParams.kwargs,
    ):
        async def setup():
            res = await fixture_function(*args, **kwargs)
            return res

        context = contextvars.copy_context()
        result = runner.run(setup(), context=context)

        # Copy the context vars modified by the setup task into the current
        # context, and (if needed) add a finalizer to reset them.
        #
        # Note that this is slightly different from the behavior of a non-async
        # fixture, which would rely on the fixture author to add a finalizer
        # to reset the variables. In this case, the author of the fixture can't
        # write such a finalizer because they have no way to capture the Context
        # in which the setup function was run, so we need to do it for them.
        reset_contextvars = _apply_contextvar_changes(context)
        if reset_contextvars is not None:
            request.addfinalizer(reset_contextvars)

        return result

    return _async_fixture_wrapper


def _apply_contextvar_changes(
    context: contextvars.Context,
) -> Callable[[], None] | None:
    """
    Copy contextvar changes from the given context to the current context.

    If any contextvars were modified by the fixture, return a finalizer that
    will restore them.
    """
    context_tokens = []
    for var in context:
        try:
            if var.get() is context.get(var):
                # This variable is not modified, so leave it as-is.
                continue
        except LookupError:
            # This variable isn't yet set in the current context at all.
            pass
        token = var.set(context.get(var))
        context_tokens.append((var, token))

    if not context_tokens:
        return None

    def restore_contextvars():
        while context_tokens:
            var, token = context_tokens.pop()
            var.reset(token)

    return restore_contextvars


class _AsyncTestKind(enum.Enum):
    """The kind of async test pytest-asyncio manages an item as."""

    COROUTINE = enum.auto()
    ASYNC_GENERATOR = enum.auto()
    HYPOTHESIS = enum.auto()
    HYPOTHESIS_UNSUPPORTED = enum.auto()


# Kinds that pytest-asyncio runs in an event loop.
_RUNNABLE_ASYNC_KINDS = frozenset({_AsyncTestKind.COROUTINE, _AsyncTestKind.HYPOTHESIS})

# Kinds that auto mode marks during collection. The unsupported-Hypothesis kind is
# deliberately excluded, so it is only acted upon when a marker is applied explicitly.
_COLLECTED_ASYNC_KINDS = frozenset(
    {
        _AsyncTestKind.COROUTINE,
        _AsyncTestKind.ASYNC_GENERATOR,
        _AsyncTestKind.HYPOTHESIS,
    }
)


def _callable_kind(obj: object) -> _AsyncTestKind | None:
    """
    Classify a test callable as an async test kind, independent of any marker.

    Returns None for callables pytest-asyncio does not manage, such as synchronous
    functions. This function is pure and must not mutate anything.

    The order of checks matters. A function decorated with ``@hypothesis.given`` is a
    synchronous driver, so the coroutine to run lives at ``obj.hypothesis.inner_test``
    and Hypothesis must be detected before the direct coroutine and async-generator
    checks. ``staticmethod`` is unwrapped so that static async tests are detected.
    """
    if getattr(obj, "is_hypothesis_test", False):
        inner = getattr(getattr(obj, "hypothesis", None), "inner_test", None)
        if inner is None:
            # Hypothesis is too old to expose ``hypothesis.inner_test``, so the
            # coroutine cannot be wrapped. Recognized, but unsupported.
            return _AsyncTestKind.HYPOTHESIS_UNSUPPORTED
        if inspect.iscoroutinefunction(inner):
            return _AsyncTestKind.HYPOTHESIS
        # A synchronous Hypothesis test that happens to carry the marker.
        return None
    func = obj.__func__ if isinstance(obj, staticmethod) else obj
    if inspect.isasyncgenfunction(func):
        return _AsyncTestKind.ASYNC_GENERATOR
    if inspect.iscoroutinefunction(func):
        return _AsyncTestKind.COROUTINE
    return None


def _managed_kind(item: Item) -> _AsyncTestKind | None:
    """
    Return the async test kind pytest-asyncio manages the item as, or None.

    An item is managed when it carries the ``asyncio`` marker (whether applied as a
    decorator, by auto mode, in a collection hook, or via a parameter set) and its
    callable is an async test kind.
    """
    if not isinstance(item, Function):
        return None
    if item.get_closest_marker("asyncio") is None:
        return None
    return _callable_kind(item.obj)


def _synchronization_target(item: Function, kind: _AsyncTestKind) -> tuple[object, str]:
    """Return the (holder, attribute) of the coroutine to synchronize."""
    if kind is _AsyncTestKind.HYPOTHESIS:
        return item.obj.hypothesis, "inner_test"
    return item, "obj"


def _item_loop_scope(item: Function) -> _ScopeName:
    """
    Return the scope of the asyncio event loop the item is run in.

    It is identical to the ``loop_scope`` value of the closest ``asyncio`` marker. If
    no such value is present, the loop scope is the ``asyncio_default_test_loop_scope``
    configuration value.
    """
    marker = item.get_closest_marker("asyncio")
    assert marker is not None
    loop_scope = marker.kwargs.get("loop_scope")
    if loop_scope is None:
        return _get_default_test_loop_scope(item.config)
    return loop_scope


def _loop_factories_configured(item: Item) -> bool:
    """Return whether any pytest_asyncio_loop_factories hook is implemented."""
    return bool(item.ihook.pytest_asyncio_loop_factories.get_hookimpls())


def _has_loop_factory_param(item: Function) -> bool:
    """Return whether the item was parametrized with a loop factory."""
    callspec = getattr(item, "callspec", None)
    return callspec is not None and _LOOP_FACTORY_PARAM in callspec.params


def _item_loop_factory(item: Function) -> _LoopFactoryVariant:
    callspec = getattr(item, "callspec", None)
    variant = callspec.params.get(_LOOP_FACTORY_PARAM) if callspec is not None else None
    if variant is None:
        variant = _loop_factory_variant(item.config, None, None)
    if not isinstance(variant, _LoopFactoryVariant):
        raise TypeError("pytest-asyncio received an invalid loop-factory token")
    return variant


def _resolve_asyncio_marker(item: Function) -> Mark | None:
    marker = item.get_closest_marker("asyncio")
    if marker is not None:
        return marker
    if _get_asyncio_mode(item.config) == Mode.AUTO:
        item.add_marker("asyncio")
        return item.get_closest_marker("asyncio")
    return None


def _adapt_auto_mode_fixtures(item: Function) -> None:
    """
    Convert plain async fixtures once, before pytest starts fixture setup.

    Pytest does not expose a public API for replacing a fixture definition after
    ``@pytest.fixture`` has created it. Keep this compatibility seam confined to
    auto mode; fixtures declared with ``pytest_asyncio.fixture`` are wrapped before
    pytest creates their FixtureDef.
    """
    fixtureinfo = get_fixture_info(item)
    if fixtureinfo is None:
        return
    for fixturedefs in fixtureinfo.name2fixturedefs.values():
        for fixturedef in fixturedefs:
            fixture_function = fixturedef.func
            if _is_asyncio_fixture_function(fixture_function):
                continue
            if not _is_coroutine_or_asyncgen(fixture_function):
                continue
            underlying_function = getattr(
                fixture_function,
                "__func__",
                fixture_function,
            )
            loop_scope = getattr(underlying_function, "_loop_scope", None)
            wrapper = _create_asyncio_fixture_wrapper(
                underlying_function,
                loop_scope,
            )
            bound_instance = getattr(fixture_function, "__self__", None)
            adapted_function = (
                wrapper.__get__(bound_instance)
                if bound_instance is not None
                else wrapper
            )
            argnames = list(fixturedef.argnames)
            if "request" not in argnames:
                argnames.append("request")
            if _LOOP_FACTORY_PARAM in argnames:
                raise pytest.UsageError(
                    f"{_LOOP_FACTORY_PARAM!r} is reserved for use by pytest-asyncio."
                )
            argnames.append(_LOOP_FACTORY_PARAM)
            replace_fixture_function(
                fixturedef,
                adapted_function,
                tuple(argnames),
            )


# The function name needs to start with "pytest_"
# see https://github.com/pytest-dev/pytest/issues/11307
@pytest.hookimpl(specname="pytest_pycollect_makeitem", wrapper=True)
def pytest_pycollect_makeitem_apply_automode_marker(
    collector: pytest.Module | pytest.Class, name: str, obj: object
) -> Generator[None, _CollectResult, _CollectResult]:
    """
    In auto mode, apply the ``asyncio`` marker to collected async test items.

    The marker is what makes pytest-asyncio manage an item. Applying it during
    collection (rather than only classifying internally) keeps it observable to
    downstream collection hooks and lets pytest_generate_tests parametrize loop
    factories. Items are marked in place and never replaced.
    """
    node_or_list_of_nodes = yield
    if node_or_list_of_nodes:
        if isinstance(node_or_list_of_nodes, Sequence):
            nodes: Iterable[object] = node_or_list_of_nodes
        else:
            nodes = (node_or_list_of_nodes,)
        for node in nodes:
            if (
                isinstance(node, Function)
                and _callable_kind(node.obj) in _COLLECTED_ASYNC_KINDS
            ):
                # Adds the marker in auto mode; a no-op in strict mode, where the
                # user supplies the marker.
                _resolve_asyncio_marker(node)
            if (
                isinstance(node, Function)
                and _get_asyncio_mode(node.config) == Mode.AUTO
            ):
                _adapt_auto_mode_fixtures(node)
    return node_or_list_of_nodes


def _is_asyncio_managed_fixture(fixturedef: FixtureDef, mode: Mode) -> bool:
    if _is_asyncio_fixture_function(fixturedef.func):
        return True
    return mode == Mode.AUTO and _is_coroutine_or_asyncgen(fixturedef.func)


def _resolve_fixture_loop_scope(fixturedef: FixtureDef, config: Config) -> _ScopeName:
    return getattr(fixturedef.func, "_loop_scope", None) or config.getini(
        "asyncio_default_fixture_loop_scope"
    )


def _resolved_fixturedef(
    fixtureinfo: FixtureInfo,
    fixture_name: str,
    override_depths: Mapping[str, int],
) -> FixtureDef | None:
    """Resolve a fixture name using pytest's override-chain semantics."""
    fixturedefs = fixtureinfo.name2fixturedefs.get(fixture_name)
    if not fixturedefs:
        return None
    index = len(fixturedefs) - override_depths.get(fixture_name, 0) - 1
    return fixturedefs[index] if index >= 0 else None


def _warn_scope_edge(
    config: Config,
    owner: object,
    owner_name: str,
    owner_scope: _ScopeName,
    fixturedef: FixtureDef,
) -> None:
    fixture_scope = _resolve_fixture_loop_scope(fixturedef, config)
    if owner_scope == fixture_scope:
        return
    edge = (owner, fixturedef)
    warned_edges = config.stash[_WARNED_SCOPE_EDGES_KEY]
    if edge in warned_edges:
        return
    warned_edges.add(edge)
    warnings.warn(
        f"{owner_name} uses an asyncio event loop with {owner_scope!r} scope, "
        f"but it requests fixture {fixturedef.argname!r}, which uses an asyncio "
        f"event loop with {fixture_scope!r} scope. Align the loop scopes to avoid "
        "jumping between event loops.",
        PytestAsyncioWarning,
        stacklevel=2,
    )


def _warn_about_static_loop_scope_jumps(item: Item) -> None:
    """Inspect the fixture graph, including dependencies hidden by sync fixtures."""
    fixtureinfo = get_fixture_info(item)
    if fixtureinfo is None:
        return
    mode = _get_asyncio_mode(item.config)
    initial_owner: tuple[object, str, _ScopeName] | None = None
    if _managed_kind(item) is not None:
        assert isinstance(item, Function)
        initial_owner = (item, f"test {item.name!r}", _item_loop_scope(item))

    seen: set[tuple[FixtureDef, object | None, tuple[tuple[str, int], ...]]] = set()

    def visit(
        fixture_name: str,
        owner_info: tuple[object, str, _ScopeName] | None,
        override_depths: Mapping[str, int],
    ) -> None:
        fixturedef = _resolved_fixturedef(
            fixtureinfo,
            fixture_name,
            override_depths,
        )
        if fixturedef is None:
            return
        state = (
            fixturedef,
            None if owner_info is None else owner_info[0],
            tuple(sorted(override_depths.items())),
        )
        if state in seen:
            return
        seen.add(state)

        next_owner = owner_info
        if _is_asyncio_managed_fixture(fixturedef, mode):
            if owner_info is not None:
                _warn_scope_edge(item.config, *owner_info, fixturedef)
            next_owner = (
                fixturedef,
                f"fixture {fixturedef.argname!r}",
                _resolve_fixture_loop_scope(fixturedef, item.config),
            )
        child_override_depths = dict(override_depths)
        child_override_depths[fixturedef.argname] = (
            child_override_depths.get(fixturedef.argname, 0) + 1
        )
        for dependency_name in fixturedef.argnames:
            visit(dependency_name, next_owner, child_override_depths)

    for fixture_name in fixtureinfo.initialnames:
        visit(fixture_name, initial_owner, {})


def _uses_asyncio_fixture(metafunc: pytest.Metafunc) -> bool:
    mode = _get_asyncio_mode(metafunc.config)
    fixtureinfo = get_fixture_info(metafunc.definition)
    assert fixtureinfo is not None
    seen: set[tuple[FixtureDef, tuple[tuple[str, int], ...]]] = set()

    def visit(fixture_name: str, override_depths: Mapping[str, int]) -> bool:
        fixturedef = _resolved_fixturedef(fixtureinfo, fixture_name, override_depths)
        if fixturedef is None:
            return False
        state = (fixturedef, tuple(sorted(override_depths.items())))
        if state in seen:
            return False
        seen.add(state)
        if _is_asyncio_managed_fixture(fixturedef, mode):
            return True
        child_override_depths = dict(override_depths)
        child_override_depths[fixturedef.argname] = (
            child_override_depths.get(fixturedef.argname, 0) + 1
        )
        return any(
            visit(dependency_name, child_override_depths)
            for dependency_name in fixturedef.argnames
        )

    return any(visit(fixture_name, {}) for fixture_name in fixtureinfo.initialnames)


def _same_loop_factory(
    first: LoopFactory | None,
    second: LoopFactory | None,
) -> bool:
    if first is second:
        return True
    if inspect.ismethod(first) and inspect.ismethod(second):
        # Attribute access creates a fresh bound-method object each time. Compare
        # the stable parts without invoking arbitrary user-defined equality.
        return first.__func__ is second.__func__ and first.__self__ is second.__self__
    return False


def _loop_factory_variant(
    config: Config,
    name: str | None,
    factory: LoopFactory | None,
) -> _LoopFactoryVariant:
    variants = config.stash[_LOOP_FACTORY_VARIANTS_KEY]
    for variant in variants:
        if variant.name == name and _same_loop_factory(variant.factory, factory):
            return variant
    variant = _LoopFactoryVariant(name, factory)
    variants.append(variant)
    return variant


@pytest.hookimpl(tryfirst=True)
def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    marker_selected_factory_names: Sequence[str] | None
    if _callable_kind(metafunc.definition.obj) in _COLLECTED_ASYNC_KINDS:
        asyncio_marker = _resolve_asyncio_marker(metafunc.definition)
        if asyncio_marker is None:
            return
        _, marker_selected_factory_names = _parse_asyncio_marker(asyncio_marker)
    else:
        if not _uses_asyncio_fixture(metafunc):
            return
        marker_selected_factory_names = None

    hook_result = _collect_hook_loop_factories(metafunc.config, metafunc.definition)
    if hook_result is None:
        if marker_selected_factory_names is not None:
            raise pytest.UsageError(
                "mark.asyncio 'loop_factories' requires at least one "
                "pytest_asyncio_loop_factories hook implementation."
            )
        factory_params: list[object] = [
            _loop_factory_variant(metafunc.config, None, None)
        ]
        factory_ids: list[object] = [pytest.HIDDEN_PARAM]
    else:
        hook_factories = hook_result
        if marker_selected_factory_names is None:
            factory_params = [
                _loop_factory_variant(metafunc.config, name, factory)
                for name, factory in hook_factories.items()
            ]
            factory_ids = list(hook_factories)
        else:
            # Iterate in marker order to preserve explicit user selection order.
            factory_ids = list(marker_selected_factory_names)
            factory_params = [
                (
                    _loop_factory_variant(
                        metafunc.config,
                        name,
                        hook_factories[name],
                    )
                    if name in hook_factories
                    else pytest.param(
                        _LoopFactoryVariant(name, None),
                        marks=pytest.mark.skip(
                            reason=(
                                f"Loop factory {name!r} is not available."
                                f" Available factories:"
                                f" {', '.join(hook_factories)}."
                            ),
                        ),
                    )
                )
                for name in marker_selected_factory_names
            ]

    if _LOOP_FACTORY_PARAM not in metafunc.fixturenames:
        metafunc.fixturenames.insert(0, _LOOP_FACTORY_PARAM)
    if len(factory_ids) == 1:
        factory_ids = [pytest.HIDDEN_PARAM]
    # Async fixture wrappers formally depend on this parameter, which lets pytest
    # invalidate their caches when the factory changes. Keep its scope wide enough
    # for every possible fixture scope. Pytest may reuse a direct-parameter
    # FixtureDef for this name across functions, so choosing the scope per item can
    # otherwise cause a ScopeMismatch when differently scoped fixtures coexist.
    metafunc.parametrize(
        _LOOP_FACTORY_PARAM,
        factory_params,
        ids=factory_ids,
        indirect=False,
        scope="session",
    )


@contextlib.contextmanager
def _temporary_event_loop(loop: AbstractEventLoop) -> Iterator[None]:
    try:
        old_loop = _get_event_loop_no_warn()
    except RuntimeError:
        old_loop = None
    if old_loop is loop:
        yield
        return
    _set_event_loop(loop)
    try:
        yield
    finally:
        _set_event_loop(old_loop)


def _get_event_loop_no_warn() -> asyncio.AbstractEventLoop:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return asyncio.get_event_loop()


def _set_event_loop(loop: AbstractEventLoop | None) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        asyncio.set_event_loop(loop)


_MARKED_AFTER_PARAMETRIZATION_LOOP_FACTORIES_ERROR = """\
The asyncio marker of {item} was not visible when loop-factory parametrization ran \
(it was applied after collection, e.g. in a pytest_collection_modifyitems hook, or \
only to a parametrize parameter set). Loop factories parametrize tests during \
collection, which requires the marker to be visible when pytest_generate_tests runs. \
Apply the marker during collection (e.g. in a pytest_pycollect_makeitem hook), run \
pytest-asyncio in auto mode, or move the marker to the test function.\
"""

_ASYNC_GENERATOR_UNSUPPORTED = (
    "Tests based on asynchronous generators are not supported. {name} will be ignored."
)

_HYPOTHESIS_UNSUPPORTED = (
    "test function {item!r} is using Hypothesis, but pytest-asyncio only works with "
    "Hypothesis 3.64.0 or later."
)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: Item) -> None:
    """Validate a pytest-asyncio test before its fixtures are set up."""
    _warn_about_static_loop_scope_jumps(item)
    kind = _managed_kind(item)
    if kind is None:
        return
    assert isinstance(item, Function)
    if kind is _AsyncTestKind.HYPOTHESIS_UNSUPPORTED:
        pytest.fail(_HYPOTHESIS_UNSUPPORTED.format(item=item.name), pytrace=False)
    if kind is _AsyncTestKind.ASYNC_GENERATOR:
        message = _ASYNC_GENERATOR_UNSUPPORTED.format(name=item.name)
        item.warn(PytestCollectionWarning(message))
        pytest.xfail(message)
    # A coroutine or Hypothesis coroutine test: run it in a scoped event loop.
    marker = item.get_closest_marker("asyncio")
    assert marker is not None
    _, selected_factory_names = _parse_asyncio_marker(marker)
    # Loop factories are delivered through callspec parametrization in
    # pytest_generate_tests, which only sees markers present on the test definition.
    # A marker that requests factories but produced no factory parameter was not
    # visible then (e.g. applied after collection or only to a parametrize parameter
    # set), so it cannot be honored.
    if (
        selected_factory_names is not None or _loop_factories_configured(item)
    ) and not _has_loop_factory_param(item):
        pytest.fail(
            _MARKED_AFTER_PARAMETRIZATION_LOOP_FACTORIES_ERROR.format(item=item.name),
            pytrace=False,
        )


@pytest.hookimpl(tryfirst=True, wrapper=True)
def pytest_pyfunc_call(pyfuncitem: Function) -> Generator[None, object, object]:
    """Synchronize the coroutine of a pytest-asyncio test before it is called."""
    kind = _managed_kind(pyfuncitem)
    if kind is None:
        if pyfuncitem.get_closest_marker("asyncio") is not None:
            pyfuncitem.warn(
                pytest.PytestWarning(
                    f"The test {pyfuncitem} is marked with '@pytest.mark.asyncio' "
                    "but it is not an async function. "
                    "Please remove the asyncio mark. "
                    "If the test is not marked explicitly, "
                    "check for global marks applied via 'pytestmark'."
                )
            )
        return (yield)
    if kind not in _RUNNABLE_ASYNC_KINDS:
        # Async generators (and the unsupported-Hypothesis kind) are handled in
        # pytest_runtest_setup and never reach the call phase.
        return (yield)
    runner = _get_loop_manager(pyfuncitem.config).get_runner(
        pyfuncitem,
        _item_loop_scope(pyfuncitem),
        _item_loop_factory(pyfuncitem),
    )
    context = contextvars.copy_context()
    target = _synchronization_target(pyfuncitem, kind)
    synchronized_obj = _synchronize_coroutine(getattr(*target), runner, context)
    with MonkeyPatch.context() as c:
        c.setattr(*target, synchronized_obj)
        return (yield)


def _synchronize_coroutine(
    func: Callable[..., Coroutine[Any, Any, Any]],
    runner: _ManagedRunner,
    context: contextvars.Context,
):
    """
    Return a sync wrapper around a coroutine executing it in the
    specified runner and context.
    """

    @functools.wraps(func)
    def inner(*args, **kwargs):
        coro = func(*args, **kwargs)
        runner.run(coro, context=context)

    return inner


@pytest.hookimpl(tryfirst=True)
def pytest_fixture_setup(fixturedef: FixtureDef, request) -> None:
    """Reject async fixtures that were not converted during collection."""
    asyncio_mode = _get_asyncio_mode(request.config)
    is_async = _is_coroutine_or_asyncgen(fixturedef.func)
    if not is_async:
        return None
    if asyncio_mode == Mode.STRICT:
        pytest.fail(
            f"Async fixture {fixturedef.argname!r} is decorated with @pytest.fixture "
            "in strict mode. Use @pytest_asyncio.fixture or switch to auto mode.",
            pytrace=False,
        )
    pytest.fail(
        f"Async fixture {fixturedef.argname!r} was requested dynamically and could "
        "not be prepared during collection. Declare it with "
        "@pytest_asyncio.fixture.",
        pytrace=False,
    )


_MARKER_SCOPE_KWARG_ERROR = """\
The "scope" keyword argument to the asyncio marker is not supported. \
Use the "loop_scope" argument instead.
"""

_INVALID_LOOP_FACTORIES_KWARG = """\
mark.asyncio 'loop_factories' must be a non-empty sequence of strings.
"""


def _parse_asyncio_marker(
    asyncio_marker: Mark,
) -> tuple[_ScopeName | None, Sequence[str] | None]:
    assert asyncio_marker.name == "asyncio"
    if "scope" in asyncio_marker.kwargs:
        raise pytest.UsageError(_MARKER_SCOPE_KWARG_ERROR)
    _validate_asyncio_marker(asyncio_marker)
    scope = asyncio_marker.kwargs.get("loop_scope")
    if scope is not None:
        assert scope in {"function", "class", "module", "package", "session"}
    marker_value = asyncio_marker.kwargs.get("loop_factories")
    if marker_value is None:
        return scope, None
    if isinstance(marker_value, str) or not isinstance(marker_value, Sequence):
        raise ValueError(_INVALID_LOOP_FACTORIES_KWARG)
    if not marker_value or any(
        not isinstance(factory_name, str) or not factory_name
        for factory_name in marker_value
    ):
        raise ValueError(_INVALID_LOOP_FACTORIES_KWARG)
    return scope, marker_value


def _validate_asyncio_marker(asyncio_marker: Mark) -> None:
    if asyncio_marker.args or (
        asyncio_marker.kwargs
        and set(asyncio_marker.kwargs) - {"loop_scope", "loop_factories"}
    ):
        msg = (
            "mark.asyncio accepts only keyword arguments 'loop_scope' and"
            " 'loop_factories'."
        )
        raise ValueError(msg)


def _get_default_test_loop_scope(config: Config) -> Any:
    return config.getini("asyncio_default_test_loop_scope")


def is_async_test(item: Item) -> bool:
    """Returns whether a test item is managed by pytest-asyncio"""
    return _managed_kind(item) is not None


def _unused_port(socket_type: int) -> int:
    """Find an unused localhost port from 1024-65535 and return it."""
    with contextlib.closing(socket.socket(type=socket_type)) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def unused_tcp_port() -> int:
    return _unused_port(socket.SOCK_STREAM)


@pytest.fixture
def unused_udp_port() -> int:
    return _unused_port(socket.SOCK_DGRAM)


@pytest.fixture(scope="session")
def unused_tcp_port_factory() -> Callable[[], int]:
    """A factory function, producing different unused TCP ports."""
    produced = set()

    def factory():
        """Return an unused port."""
        port = _unused_port(socket.SOCK_STREAM)

        while port in produced:
            port = _unused_port(socket.SOCK_STREAM)

        produced.add(port)

        return port

    return factory


@pytest.fixture(scope="session")
def unused_udp_port_factory() -> Callable[[], int]:
    """A factory function, producing different unused UDP ports."""
    produced = set()

    def factory():
        """Return an unused port."""
        port = _unused_port(socket.SOCK_DGRAM)

        while port in produced:
            port = _unused_port(socket.SOCK_DGRAM)

        produced.add(port)

        return port

    return factory
