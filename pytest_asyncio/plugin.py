"""pytest-asyncio implementation."""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import enum
import functools
import inspect
import socket
import sys
import traceback
import warnings
from asyncio import AbstractEventLoop
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Collection,
    Generator,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from types import AsyncGeneratorType, CoroutineType
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    ParamSpec,
    TypeAlias,
    TypeVar,
    cast,
    overload,
)

import pluggy

import pytest
from _pytest.fixtures import resolve_fixture_function
from _pytest.scope import Scope
from pytest import (
    Config,
    FixtureDef,
    FixtureRequest,
    Function,
    Item,
    Mark,
    MonkeyPatch,
    Parser,
    PytestCollectionWarning,
    PytestDeprecationWarning,
    PytestPluginManager,
)

if sys.version_info >= (3, 11):
    from asyncio import Runner
else:
    from backports.asyncio.runner import Runner

if TYPE_CHECKING:
    # AbstractEventLoopPolicy is deprecated and scheduled for removal in Python 3.16
    # Import it for type checking only to avoid raising a DeprecationWarning.
    from asyncio import AbstractEventLoopPolicy

_ScopeName = Literal["session", "package", "module", "class", "function"]
_R = TypeVar("_R", bound=Awaitable[Any] | AsyncIterator[Any])
_P = ParamSpec("_P")
FixtureFunction = Callable[_P, _R]
LoopFactory: TypeAlias = Callable[[], AbstractEventLoop]
_CollectResult: TypeAlias = (
    "pytest.Item | pytest.Collector | list[pytest.Item | pytest.Collector] | None"
)


class PytestAsyncioError(Exception):
    """Base class for exceptions raised by pytest-asyncio"""


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
        default=None,
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
        _make_asyncio_fixture_function(fixture_function, loop_scope)
        return pytest.fixture(fixture_function, **kwargs)

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
    if not hook_caller.get_hookimpls():
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


_DEFAULT_FIXTURE_LOOP_SCOPE_UNSET = """\
The configuration option "asyncio_default_fixture_loop_scope" is unset.
The event loop scope for asynchronous fixtures will default to the "fixture" caching \
scope. Future versions of pytest-asyncio will default the loop scope for asynchronous \
fixtures to "function" scope. Set the default fixture loop scope explicitly in order \
to avoid unexpected behavior in the future. Valid fixture loop scopes are: \
"function", "class", "module", "package", "session"
"""


def _validate_scope(scope: str | None, option_name: str) -> None:
    if scope is None:
        return
    valid_scopes = [s.value for s in Scope]
    if scope not in valid_scopes:
        raise pytest.UsageError(
            f"{scope!r} is not a valid {option_name}. "
            f"Valid scopes are: {', '.join(valid_scopes)}."
        )


def pytest_configure(config: Config) -> None:
    default_fixture_loop_scope = config.getini("asyncio_default_fixture_loop_scope")
    _validate_scope(default_fixture_loop_scope, "asyncio_default_fixture_loop_scope")
    if not default_fixture_loop_scope:
        warnings.warn(PytestDeprecationWarning(_DEFAULT_FIXTURE_LOOP_SCOPE_UNSET))

    default_test_loop_scope = config.getini("asyncio_default_test_loop_scope")
    _validate_scope(default_test_loop_scope, "asyncio_default_test_loop_scope")
    config.addinivalue_line(
        "markers",
        "asyncio: "
        "mark the test as a coroutine, it will be "
        "run using an asyncio event loop",
    )


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


def _fixture_synchronizer(
    fixturedef: FixtureDef, runner: Runner, request: FixtureRequest
) -> Callable:
    """Returns a synchronous function evaluating the specified fixture."""
    fixture_function = resolve_fixture_function(fixturedef, request)
    if inspect.isasyncgenfunction(fixturedef.func):
        return _wrap_asyncgen_fixture(fixture_function, runner, request)  # type: ignore[arg-type]
    elif inspect.iscoroutinefunction(fixturedef.func):
        return _wrap_async_fixture(fixture_function, runner, request)  # type: ignore[arg-type]
    elif inspect.isgeneratorfunction(fixturedef.func):
        return _wrap_syncgen_fixture(fixture_function, runner)  # type: ignore[arg-type]
    else:
        return _wrap_sync_fixture(fixture_function, runner)  # type: ignore[arg-type]


SyncGenFixtureParams = ParamSpec("SyncGenFixtureParams")
SyncGenFixtureYieldType = TypeVar("SyncGenFixtureYieldType")


def _wrap_syncgen_fixture(
    fixture_function: Callable[
        SyncGenFixtureParams, Generator[SyncGenFixtureYieldType]
    ],
    runner: Runner,
) -> Callable[SyncGenFixtureParams, Generator[SyncGenFixtureYieldType]]:
    @functools.wraps(fixture_function)
    def _syncgen_fixture_wrapper(
        *args: SyncGenFixtureParams.args,
        **kwargs: SyncGenFixtureParams.kwargs,
    ) -> Generator[SyncGenFixtureYieldType]:
        with _temporary_event_loop(runner.get_loop()):
            yield from fixture_function(*args, **kwargs)

    return _syncgen_fixture_wrapper


SyncFixtureParams = ParamSpec("SyncFixtureParams")
SyncFixtureReturnType = TypeVar("SyncFixtureReturnType")


def _wrap_sync_fixture(
    fixture_function: Callable[SyncFixtureParams, SyncFixtureReturnType],
    runner: Runner,
) -> Callable[SyncFixtureParams, SyncFixtureReturnType]:
    @functools.wraps(fixture_function)
    def _sync_fixture_wrapper(
        *args: SyncFixtureParams.args,
        **kwargs: SyncFixtureParams.kwargs,
    ) -> SyncFixtureReturnType:
        with _temporary_event_loop(runner.get_loop()):
            return fixture_function(*args, **kwargs)

    return _sync_fixture_wrapper


AsyncGenFixtureParams = ParamSpec("AsyncGenFixtureParams")
AsyncGenFixtureYieldType = TypeVar("AsyncGenFixtureYieldType")


def _wrap_asyncgen_fixture(
    fixture_function: Callable[
        AsyncGenFixtureParams, AsyncGeneratorType[AsyncGenFixtureYieldType, Any]
    ],
    runner: Runner,
    request: FixtureRequest,
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
        AsyncFixtureParams, CoroutineType[Any, Any, AsyncFixtureReturnType]
    ],
    runner: Runner,
    request: FixtureRequest,
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
    loop_scope = marker.kwargs.get("loop_scope") or marker.kwargs.get("scope")
    if loop_scope is None:
        return _get_default_test_loop_scope(item.config)
    return loop_scope


def _loop_factories_configured(item: Item) -> bool:
    """Return whether any pytest_asyncio_loop_factories hook is implemented."""
    return bool(item.ihook.pytest_asyncio_loop_factories.get_hookimpls())


def _has_loop_factory_param(item: Function) -> bool:
    """Return whether the item was parametrized with a loop factory."""
    callspec = getattr(item, "callspec", None)
    return callspec is not None and _asyncio_loop_factory.__name__ in callspec.params


def _resolve_asyncio_marker(item: Function) -> Mark | None:
    marker = item.get_closest_marker("asyncio")
    if marker is not None:
        return marker
    if _get_asyncio_mode(item.config) == Mode.AUTO:
        item.add_marker("asyncio")
        return item.get_closest_marker("asyncio")
    return None


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
    return node_or_list_of_nodes


@pytest.hookimpl(tryfirst=True)
def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if _callable_kind(metafunc.definition.obj) not in _COLLECTED_ASYNC_KINDS:
        return

    asyncio_marker = _resolve_asyncio_marker(metafunc.definition)
    if asyncio_marker is None:
        return
    marker_loop_scope, marker_selected_factory_names = _parse_asyncio_marker(
        asyncio_marker
    )

    hook_factories = _collect_hook_loop_factories(metafunc.config, metafunc.definition)
    if hook_factories is None:
        if marker_selected_factory_names is not None:
            raise pytest.UsageError(
                "mark.asyncio 'loop_factories' requires at least one "
                "pytest_asyncio_loop_factories hook implementation."
            )
        return

    factory_params: Collection[object]
    factory_ids: Collection[str]
    if marker_selected_factory_names is None:
        factory_params = hook_factories.values()
        factory_ids = hook_factories.keys()
    else:
        # Iterate in marker order to preserve explicit user selection
        # order.
        factory_ids = marker_selected_factory_names
        factory_params = [
            (
                hook_factories[name]
                if name in hook_factories
                else pytest.param(
                    None,
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
    metafunc.fixturenames.append(_asyncio_loop_factory.__name__)
    default_loop_scope = _get_default_test_loop_scope(metafunc.config)
    loop_scope = marker_loop_scope or default_loop_scope
    # pytest.HIDDEN_PARAM was added in pytest 8.4
    hide_id = len(factory_ids) == 1 and hasattr(pytest, "HIDDEN_PARAM")
    metafunc.parametrize(
        _asyncio_loop_factory.__name__,
        factory_params,
        ids=(pytest.HIDDEN_PARAM,) if hide_id else factory_ids,
        indirect=True,
        scope=loop_scope,
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


@contextlib.contextmanager
def _temporary_event_loop_policy(
    policy: AbstractEventLoopPolicy,
) -> Iterator[None]:
    old_loop_policy = _get_event_loop_policy()
    _set_event_loop_policy(policy)
    try:
        yield
    finally:
        _set_event_loop_policy(old_loop_policy)


def _get_event_loop_policy() -> AbstractEventLoopPolicy:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return asyncio.get_event_loop_policy()


def _set_event_loop_policy(policy: AbstractEventLoopPolicy) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        asyncio.set_event_loop_policy(policy)


def _get_event_loop_no_warn(
    policy: AbstractEventLoopPolicy | None = None,
) -> asyncio.AbstractEventLoop:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        if policy is not None:
            return policy.get_event_loop()
        else:
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

# Tracks the test functions for which the deprecated "scope" marker argument has
# already been reported, so the warning is emitted once per test function rather than
# once per parametrized item (see https://github.com/pytest-dev/pytest-asyncio/pull/798).
_DEPRECATED_SCOPE_WARNED = pytest.StashKey[set[int]]()


def _warn_deprecated_scope_once(item: Function, marker: Mark) -> None:
    """Emit the deprecated ``scope`` marker-argument warning once per test function."""
    if "scope" not in marker.kwargs:
        return
    warned = item.config.stash.setdefault(_DEPRECATED_SCOPE_WARNED, set())
    if id(item.function) in warned:
        return
    warned.add(id(item.function))
    warnings.warn(PytestDeprecationWarning(_MARKER_SCOPE_KWARG_DEPRECATION_WARNING))


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: Item) -> None:
    """
    Prepare a pytest-asyncio test before its fixtures are set up.

    Runs ``tryfirst`` so that the scoped event-loop runner fixture is appended to
    ``item.fixturenames`` before pytest core fills fixtures from it in
    ``Function.setup``.
    """
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
    _warn_deprecated_scope_once(item, marker)
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
    runner_fixture_id = f"_{_item_loop_scope(item)}_scoped_runner"
    if _loop_factories_configured(item):
        # Set up the runner (and, transitively, the loop factory it depends on) before
        # the test's own fixtures, so a factory variant change cascades cache
        # invalidation before any async fixture reuses its cache. pytest core fills
        # fixtures in fixturenames order once the request is active, so order the
        # runner first.
        if runner_fixture_id in item.fixturenames:
            item.fixturenames.remove(runner_fixture_id)
        item.fixturenames.insert(0, runner_fixture_id)
    elif runner_fixture_id not in item.fixturenames:
        item.fixturenames.append(runner_fixture_id)


def _warn_about_strict_mode_async_fixtures(pyfuncitem: Function) -> None:
    """Warn when a strict-mode asyncio test requests a plain async @pytest.fixture."""
    if _get_asyncio_mode(pyfuncitem.config) != Mode.STRICT:
        return
    for fixname, fixtures in pyfuncitem._fixtureinfo.name2fixturedefs.items():
        # name2fixturedefs is a dict between fixture name and a list of matching
        # fixturedefs. The last entry in the list is closest and the one used.
        func = fixtures[-1].func
        if _is_coroutine_or_asyncgen(func) and not _is_asyncio_fixture_function(func):
            warnings.warn(
                PytestDeprecationWarning(
                    f"asyncio test {pyfuncitem.name!r} requested async "
                    "@pytest.fixture "
                    f"{fixname!r} in strict mode. "
                    "You might want to use @pytest_asyncio.fixture or switch "
                    "to auto mode. "
                    "This will become an error in future versions of "
                    "pytest-asyncio."
                ),
                stacklevel=1,
            )
            # no stacklevel points at the users code, so we set stacklevel=1
            # so it at least indicates that it's the plugin complaining.
            # Pytest gives the test file & name in the warnings summary at least


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
    _warn_about_strict_mode_async_fixtures(pyfuncitem)
    # pytest_runtest_setup added the scoped runner to the item's fixtures, so it has
    # already been filled into funcargs by the time the test is called.
    runner = cast(
        Runner, pyfuncitem.funcargs[f"_{_item_loop_scope(pyfuncitem)}_scoped_runner"]
    )
    context = contextvars.copy_context()
    target = _synchronization_target(pyfuncitem, kind)
    synchronized_obj = _synchronize_coroutine(getattr(*target), runner, context)
    with MonkeyPatch.context() as c:
        c.setattr(*target, synchronized_obj)
        return (yield)


def _synchronize_coroutine(
    func: Callable[..., CoroutineType],
    runner: asyncio.Runner,
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


@pytest.hookimpl(wrapper=True)
def pytest_fixture_setup(fixturedef: FixtureDef, request) -> object | None:
    if (
        fixturedef.argname == "event_loop_policy"
        and fixturedef.func.__module__ != __name__
    ):
        warnings.warn(
            PytestDeprecationWarning(_EVENT_LOOP_POLICY_FIXTURE_DEPRECATION_WARNING),
        )
    asyncio_mode = _get_asyncio_mode(request.config)
    if not _is_asyncio_fixture_function(fixturedef.func):
        if asyncio_mode == Mode.STRICT:
            # Ignore async fixtures without explicit asyncio mark in strict mode
            # This applies to pytest_trio fixtures, for example
            return (yield)
        if not _is_coroutine_or_asyncgen(fixturedef.func):
            return (yield)
    default_loop_scope = request.config.getini("asyncio_default_fixture_loop_scope")
    loop_scope = (
        getattr(fixturedef.func, "_loop_scope", None)
        or default_loop_scope
        or fixturedef.scope
    )
    runner_fixture_id = f"_{loop_scope}_scoped_runner"
    runner = request.getfixturevalue(runner_fixture_id)
    # Prevent the runner closing before the fixture's async teardown.
    runner_fixturedef = request._get_active_fixturedef(runner_fixture_id)
    runner_fixturedef.addfinalizer(
        functools.partial(fixturedef.finish, request=request)
    )
    synchronizer = _fixture_synchronizer(fixturedef, runner, request)
    _make_asyncio_fixture_function(synchronizer, loop_scope)
    with MonkeyPatch.context() as c:
        c.setattr(fixturedef, "func", synchronizer)
        hook_result = yield
    return hook_result


_DUPLICATE_LOOP_SCOPE_DEFINITION_ERROR = """\
An asyncio pytest marker defines both "scope" and "loop_scope", \
but it should only use "loop_scope".
"""

_MARKER_SCOPE_KWARG_DEPRECATION_WARNING = """\
The "scope" keyword argument to the asyncio marker has been deprecated. \
Please use the "loop_scope" argument instead.
"""

_INVALID_LOOP_FACTORIES_KWARG = """\
mark.asyncio 'loop_factories' must be a non-empty sequence of strings.
"""

_EVENT_LOOP_POLICY_FIXTURE_DEPRECATION_WARNING = """\
Overriding the "event_loop_policy" fixture is deprecated \
and will be removed in a future version of pytest-asyncio. \
Use the "pytest_asyncio_loop_factories" hook to customize event loop creation.\
"""


def _parse_asyncio_marker(
    asyncio_marker: Mark,
) -> tuple[_ScopeName | None, Sequence[str] | None]:
    assert asyncio_marker.name == "asyncio"
    _validate_asyncio_marker(asyncio_marker)
    # The deprecation warning for the "scope" argument is emitted once per test
    # function by _warn_deprecated_scope_once, not here, because this function is
    # called for every parametrized item.
    if "scope" in asyncio_marker.kwargs and "loop_scope" in asyncio_marker.kwargs:
        raise pytest.UsageError(_DUPLICATE_LOOP_SCOPE_DEFINITION_ERROR)
    scope = asyncio_marker.kwargs.get("loop_scope") or asyncio_marker.kwargs.get(
        "scope"
    )
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
        and set(asyncio_marker.kwargs) - {"loop_scope", "scope", "loop_factories"}
    ):
        msg = (
            "mark.asyncio accepts only keyword arguments 'loop_scope' and"
            " 'loop_factories'."
        )
        raise ValueError(msg)


def _get_default_test_loop_scope(config: Config) -> Any:
    return config.getini("asyncio_default_test_loop_scope")


_RUNNER_TEARDOWN_WARNING = """\
An exception occurred during teardown of an asyncio.Runner. \
The reason is likely that you closed the underlying event loop in a test, \
which prevents the cleanup of asynchronous generators by the runner.
This warning will become an error in future versions of pytest-asyncio. \
Please ensure that your tests don't close the event loop. \
Here is the traceback of the exception triggered during teardown:
%s
"""


def _create_scoped_runner_fixture(scope: _ScopeName) -> Callable:
    @pytest.fixture(
        scope=scope,
        name=f"_{scope}_scoped_runner",
    )
    def _scoped_runner(
        event_loop_policy,
        _asyncio_loop_factory,
        request: FixtureRequest,
    ) -> Iterator[Runner]:
        new_loop_policy = event_loop_policy
        debug_mode = _get_asyncio_debug(request.config)
        with _temporary_event_loop_policy(new_loop_policy):
            runner = Runner(
                debug=debug_mode,
                loop_factory=_asyncio_loop_factory,
            ).__enter__()
            if _asyncio_loop_factory is not None:
                _set_event_loop(runner.get_loop())
            try:
                yield runner
            except Exception as e:
                runner.__exit__(type(e), e, e.__traceback__)
            else:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", ".*BaseEventLoop.shutdown_asyncgens.*", RuntimeWarning
                    )
                    try:
                        runner.__exit__(None, None, None)
                    except RuntimeError:
                        warnings.warn(
                            _RUNNER_TEARDOWN_WARNING % traceback.format_exc(),
                            RuntimeWarning,
                        )
            finally:
                if _asyncio_loop_factory is not None:
                    _set_event_loop(None)

    return _scoped_runner


for scope in Scope:
    globals()[f"_{scope.value}_scoped_runner"] = _create_scoped_runner_fixture(
        scope.value
    )


@pytest.fixture(scope="session")
def _asyncio_loop_factory(request: FixtureRequest) -> LoopFactory | None:
    return getattr(request, "param", None)


@pytest.fixture(scope="session", autouse=True)
def event_loop_policy() -> AbstractEventLoopPolicy:
    """Return an instance of the policy used to create asyncio event loops."""
    return _get_event_loop_policy()


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
