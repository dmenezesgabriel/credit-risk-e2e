"""
sagemaker_session.py — SageMaker session factory
=================================================
Abstracts local vs production session setup so sm_pipeline.py
only needs to change one flag to switch environments.

Local mode:
    session = make_sagemaker_session(mode="local", ...)

Production (real AWS):
    session = make_sagemaker_session(mode="aws", ...)

Adding a new mode requires only writing a builder function and
registering it via create_session_registry() — no modification
to existing code.

In tests, create an isolated registry and register mock builders
without touching the default registry.
"""

import logging
from typing import TypeAlias

import boto3
import sagemaker
import sagemaker.local.image as sm_image
import sagemaker.local.utils as sm_local_utils
import yaml
from typing_extensions import Protocol, TypedDict, Unpack

logger = logging.getLogger("sagemaker_session")


SessionPair: TypeAlias = tuple[sagemaker.Session, boto3.Session]


class SessionBuilderKwargs(TypedDict, total=False):
    s3_bucket: str
    s3_endpoint: str | None
    network: str | None
    aws_region: str


class SessionBuilder(Protocol):
    def __call__(
        self, **kwargs: Unpack[SessionBuilderKwargs]
    ) -> SessionPair: ...


SessionRegistry: TypeAlias = dict[str, SessionBuilder]
SupportedMode: TypeAlias = str


def _apply_network_patch(network: str) -> None:
    """
    Injects an external Docker network into SageMaker's generated
    compose files so local containers can resolve service hostnames
    (localstack, mlflow, etc.).

    Caveat: patches a private SageMaker SDK internal (_SageMakerContainer._compose).
    Pin sagemaker SDK version in requirements.txt and treat upgrades as
    deliberate events that include verifying this patch still works.
    """
    _original_compose = sm_image._SageMakerContainer._compose

    def _patched_compose(self, *args, **kwargs):
        compose_cmd = _original_compose(self, *args, **kwargs)
        try:
            yaml_path = compose_cmd[compose_cmd.index("-f") + 1]
            with open(yaml_path) as f:
                data = yaml.safe_load(f)
            for svc in data.get("services", {}).values():
                svc.setdefault("networks", {})[network] = {}
            data.setdefault("networks", {})[network] = {
                "external": True,
                "name": network,
            }
            with open(yaml_path, "w") as f:
                yaml.dump(data, f, default_flow_style=False)
            logger.info(f"Injected network '{network}' into {yaml_path}")
        except Exception as e:
            logger.warning(f"Network patch failed: {e}")
        return compose_cmd

    sm_image._SageMakerContainer._compose = _patched_compose


def _apply_serving_host_patch() -> None:
    """
    Patch SageMaker local-mode host resolution so the SDK can reach
    serving containers (Batch Transform) from inside a Docker container.

    The SDK defaults to ``localhost`` for the serving endpoint, which
    fails in a container-in-container setup because ``localhost`` refers
    to the *caller* container, not the Docker host where the port is
    mapped.  We resolve the Docker host gateway IP from /proc/net/route
    instead, so ``gateway_ip:8080`` reaches the inference container via
    the port mapping.

    Caveat: same as _apply_network_patch — private SDK internal.
    """
    import sagemaker.local.entities as sm_entities
    import sagemaker.local.local_session as sm_local_session

    _original = getattr(sm_local_utils, "get_docker_host", None)
    if _original is None:
        logger.warning(
            "Cannot patch get_docker_host: function not found in "
            "sagemaker.local.utils — Batch Transform may not work"
        )
        return

    def _patched_get_docker_host() -> str:
        try:
            with open("/proc/net/route") as f:
                for line in f:
                    fields = line.strip().split()
                    if len(fields) >= 3 and fields[1] == "00000000":
                        hex_gw = fields[2]
                        ip = ".".join(
                            str(int(hex_gw[i : i + 2], 16))
                            for i in (6, 4, 2, 0)
                        )
                        logger.info(f"Resolved Docker host gateway: {ip}")
                        return ip
        except Exception as e:
            logger.warning(f"Gateway resolution failed: {e}")
        return _original()

    # Patch every module that imported get_docker_host by name
    sm_local_utils.get_docker_host = _patched_get_docker_host
    sm_entities.get_docker_host = _patched_get_docker_host
    sm_local_session.get_docker_host = _patched_get_docker_host


def _build_local_session(
    **kwargs: Unpack[SessionBuilderKwargs],
) -> SessionPair:
    s3_bucket: str = kwargs["s3_bucket"]
    s3_endpoint: str | None = kwargs.get("s3_endpoint")
    network: str | None = kwargs.get("network")
    aws_region: str = kwargs.get("aws_region", "us-east-1")

    if not s3_endpoint:
        raise ValueError("s3_endpoint is required for local mode")
    if not network:
        raise ValueError("network is required for local mode")

    boto_session = boto3.Session(
        aws_access_key_id="test",
        aws_secret_access_key="test",
        region_name=aws_region,
    )
    sagemaker_session = sagemaker.local.LocalSession(boto_session=boto_session)
    sagemaker_session.local_mode = True
    sagemaker_session._s3_client = boto_session.client(
        "s3", endpoint_url=s3_endpoint
    )
    sagemaker_session._sts_client = boto_session.client(
        "sts", endpoint_url=s3_endpoint
    )
    sagemaker_session.default_bucket = lambda: s3_bucket

    _apply_network_patch(network)
    _apply_serving_host_patch()

    logger.info(f"Local SageMaker session ready (bucket={s3_bucket})")
    return sagemaker_session, boto_session


def _build_aws_session(**kwargs: Unpack[SessionBuilderKwargs]) -> SessionPair:
    s3_bucket: str = kwargs["s3_bucket"]
    aws_region: str = kwargs.get("aws_region", "us-east-1")

    boto_session = boto3.Session(region_name=aws_region)
    sagemaker_session = sagemaker.Session(
        boto_session=boto_session,
        default_bucket=s3_bucket,
    )
    logger.info(f"AWS SageMaker session ready (bucket={s3_bucket})")
    return sagemaker_session, boto_session


def create_session_registry() -> SessionRegistry:
    """
    Returns a fresh registry pre-loaded with built-in session builders.

    Production: use the default via make_sagemaker_session() — no need
    to call this directly.

    Tests: call this to get an isolated registry, inject mock builders,
    and pass it to make_sagemaker_session(registry=...) without polluting
    the default registry or other tests.
    """
    return {
        "local": _build_local_session,
        "aws": _build_aws_session,
    }


def make_sagemaker_session(
    mode: SupportedMode,
    s3_bucket: str,
    s3_endpoint: str | None = None,
    network: str | None = None,
    aws_region: str = "us-east-1",
    registry: SessionRegistry | None = None,
) -> SessionPair:
    """
    Create and return (sagemaker_session, boto_session) for the given mode.

    Parameters
    ----------
    mode        : "local" or "aws" (or any mode registered in registry)
    s3_bucket   : Default S3 bucket for pipeline artifacts
    s3_endpoint : LocalStack endpoint URL — required for local mode
    network     : Docker network name to inject — required for local mode
    aws_region  : AWS region (default: us-east-1)
    registry    : Builder registry — defaults to create_session_registry().
                  Inject a custom registry in tests to use mock builders.

    Returns
    -------
    (sagemaker_session, boto_session)

    Raises
    ------
    ValueError  : unknown mode or missing required args for the chosen mode
    """
    active_registry: SessionRegistry = (
        registry if registry is not None else create_session_registry()
    )

    builder = active_registry.get(mode)
    if not builder:
        supported = ", ".join(f"'{m}'" for m in active_registry)
        raise ValueError(
            f"Unknown mode '{mode}'. Supported modes: {supported}"
        )

    return builder(
        s3_bucket=s3_bucket,
        s3_endpoint=s3_endpoint,
        network=network,
        aws_region=aws_region,
    )
