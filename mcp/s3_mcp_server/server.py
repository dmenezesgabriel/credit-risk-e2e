# /// script
# dependencies = [
#   "mcp[cli]",
#   "aioboto3",
#   "python-dotenv",
# ]
# ///

import base64
import logging
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Optional

# Allow absolute imports when run as a script via `uv run`
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
from mcp.server.fastmcp import Context, FastMCP
from resources.s3_resource import S3Resource

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp_s3_server")


@dataclass
class AppContext:
    s3: S3Resource


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    max_buckets = int(os.getenv("S3_MAX_BUCKETS", "5"))
    s3 = S3Resource(
        region_name=os.getenv("AWS_REGION", "us-east-1"),
        max_buckets=max_buckets,
    )
    yield AppContext(s3=s3)


mcp = FastMCP("s3_service", lifespan=app_lifespan)


@mcp.tool()
async def list_buckets(ctx: Context, start_after: Optional[str] = None) -> str:
    """Returns a list of all buckets owned by the authenticated sender of the request.
    To grant IAM permission to use this operation, you must add the s3:ListAllMyBuckets policy action.
    """
    s3: S3Resource = ctx.request_context.lifespan_context.s3
    buckets = await s3.list_buckets(start_after)
    logger.info(f"list_buckets returning {len(buckets)} buckets")
    return str(buckets)


@mcp.tool()
async def list_objects_v2(
    ctx: Context,
    bucket_name: str,
    prefix: str = "",
    max_keys: int = 1000,
) -> str:
    """Returns some or all (up to 1,000) of the objects in a bucket with each request.
    You can use prefix as a selection criteria to return a subset of the objects in a bucket.
    """
    s3: S3Resource = ctx.request_context.lifespan_context.s3
    objects = await s3.list_objects(
        bucket_name, prefix=prefix, max_keys=max_keys
    )
    logger.info(f"list_objects_v2 returning {len(objects)} objects")
    return str(objects)


@mcp.tool()
async def get_object(
    ctx: Context,
    bucket_name: str,
    key: str,
    max_retries: int = 3,
) -> str:
    """Retrieves an object from Amazon S3. Returns text content for text files,
    base64-encoded content for binary files."""
    s3: S3Resource = ctx.request_context.lifespan_context.s3
    response = await s3.get_object(bucket_name, key, max_retries=max_retries)
    logger.info(f"get_object retrieved key {key} from {bucket_name}")
    data: bytes = response["Body"]
    if s3.is_text_file(key):
        return data.decode("utf-8")
    return base64.b64encode(data).decode("utf-8")


if __name__ == "__main__":
    mcp.run()
