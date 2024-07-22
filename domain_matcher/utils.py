import hashlib

import datasets
import orjson


class DisableHFTqdm:
    def __enter__(self):
        datasets.disable_progress_bar()

    def __exit__(self, exc_type, exc_val, exc_tb):
        datasets.enable_progress_bar()


def orjson_dumps(v, *, default=...):
    return orjson.dumps(
        v, default=default, option=orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY
    )


def md5_hash(d):
    """Hash any structure!
    No Security because we use it only to get a unique id.
    """
    return hashlib.md5(orjson_dumps(d)).hexdigest()  # nosec
