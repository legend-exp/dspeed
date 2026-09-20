import lgdo
import numpy as np
import os
import platformdirs
import subprocess
import sys
from pathlib import Path

import dspeed
from dspeed.utils import numba_defaults


def test_numba_defaults_loading():
    numba_defaults.cache = False
    numba_defaults.boundscheck = True

def test_cache_management(tmptestdir):
    # test precompile and clean cache

    # Test using NUMBA_CACHE
    nb_cache = Path(tmptestdir)/"test_cache"
    subprocess.run(
        ["dspeed-nbcache", "precompile"],
        env = os.environ | {"NUMBA_CACHE_DIR":nb_cache},
        capture_output=True,
        text=True,
        check=True,
    )

    # count cached numba files
    cached_procs = list(nb_cache.rglob("*.nb?"))
    assert len(cached_procs)>0

    # now clean the cache, and recount files
    subprocess.run(
        ["dspeed-nbcache", "clean"],
        env = os.environ | {"NUMBA_CACHE_DIR":nb_cache},
        capture_output=True,
        text=True,
        check=True,
    )
    cached_procs = list(nb_cache.rglob("*.nb?"))
    assert len(cached_procs)==0

    # Test using user cache; redirect home to tmp dir to do this
    user_cache = Path(tmptestdir)/platformdirs.user_cache_path().relative_to(Path.home())
    subprocess.run(
        ["dspeed-nbcache", "precompile"],
        env = os.environ | {
            "NUMBA_CACHE_DIR":"",
            "NUMBA_CACHE_LOCATOR_CLASSES":"UserWideCacheLocator",
            "HOME":str(tmptestdir)
        },
        capture_output=True,
        text=True,
        check=True,
    )

    # count cached numba files
    cached_procs = list(user_cache.rglob("*.nb?"))
    assert len(cached_procs)>0

    # now clean the cache, and recount files
    subprocess.run(
        ["dspeed-nbcache", "clean"],
        env = os.environ | {
            "NUMBA_CACHE_DIR":"",
            "NUMBA_CACHE_LOCATOR_CLASSES":"UserWideCacheLocator",
            "HOME":str(tmptestdir)
        },
        capture_output=True,
        text=True,
        check=True,
    )
    cached_procs = list(user_cache.rglob("*.nb?"))
    assert len(cached_procs)==0

    # now test in-tree
    tree_cache = Path(dspeed.__path__[0])

    # tree should start empty due to previous cleaning of cache
    cached_procs = list(tree_cache.rglob("*.nb?"))
    assert len(cached_procs)==0

    # repopulate in-tree cache
    subprocess.run(
        ["dspeed-nbcache", "precompile"],
        env = os.environ | {
            "NUMBA_CACHE_DIR":"",
            "NUMBA_CACHE_LOCATOR_CLASSES":"InTreeCacheLocator,InTreeCacheLocatorFsAgnostic",
            "HOME":str(tmptestdir)
        },
        capture_output=True,
        text=True,
        check=True,
    )
    cached_procs = list(tree_cache.rglob("*.nb?"))
    assert len(cached_procs)>0

def isclose(lhs, rhs, rtol=1e-5, atol=1e-8, equal_nan=True):
    # an is close comparison for LGDO structures

    if isinstance(lhs, lgdo.Struct) and isinstance(rhs, lgdo.Struct):
        if set(lhs) != set(rhs) or lhs.attrs != rhs.attrs:
            return False

        for k in lhs:
            if not isclose(lhs[k], rhs[k], rtol=rtol, atol=atol, equal_nan=equal_nan):
                return False
        return True

    elif isinstance(lhs, lgdo.Array) and isinstance(rhs, lgdo.Array):
        if len(lhs) != len(rhs) or lhs.attrs != rhs.attrs:
            return False
        return np.all(np.isclose(lhs, rhs, rtol=rtol, atol=atol, equal_nan=equal_nan))

    elif isinstance(lhs, lgdo.VectorOfVectors) and isinstance(
        rhs, lgdo.VectorOfVectors
    ):
        if len(lhs) != len(rhs) or lhs.attrs != rhs.attrs:
            return False
        return lhs.cumulative_length == rhs.cumulative_length and np.all(
            np.isclose(lhs, rhs, rtol=rtol, atol=atol, equal_nan=equal_nan)
        )

    return False
