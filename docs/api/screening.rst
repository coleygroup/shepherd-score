Virtual Screening
=================

:mod:`shepherd_score.screen` aligns a query against a library of molecules too large
to hold in memory and returns the best-scoring hits. It takes two steps:

1. **Build a store, once.** A :class:`~shepherd_score.screen.ProfileStore` saves the
   arrays each library molecule needs for alignment to sharded files on disk.
2. **Screen it, any number of times.** :func:`~shepherd_score.screen.screen` streams the
   store past a query one shard at a time and keeps a running top-K, so memory use does
   not grow with library size.

Building a store
----------------

.. code-block:: python

   from shepherd_score.container import Molecule
   from shepherd_score.screen import ProfileStore

   with ProfileStore.create("library.fss", num_surf_points=200,
                            modes=("vol", "surf_esp")) as store:
       for i, rdmol in enumerate(library_rdkit_mols):
           store.add(Molecule(rdmol, num_surf_points=200), id=f"cmpd-{i}")

``modes`` lists the modes the store will be screened with. Only the arrays those modes
need are saved, so a ``("vol",)`` store is much smaller than a ``("surf_esp",)`` one, and
screening a store in a mode it was not built for raises an error
(:meth:`~shepherd_score.screen.ProfileStore.supports` checks this). Any mode in the
registry (``shepherd_score/accel/_modes.py``) can be used.

The defaults suit most libraries:

* ``pre_centered=True`` centers each molecule on its heavy-atom centroid. The fast and
  multi-GPU screening paths need it.
* ``canonical`` (on by default when a volume mode is requested) also rotates each molecule
  into its principal-axis frame, so volume-mode screens can share one set of starting
  orientations. Returned transforms are still in the molecule's own frame.
* ``dtype="float16"`` halves disk use at about 0.01 Å coordinate precision.

Every molecule in a store, and every query, must use the same ``num_surf_points`` for
the surface modes.

A store has a single writer. To build a library in parallel, give each worker its own
store (``library.part0.fss``, ``library.part1.fss``, ...), screen each part and merge the
hit lists.

Screening one query
-------------------

.. code-block:: python

   from shepherd_score.screen import ProfileStore, screen

   store = ProfileStore.open("library.fss")
   query = Molecule(query_rdmol, num_surf_points=200)
   hits = screen(query, store, mode="surf_esp", lam=0.3, top_k=100)

   for hit in hits[:5]:
       print(hit.id, hit.score)

Each result is a :data:`~shepherd_score.screen.Hit` ``(score, id, transform)``, best
first. ``id`` is the value passed to :meth:`~shepherd_score.screen.ProfileStore.add` (by
default the molecule's index in the store), and ``transform`` is the 4x4 SE(3) matrix that
moves the library molecule onto the query.

Mode parameters (``lam``, ``color_weight``, ``tversky_alpha``, ...) and optimizer settings
(``num_repeats``, ``max_num_steps``, ``lr``) are passed as keyword arguments; anything not
given takes the mode's registry default. ``alpha`` defaults to the value calibrated for
``num_surf_points`` in the surface modes.

``backend`` defaults to ``"triton"`` when a GPU is available and ``"numba"`` otherwise.

Several queries
---------------

:func:`~shepherd_score.screen.screen_many` gives the same results as calling
:func:`~shepherd_score.screen.screen` once per query, but reads the library from disk
only once:

.. code-block:: python

   from shepherd_score.screen import screen_many

   per_query_hits = screen_many([query_a, query_b], store, mode="vol", top_k=100)

Every score, not just the top-K
-------------------------------

Pass ``scores_out``, a preallocated ``(len(store),)`` array, to receive every library
molecule's score in store order. A memory-mapped array keeps this cheap for large
libraries:

.. code-block:: python

   import numpy as np

   scores = np.lib.format.open_memmap("scores.npy", mode="w+", dtype=np.float64,
                                      shape=(len(store),))
   hits = screen(query, store, mode="vol", scores_out=scores)

For :func:`~shepherd_score.screen.screen_many`, pass one array per query.

Multiple GPUs
-------------

``ndev=N`` spreads the shards over N GPUs, one worker process per GPU:

.. code-block:: python

   from shepherd_score.screen import close_multigpu_pool

   hits = screen(query, store, mode="vol", ndev=4)
   close_multigpu_pool()   # optional: free the GPUs before the interpreter exits

The workers start on the first multi-GPU call and are reused by later ones.
``worker_threads`` sets the CPU threads per worker (by default, the available cores
divided by ``ndev``). ``scores_out`` is not supported with ``ndev > 1``.

Reference
---------

.. autofunction:: shepherd_score.screen.screen

.. autofunction:: shepherd_score.screen.screen_many

.. autodata:: shepherd_score.screen.Hit
   :annotation:

.. autoclass:: shepherd_score.screen.ProfileStore
   :members: create, open, add, add_profile, close, supports, num_surf_points, modes,
             pre_centered, canonical, num_shards, schema
   :special-members: __len__

.. autofunction:: shepherd_score.screen.close_multigpu_pool

.. autoclass:: shepherd_score.screen.MoleculeProfile
   :members: from_molecule, center_to

Utilities
---------

.. autofunction:: shepherd_score.accel.clear_caches

.. autofunction:: shepherd_score.accel.has_triton

Shard-parallel CPU screening
----------------------------

A RAM-resident list of molecules screened across forked worker processes, one
core each; the pool is kept for later calls against the same library.

.. autofunction:: shepherd_score.accel.screen_parallel.screen_parallel

.. autofunction:: shepherd_score.accel.screen_parallel.screen_parallel_close

Multi-GPU alignment
-------------------

Data-parallel alignment of an in-memory list of pairs across every visible GPU. Also re-exported
from :mod:`shepherd_score.container`.

.. autofunction:: shepherd_score.accel.multi_gpu.align_multi_gpu

.. autoclass:: shepherd_score.accel.multi_gpu.MultiGPUAligner
   :members:
   :undoc-members:
