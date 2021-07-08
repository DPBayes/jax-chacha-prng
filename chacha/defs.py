# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2021 Aalto University

""" A JAX-accelerated implementation of the 20-round ChaCha cipher.

Basic definitions/constants.
"""

import jax.numpy as jnp
import numpy as np  # type: ignore

ChaChaStateShape = (4, 4)
ChaChaStateElementCount = np.prod(ChaChaStateShape)
ChaChaStateElementType = jnp.uint32
ChaChaStateElementBitWidth = jnp.iinfo(ChaChaStateElementType).bits
ChaChaStateBitSize = ChaChaStateElementCount * ChaChaStateElementBitWidth

ChaChaKeySizeInBits = 256
ChaChaKeySizeInBytes = ChaChaKeySizeInBits >> 3
ChaChaKeySizeInWords = ChaChaKeySizeInBytes >> 2

ChaChaNonceSizeInBits = 96
ChaChaNonceSizeInBytes = ChaChaNonceSizeInBits >> 3
ChaChaNonceSizeInWords = ChaChaNonceSizeInBytes >> 2

ChaChaCounterSizeInBits = 32
ChaChaCounterSizeInBytes = ChaChaCounterSizeInBits >> 3
ChaChaCounterSizeInWords = ChaChaCounterSizeInBytes >> 2
