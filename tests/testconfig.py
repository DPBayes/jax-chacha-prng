# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2021 Aalto University

import numpy as np  # type: ignore
import jax

jax.config.update("jax_enable_x64", True)
np.set_printoptions(formatter={'int': hex})
