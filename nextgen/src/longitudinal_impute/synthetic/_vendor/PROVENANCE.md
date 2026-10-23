# Unmodified author source

These R files, DESCRIPTION and LICENSE.md were downloaded without modification
from https://github.com/amices/mice at commit
61083e667fa41cc4bc655492591b100bf8a7e443.
manifest.json records original URLs and SHA256 hashes, checked before execution.
See LICENSE.md for the upstream GNU GPL license and DESCRIPTION for authorship.

The executed functions belong to the Schouten et al. multivariate amputation
implementation. Inclusion of the upstream source is not a claim of reproducing
the package version used in 2018. This modern pinned implementation is explicitly
identified, and its seed/RNG/runtime are logged.

Only ampute's continuous MCAR/MAR path is exposed for count-data masks. Original
MNAR is invoked only in Table 1 reference designs. No discrete amputation,
alternative paper method, latent-p mechanism or random block algorithm is enabled.
