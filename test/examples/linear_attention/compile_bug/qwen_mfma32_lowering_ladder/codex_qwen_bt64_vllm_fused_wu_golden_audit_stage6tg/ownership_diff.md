# Ownership Difference

Both normal paths use one CTA per `(chunk,value-head)`. F1's launch merge is therefore successful, but its CTA body remains four-way fragmentized and two-pass. Native vLLM uses one-pass MFMA32 tiles.
